from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import re
from itertools import combinations, product
import time

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Models
class MinimizeRequest(BaseModel):
    num_vars: int = Field(..., ge=2, le=4)
    input_mode: str = Field(default="minterm")  # minterm, maxterm, or expression
    minterms: List[int] = Field(default_factory=list)
    maxterms: List[int] = Field(default_factory=list)
    dont_cares: List[int] = Field(default_factory=list)
    expression: Optional[str] = None
    variable_names: List[str] = Field(default=["A", "B", "C", "D"])

class MinimizeResponse(BaseModel):
    truth_table: List[Dict[str, Any]]
    prime_implicants: List[Dict[str, Any]]
    essential_prime_implicants: List[str]
    minimal_sop: str
    minimal_pos: str
    canonical_sop: str
    canonical_pos: str
    groups: List[Dict[str, Any]]
    verilog_behavioral: str
    verilog_dataflow: str
    verilog_gate_level: str
    verilog_testbench: str
    simulation_output: str
    waveform_data: Dict[str, Any]
    steps: List[str]


# Boolean Expression Parser
class BooleanExpressionParser:
    def __init__(self, expression, var_names):
        self.expression = expression.upper().strip()
        self.var_names = [v.upper() for v in var_names]
        
    def parse_to_minterms(self, num_vars):
        """Parse boolean expression and return minterms"""
        # Normalize the expression
        expr = self.expression
        
        # Replace operators
        expr = expr.replace("'", "'")
        expr = expr.replace("^", "'")
        expr = expr.replace("¬", "'")
        expr = expr.replace("!", "'")
        expr = expr.replace("*", "")
        expr = expr.replace(".", "")
        
        # Generate all possible combinations
        minterms = []
        
        for i in range(2 ** num_vars):
            binary = format(i, f'0{num_vars}b')
            # Create variable assignments
            var_values = {self.var_names[j]: int(binary[j]) for j in range(num_vars)}
            
            # Evaluate expression
            if self.evaluate_expression(expr, var_values):
                minterms.append(i)
        
        return minterms
    
    def evaluate_expression(self, expr, var_values):
        """Evaluate boolean expression with given variable values"""
        try:
            # Replace variables with their values
            eval_expr = expr
            
            # Sort variables by length (descending) to avoid partial replacements
            sorted_vars = sorted(self.var_names, key=len, reverse=True)
            
            for var in sorted_vars:
                if var in eval_expr:
                    # Handle inverted variables
                    eval_expr = re.sub(f"{var}'", f"(not {var_values[var]})", eval_expr)
                    # Handle normal variables
                    eval_expr = re.sub(f"(?<!not ){var}(?!')", str(var_values[var]), eval_expr)
            
            # Replace boolean operators
            eval_expr = eval_expr.replace("+", " or ")
            # Implicit AND (adjacent terms)
            eval_expr = re.sub(r'(\d)\s*\(', r'\1 and (', eval_expr)
            eval_expr = re.sub(r'\)\s*(\d)', r') and \1', eval_expr)
            eval_expr = re.sub(r'(\d)\s+(\d)', r'\1 and \2', eval_expr)
            
            # Evaluate
            result = eval(eval_expr)
            return bool(result)
        except:
            return False


# Quine-McCluskey Implementation
class QuineMcCluskey:
    def __init__(self, num_vars, minterms, dont_cares=[]):
        self.num_vars = num_vars
        self.minterms = sorted(set(minterms))
        self.dont_cares = sorted(set(dont_cares))
        self.all_terms = sorted(set(minterms + dont_cares))
        self.steps = []
        
    def dec_to_bin(self, num):
        return format(num, f'0{self.num_vars}b')
    
    def count_ones(self, binary):
        return binary.count('1')
    
    def can_combine(self, term1, term2):
        diff_count = 0
        diff_pos = -1
        for i in range(len(term1)):
            if term1[i] != term2[i]:
                if term1[i] == '-' or term2[i] == '-':
                    return False, -1
                diff_count += 1
                diff_pos = i
        return diff_count == 1, diff_pos
    
    def combine_terms(self, term1, term2, pos):
        result = list(term1)
        result[pos] = '-'
        return ''.join(result)
    
    def find_prime_implicants(self):
        if not self.all_terms:
            return []
        
        # Convert to binary
        terms = {term: self.dec_to_bin(term) for term in self.all_terms}
        
        # Group by number of ones
        groups = {}
        for num, binary in terms.items():
            ones = self.count_ones(binary)
            if ones not in groups:
                groups[ones] = []
            groups[ones].append((binary, [num]))
        
        self.steps.append(f"Initial grouping by number of 1s: {len(groups)} groups")
        
        prime_implicants = []
        used = set()
        
        while groups:
            new_groups = {}
            current_used = set()
            
            sorted_keys = sorted(groups.keys())
            for i in range(len(sorted_keys) - 1):
                key1, key2 = sorted_keys[i], sorted_keys[i + 1]
                
                for term1, mints1 in groups[key1]:
                    for term2, mints2 in groups[key2]:
                        can_comb, pos = self.can_combine(term1, term2)
                        if can_comb:
                            combined = self.combine_terms(term1, term2, pos)
                            combined_mints = sorted(set(mints1 + mints2))
                            
                            current_used.add(term1)
                            current_used.add(term2)
                            
                            ones = self.count_ones(combined)
                            if ones not in new_groups:
                                new_groups[ones] = []
                            
                            # Avoid duplicates
                            if (combined, tuple(combined_mints)) not in [(t, tuple(m)) for t, m in new_groups[ones]]:
                                new_groups[ones].append((combined, combined_mints))
            
            # Add unused terms as prime implicants
            for key in groups:
                for term, mints in groups[key]:
                    if term not in current_used and term not in used:
                        prime_implicants.append((term, mints))
                        used.add(term)
            
            groups = new_groups
            if new_groups:
                self.steps.append(f"Combined terms, created {sum(len(v) for v in new_groups.values())} new terms")
        
        self.steps.append(f"Found {len(prime_implicants)} prime implicants")
        return prime_implicants
    
    def find_essential_prime_implicants(self, prime_implicants):
        if not self.minterms:
            return [], []
        
        # Create coverage table
        coverage = {mint: [] for mint in self.minterms}
        for i, (term, mints) in enumerate(prime_implicants):
            for mint in mints:
                if mint in coverage:
                    coverage[mint].append(i)
        
        essential = set()
        for mint, covering_pis in coverage.items():
            if len(covering_pis) == 1:
                essential.add(covering_pis[0])
        
        essential_pis = [prime_implicants[i] for i in essential]
        
        # Find minimal cover
        covered = set()
        for term, mints in essential_pis:
            covered.update([m for m in mints if m in self.minterms])
        
        remaining = [pi for i, pi in enumerate(prime_implicants) if i not in essential]
        
        # Greedy selection for remaining minterms
        selected = list(essential_pis)
        uncovered = set(self.minterms) - covered
        
        while uncovered and remaining:
            best_pi = max(remaining, key=lambda pi: len([m for m in pi[1] if m in uncovered]))
            selected.append(best_pi)
            uncovered -= set([m for m in best_pi[1] if m in self.minterms])
            remaining.remove(best_pi)
        
        self.steps.append(f"Found {len(essential_pis)} essential prime implicants")
        return essential_pis, selected
    
    def term_to_expression(self, term, var_names):
        expr = []
        for i, bit in enumerate(term):
            if bit == '1':
                expr.append(var_names[i])
            elif bit == '0':
                expr.append(var_names[i] + "'")
        return ''.join(expr) if expr else '1'
    
    def minimize(self, var_names):
        prime_implicants = self.find_prime_implicants()
        essential_pis, selected_pis = self.find_essential_prime_implicants(prime_implicants)
        
        if not selected_pis:
            return "0", [], [], []
        
        expression_terms = [self.term_to_expression(term, var_names) for term, _ in selected_pis]
        expression = ' + '.join(expression_terms)
        
        return expression, prime_implicants, essential_pis, selected_pis


def maxterms_to_minterms(maxterms, num_vars):
    """Convert maxterms to minterms"""
    all_terms = set(range(2 ** num_vars))
    minterms = list(all_terms - set(maxterms))
    return sorted(minterms)


def generate_canonical_sop(minterms, num_vars, var_names):
    """Generate canonical Sum of Products"""
    if not minterms:
        return "0"
    
    terms = []
    for m in minterms:
        binary = format(m, f'0{num_vars}b')
        term = ''.join([var_names[i] if binary[i] == '1' else var_names[i] + "'" 
                       for i in range(num_vars)])
        terms.append(term)
    
    return ' + '.join(terms)


def generate_canonical_pos(maxterms, num_vars, var_names):
    """Generate canonical Product of Sums"""
    if not maxterms:
        return "1"
    
    terms = []
    for m in maxterms:
        binary = format(m, f'0{num_vars}b')
        term_parts = [var_names[i] if binary[i] == '0' else var_names[i] + "'" 
                     for i in range(num_vars)]
        terms.append('(' + ' + '.join(term_parts) + ')')
    
    return ''.join(terms)


def generate_minimal_pos(maxterms, num_vars, var_names, dont_cares=[]):
    """Generate minimal POS using Quine-McCluskey on maxterms"""
    if not maxterms:
        return "1"
    
    qm = QuineMcCluskey(num_vars, maxterms, dont_cares)
    prime_implicants = qm.find_prime_implicants()
    essential_pis, selected_pis = qm.find_essential_prime_implicants(prime_implicants)
    
    if not selected_pis:
        return "1"
    
    expression_terms = []
    for term, _ in selected_pis:
        term_parts = []
        for i, bit in enumerate(term):
            if bit == '0':
                term_parts.append(var_names[i])
            elif bit == '1':
                term_parts.append(var_names[i] + "'")
        if term_parts:
            expression_terms.append('(' + ' + '.join(term_parts) + ')')
    
    return ''.join(expression_terms) if expression_terms else "1"


def generate_truth_table(num_vars, minterms, dont_cares, var_names):
    table = []
    for i in range(2 ** num_vars):
        binary = format(i, f'0{num_vars}b')
        row = {var_names[j]: int(binary[j]) for j in range(num_vars)}
        
        if i in minterms:
            row['F'] = 1
        elif i in dont_cares:
            row['F'] = 'X'
        else:
            row['F'] = 0
        
        row['minterm'] = i
        table.append(row)
    
    return table


def generate_waveform_data(truth_table, num_vars, var_names):
    """Generate GTKWave-style waveform data"""
    signals = {}
    
    # Initialize signals
    for var in var_names[:num_vars]:
        signals[var] = []
    signals['F'] = []
    
    # Generate waveform for each time step
    for i, row in enumerate(truth_table):
        for var in var_names[:num_vars]:
            signals[var].append(row[var])
        signals['F'].append(1 if row['F'] == 1 else (0 if row['F'] == 0 else 0))
    
    return {
        "signals": signals,
        "time_steps": len(truth_table),
        "signal_names": var_names[:num_vars] + ['F']
    }


def generate_verilog_behavioral(expression, num_vars, var_names):
    verilog_expr = expression.replace("'", "_n").replace(" + ", " | ").replace(" ", " & ")
    for var in var_names[:num_vars]:
        verilog_expr = verilog_expr.replace(var + "_n", f"~{var}")
    
    inputs = ', '.join(var_names[:num_vars])
    
    code = f"""module kmap_behavioral(
    input {inputs},
    output reg F
);

always @(*) begin
    F = {verilog_expr};
end

endmodule"""
    
    return code


def generate_verilog_dataflow(expression, num_vars, var_names):
    verilog_expr = expression.replace("'", "_n").replace(" + ", " | ").replace(" ", " & ")
    for var in var_names[:num_vars]:
        verilog_expr = verilog_expr.replace(var + "_n", f"~{var}")
    
    inputs = ', '.join(var_names[:num_vars])
    
    code = f"""module kmap_dataflow(
    input {inputs},
    output F
);

    assign F = {verilog_expr};

endmodule"""
    
    return code


def generate_verilog_gate_level(selected_pis, num_vars, var_names):
    inputs = ', '.join(var_names[:num_vars])
    
    wires = []
    gates = []
    not_wires = []
    
    # Generate NOT gates for inverted inputs
    for i, var in enumerate(var_names[:num_vars]):
        not_wires.append(f"{var}_n")
        gates.append(f"    not n{i}({var}_n, {var});")
    
    # Generate AND gates for each product term
    for idx, (term, mints) in enumerate(selected_pis):
        wire_name = f"term{idx}"
        wires.append(wire_name)
        
        and_inputs = []
        for i, bit in enumerate(term):
            if bit == '1':
                and_inputs.append(var_names[i])
            elif bit == '0':
                and_inputs.append(f"{var_names[i]}_n")
        
        if len(and_inputs) == 0:
            gates.append(f"    assign {wire_name} = 1'b1;")
        elif len(and_inputs) == 1:
            gates.append(f"    assign {wire_name} = {and_inputs[0]};")
        elif len(and_inputs) == 2:
            gates.append(f"    and a{idx}({wire_name}, {', '.join(and_inputs)});")
        else:
            gates.append(f"    and a{idx}({wire_name}, {', '.join(and_inputs)});")
    
    # OR gate for final output
    if len(wires) == 0:
        or_gate = "    assign F = 1'b0;"
    elif len(wires) == 1:
        or_gate = f"    assign F = {wires[0]};"
    elif len(wires) == 2:
        or_gate = f"    or o1(F, {', '.join(wires)});"
    else:
        or_gate = f"    or o1(F, {', '.join(wires)});"
    
    not_wire_decl = f"    wire {', '.join(not_wires)};" if not_wires else ""
    wire_decl = f"    wire {', '.join(wires)};" if wires else ""
    
    code = f"""module kmap_gate_level(
    input {inputs},
    output F
);

{not_wire_decl}
{wire_decl}

{chr(10).join(gates)}
{or_gate}

endmodule"""
    
    return code


def generate_verilog_testbench(num_vars, var_names, truth_table):
    inputs = ', '.join(var_names[:num_vars])
    
    test_cases = []
    for row in truth_table:
        input_vals = ''.join([str(row[var]) for var in var_names[:num_vars]])
        output_val = '1' if row['F'] == 1 else ('x' if row['F'] == 'X' else '0')
        test_cases.append(f"        {{{len(var_names[:num_vars])}'b{input_vals}, 1'b{output_val}}}")
    
    code = f"""module kmap_tb;
    reg {', '.join(var_names[:num_vars])};
    wire F;
    
    // Instantiate the design under test
    kmap_dataflow dut(
        {', '.join([f'.{v}({v})' for v in var_names[:num_vars]])},
        .F(F)
    );
    
    integer i;
    reg [{num_vars}:0] test_vectors [{len(truth_table)-1}:0];
    
    initial begin
        $dumpfile(\"kmap.vcd\");
        $dumpvars(0, kmap_tb);
        
        // Initialize test vectors
{chr(10).join(test_cases)};
        
        // Apply test vectors
        for (i = 0; i < {len(truth_table)}; i = i + 1) begin
            {{{', '.join(var_names[:num_vars])}}} = test_vectors[i][{num_vars}:1];
            #10;
            $display(\"{' '.join(['%b' for _ in var_names[:num_vars]])} | F=%b (expected=%b)\",
                {', '.join(var_names[:num_vars])}, F, test_vectors[i][0]);
        end
        
        $finish;
    end
endmodule"""
    
    return code


def generate_simulation_output(truth_table, num_vars, var_names):
    output_lines = ["VVP Simulation Output:"]
    output_lines.append("=" * 50)
    output_lines.append(f"{' '.join(var_names[:num_vars])} | F | Expected")
    output_lines.append("-" * 50)
    
    for row in truth_table:
        input_vals = ' '.join([str(row[var]) for var in var_names[:num_vars]])
        output_val = row['F']
        expected = '1' if output_val == 1 else ('X' if output_val == 'X' else '0')
        status = "✓" if output_val != 'X' else "(don't care)"
        output_lines.append(f"{input_vals} | {output_val} | {expected} {status}")
    
    output_lines.append("=" * 50)
    output_lines.append("Simulation completed successfully!")
    
    return '\n'.join(output_lines)


def generate_kmap_groups(selected_pis, num_vars):
    groups = []
    for idx, (term, mints) in enumerate(selected_pis):
        groups.append({
            "id": idx,
            "cells": mints,
            "term": term,
            "color": f"hsl({(idx * 60) % 360}, 70%, 60%)"
        })
    return groups


@api_router.post("/minimize", response_model=MinimizeResponse)
async def minimize_kmap(request: MinimizeRequest):
    try:
        # Process based on input mode
        if request.input_mode == "expression":
            # Parse Boolean expression
            parser = BooleanExpressionParser(request.expression, request.variable_names)
            minterms = parser.parse_to_minterms(request.num_vars)
            all_terms = set(range(2 ** request.num_vars))
            maxterms = list(all_terms - set(minterms) - set(request.dont_cares))
        elif request.input_mode == "maxterm":
            # Convert maxterms to minterms
            all_terms = set(range(2 ** request.num_vars))
            minterms = list(all_terms - set(request.maxterms) - set(request.dont_cares))
            maxterms = request.maxterms
        else:  # minterm mode
            minterms = request.minterms
            all_terms = set(range(2 ** request.num_vars))
            maxterms = list(all_terms - set(minterms) - set(request.dont_cares))
        
        # Validate inputs
        max_val = 2 ** request.num_vars
        if any(m >= max_val for m in minterms + maxterms + request.dont_cares):
            raise HTTPException(400, "Term values exceed variable range")
        
        var_names = request.variable_names[:request.num_vars]
        
        # Run Quine-McCluskey for SOP
        qm = QuineMcCluskey(request.num_vars, minterms, request.dont_cares)
        start_time = time.perf_counter()
        minimal_sop, prime_implicants, essential_pis, selected_pis = qm.minimize(var_names)
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        logger.info(
            f"[PERF] Vars={request.num_vars}, "
            f"Mode={request.input_mode}, "
            f"Minterms={minterms}, "
            f"DontCares={request.dont_cares}, "
            f"QM_Time={execution_time_ms:.3f} ms"
        )

        # Generate canonical forms
        canonical_sop = generate_canonical_sop(minterms, request.num_vars, var_names)
        canonical_pos = generate_canonical_pos(maxterms, request.num_vars, var_names)
        
        # Generate minimal POS
        minimal_pos = generate_minimal_pos(maxterms, request.num_vars, var_names, request.dont_cares)
        
        # Generate outputs
        truth_table = generate_truth_table(request.num_vars, minterms, request.dont_cares, var_names)
        
        pi_list = [{
            "term": pi[0],
            "minterms": pi[1],
            "expression": qm.term_to_expression(pi[0], var_names),
            "essential": pi in essential_pis
        } for pi in prime_implicants]
        
        essential_pi_exprs = [qm.term_to_expression(pi[0], var_names) for pi in essential_pis]
        
        groups = generate_kmap_groups(selected_pis, request.num_vars)
        
        verilog_behavioral = generate_verilog_behavioral(minimal_sop, request.num_vars, var_names)
        verilog_dataflow = generate_verilog_dataflow(minimal_sop, request.num_vars, var_names)
        verilog_gate_level = generate_verilog_gate_level(selected_pis, request.num_vars, var_names)
        verilog_testbench = generate_verilog_testbench(request.num_vars, var_names, truth_table)
        simulation_output = generate_simulation_output(truth_table, request.num_vars, var_names)
        waveform_data = generate_waveform_data(truth_table, request.num_vars, var_names)
        
        return MinimizeResponse(
            truth_table=truth_table,
            prime_implicants=pi_list,
            essential_prime_implicants=essential_pi_exprs,
            minimal_sop=minimal_sop,
            minimal_pos=minimal_pos,
            canonical_sop=canonical_sop,
            canonical_pos=canonical_pos,
            groups=groups,
            verilog_behavioral=verilog_behavioral,
            verilog_dataflow=verilog_dataflow,
            verilog_gate_level=verilog_gate_level,
            verilog_testbench=verilog_testbench,
            simulation_output=simulation_output,
            waveform_data=waveform_data,
            steps=qm.steps
        )
    
    except Exception as e:
        logging.error(f"Minimization error: {str(e)}")
        raise HTTPException(500, str(e))


@api_router.get("/")
async def root():
    return {"message": "K-Map Minimizer API"}


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
