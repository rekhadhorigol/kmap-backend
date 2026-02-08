"""Test the BitSliceQuineMcCluskey from server.py directly."""
import sys
import time
import importlib.util
import os

# Load server module without starting FastAPI
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("DB_NAME", "test_db")

# Extract just the algorithm class via exec to avoid MongoDB connection
with open("server.py", "r") as f:
    source = f.read()

# Find the class definition and supporting code
import re as _re
from itertools import combinations, product

exec_globals = {"re": _re, "combinations": combinations, "product": product}
exec_locals = {}

# Execute just the algorithm classes
class_code = []
recording = False
brace_depth = 0

lines = source.split("\n")
i = 0
while i < len(lines):
    line = lines[i]
    if "class BitSliceQuineMcCluskey" in line or "class QuineMcCluskey" in line:
        recording = True
    if recording:
        class_code.append(line)
        if line.strip() and not line.strip().startswith("#") and not line.strip().startswith('"""'):
            pass
        # Stop recording if we hit a top-level def/class that's not part of the class
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if (next_line and not next_line.startswith(" ") and not next_line.startswith("\t")
                and not next_line.strip() == "" and not next_line.strip().startswith("#")
                and "class QuineMcCluskey" not in next_line):
                recording = False
    i += 1

exec("\n".join(class_code), exec_globals, exec_locals)
BitSliceQuineMcCluskey = exec_locals.get("BitSliceQuineMcCluskey")
QuineMcCluskey = exec_locals.get("QuineMcCluskey", BitSliceQuineMcCluskey)


def verify_coverage(num_vars, minterms, selected_compat):
    """Verify that selected PIs actually cover all minterms."""
    covered = set()
    for term_str, mint_list in selected_compat:
        covered.update(mint_list)
    minterm_set = set(minterms)
    missing = minterm_set - covered
    if missing:
        print(f"  FAIL: Minterms not covered: {sorted(missing)[:20]}...")
        return False
    print(f"  PASS: All {len(minterm_set)} minterms covered by {len(selected_compat)} PIs")
    return True


def test_basic_3var():
    print("=== Test 1: 3 vars, minterms [0,2,5,7] ===")
    qm = QuineMcCluskey(3, [0, 2, 5, 7])
    expr, pis, epis, spis = qm.minimize(["A", "B", "C"])
    print(f"  Expression: F = {expr}")
    print(f"  PIs: {[(t, m) for t, m in pis]}")
    assert verify_coverage(3, [0, 2, 5, 7], spis)


def test_basic_4var():
    print("=== Test 2: 4 vars, minterms [0,1,2,5,6,7,8,9,14] ===")
    qm = QuineMcCluskey(4, [0, 1, 2, 5, 6, 7, 8, 9, 14])
    expr, pis, epis, spis = qm.minimize(["A", "B", "C", "D"])
    print(f"  Expression: F = {expr}")
    print(f"  Total PIs: {len(pis)}, Selected: {len(spis)}")
    assert verify_coverage(4, [0, 1, 2, 5, 6, 7, 8, 9, 14], spis)


def test_dont_cares():
    print("=== Test 3: 4 vars, minterms [1,3,5,7], dont_cares [0,2] ===")
    qm = QuineMcCluskey(4, [1, 3, 5, 7], [0, 2])
    expr, pis, epis, spis = qm.minimize(["A", "B", "C", "D"])
    print(f"  Expression: F = {expr}")
    assert verify_coverage(4, [1, 3, 5, 7], spis)


def test_term_to_expression_compat():
    print("=== Test 4: term_to_expression compatibility ===")
    qm = QuineMcCluskey(3, [0])
    result1 = qm.term_to_expression("10-", ["A", "B", "C"])
    result2 = qm.term_to_expression("0-0", ["A", "B", "C"])
    result3 = qm.term_to_expression("---", ["A", "B", "C"])
    print(f"  '10-' -> {result1}")
    print(f"  '0-0' -> {result2}")
    print(f"  '---' -> {result3}")
    assert result1 == "AB'", f"Expected AB', got {result1}"
    assert result2 == "A'C'", f"Expected A'C', got {result2}"
    assert result3 == "1", f"Expected 1, got {result3}"
    print("  PASS")


def test_perf_8var():
    print("=== Test 5: 8 vars performance (80 minterms) ===")
    import random
    random.seed(42)
    minterms = random.sample(range(256), 80)
    t0 = time.perf_counter()
    qm = QuineMcCluskey(8, minterms)
    expr, pis, epis, spis = qm.minimize(["A", "B", "C", "D", "E", "F", "G", "H"])
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Time: {elapsed:.2f}ms")
    print(f"  PIs: {len(pis)}, Selected: {len(spis)}")
    assert verify_coverage(8, minterms, spis)


def test_perf_10var():
    print("=== Test 6: 10 vars performance (200 minterms) ===")
    import random
    random.seed(123)
    minterms = random.sample(range(1024), 200)
    t0 = time.perf_counter()
    qm = QuineMcCluskey(10, minterms)
    expr, pis, epis, spis = qm.minimize(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Time: {elapsed:.2f}ms")
    print(f"  PIs: {len(pis)}, Selected: {len(spis)}")
    assert verify_coverage(10, minterms, spis)


def test_edge_all_minterms():
    print("=== Test 7: All minterms (F=1) ===")
    qm = QuineMcCluskey(3, [0, 1, 2, 3, 4, 5, 6, 7])
    expr, pis, epis, spis = qm.minimize(["A", "B", "C"])
    print(f"  Expression: F = {expr}")
    assert expr == "1", f"Expected '1', got '{expr}'"
    print("  PASS")


def test_edge_single():
    print("=== Test 8: Single minterm ===")
    qm = QuineMcCluskey(3, [5])
    expr, pis, epis, spis = qm.minimize(["A", "B", "C"])
    print(f"  Expression: F = {expr}")
    assert verify_coverage(3, [5], spis)


def test_edge_empty():
    print("=== Test 9: No minterms ===")
    qm = QuineMcCluskey(3, [])
    expr, pis, epis, spis = qm.minimize(["A", "B", "C"])
    print(f"  Expression: F = {expr}")
    assert expr == "0", f"Expected '0', got '{expr}'"
    print("  PASS")


if __name__ == "__main__":
    all_passed = True
    tests = [
        test_basic_3var,
        test_basic_4var,
        test_dont_cares,
        test_term_to_expression_compat,
        test_perf_8var,
        test_perf_10var,
        test_edge_all_minterms,
        test_edge_single,
        test_edge_empty,
    ]
    for test in tests:
        try:
            test()
            print()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {e}")
            all_passed = False
            print()

    if all_passed:
        print("=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
    else:
        print("=" * 50)
        print("SOME TESTS FAILED")
        print("=" * 50)
        sys.exit(1)
