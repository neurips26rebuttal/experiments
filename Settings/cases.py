"""Canonical attack-case naming, shared by the dispatcher and the aggregators.

``caseN`` is the key used everywhere on disk (manifest rows, output dirs,
runtime/metrics JSON); ATTACK_NAMES maps it to the name the paper uses for
that method. Import this instead of restating the mapping so a rename (or a
new case) happens in exactly one place.
"""

ATTACK_NAMES = {
    "case1": "DGF-PGD (proposed)",
    "case2": "PGD",
    "case3": "F-PGD",
    "case4": "AutoAttack",
    "case5": "SSA",
    "case6": "AdvDrop",
}


def attack_name(case):
    """'case3', 'case3_robustbench' or '3' -> 'F-PGD'
    """
    key = str(case)
    if not key.startswith("case"):
        key = f"case{key}"
    root = key.split("_", 1)[0]
    return ATTACK_NAMES.get(root, str(case))
