"""Naming and parsing of the flat results tree.
"""
import os
import re

RUN_DIR_FIELDS = ("dataset", "case", "group", "eps", "hparams")


def format_eps_tag(epsilon):
    """0.0156862745 -> '4div255'. The readable form, and filename-safe."""
    return f"{round(epsilon * 255)}div255"


def hparam_subpath(a=None, b=None, window=None, gamma=None,
                   num_steps=None, aa_version=None, aa_norm=None):
    """The hyperparameter part of a run's directory NAME, not a path.
    """
    parts = []
    if a is not None and b is not None:
        parts.append(f"a{a}_b{b}")
    if window is not None:
        parts.append(window)
    if gamma is not None:
        parts.append(f"gamma{gamma}")
    if num_steps is not None:
        parts.append(f"steps{num_steps}")
    if aa_version is not None and aa_norm is not None:
        parts.append(f"aa_{aa_version}_{aa_norm}")
    return "_".join(parts)


def case_hparam_token(case, args, a=None, b=None, window=None, gamma=None,
                      num_steps=None):
    """The hyperparameter token for `case`, covering all six methods.
    """
    if case == "case1":
        return hparam_subpath(a=a, b=b, window=window, gamma=gamma,
                              num_steps=num_steps)
    if case in ("case2", "case3"):
        return hparam_subpath(num_steps=num_steps)
    if case == "case4":
        return hparam_subpath(aa_version=args.aa_version, aa_norm=args.aa_norm)
    if case == "case5":
        steps = args.ssa_steps or args.num_steps
        parts = [f"N{args.ssa_N}", f"rho{args.ssa_rho}",
                 f"sigma{args.ssa_sigma}", f"steps{steps}"]
        if float(args.ssa_momentum) != 0.0:
            parts.append(f"mom{args.ssa_momentum}")
        return "_".join(parts)
    if case == "case6":
        parts = [f"q{args.advdrop_q_size}", f"block{args.advdrop_block_size}",
                 f"steps{args.advdrop_steps}"]
        if float(args.advdrop_lr) != 0.01:
            parts.append(f"lr{args.advdrop_lr}")
        return "_".join(parts)
    return ""


def run_dir_name(dataset, group_name, case, eps_tag, hp_subpath):
    """The single directory name identifying one (case, model family) run."""
    parts = [dataset, case, group_name, f"eps{eps_tag}"]
    if hp_subpath:
        parts.append(hp_subpath)
    return "_".join(parts)


def parse_run_dir(name):
    """Inverse of run_dir_name(): a directory name -> its fields.
    """
    toks = str(name).split("_")

    case = next((t for t in toks if re.fullmatch(r"case\d+", t)), "")
    if not case:
        return dict.fromkeys(RUN_DIR_FIELDS, "")
    i_case = toks.index(case)
    i_eps = next((i for i, t in enumerate(toks) if t.startswith("eps")), -1)

    return {
        "dataset": "_".join(toks[:i_case]),
        "case": case,
        # everything between the case and the epsilon is the model family
        "group": "_".join(toks[i_case + 1:i_eps]) if i_case < i_eps else "",
        "eps": toks[i_eps][len("eps"):] if i_eps >= 0 else "",
        "hparams": "_".join(toks[i_eps + 1:]) if i_eps >= 0 else "",
    }


def parse_run_path(path):
    """parse_run_dir() for a path pointing at (or inside) a run directory.
    """
    for part in reversed(os.path.normpath(path).split(os.sep)):
        meta = parse_run_dir(part)
        if meta["case"]:
            return meta
    return dict.fromkeys(RUN_DIR_FIELDS, "")
