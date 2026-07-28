"""Naming and parsing of the flat results tree. Imported by everything.

The results tree is FLAT: one directory per run, directly under the results
root, whose NAME carries every field that identifies it --

    results/imagenet_case1_standard_eps4div255_a1_b112_Hann_gamma0.1_steps20/
    results/cifar100_case5_pretrained_eps8div255_N20_rho0.5_sigma16.0_steps10/

so `ls results/` is the index of a sweep, no two runs can collide, and the
aggregators rebuild their tables from the names alone.

Both directions live here, next to each other, because they are inverses and a
change to one is a bug unless it is a change to both:

    run_dir_name()   the writers (eval_imagenet.py, eval_cifar100.py)
    parse_run_dir()  the readers (aggregate_run.py,
                     build_transferability_table.py)

Deliberately dependency-free -- stdlib only, no torch. The aggregators are
pure-Python post-processing and must not have to import a 70k-line eval script
(and through it torch, lpips, matplotlib) just to split a directory name.
"""
import os
import re

#: Fields of a run directory name, in the order they appear.
RUN_DIR_FIELDS = ("dataset", "case", "group", "eps", "hparams")


def format_eps_tag(epsilon):
    """0.0156862745 -> '4div255'. The readable form, and filename-safe."""
    return f"{round(epsilon * 255)}div255"


def hparam_subpath(a=None, b=None, window=None, gamma=None,
                   num_steps=None, aa_version=None, aa_norm=None):
    """The hyperparameter part of a run's directory NAME, not a path.

    Joined with '_' into one token rather than one directory level per knob:
    the tree is flat, so a run occupies exactly one directory. Only the knobs
    the method actually reads are passed in, so a directory name never claims a
    parameter the run ignored.
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

    One implementation for both datasets, so a directory name means the same
    thing wherever it was written.

    Cases 1-3 take their values as ARGUMENTS because eval_imagenet.py sweeps
    them; cases 4-6 read their fixed knobs straight off `args`. Only the knobs a
    method actually consumes appear, so a name never advertises a parameter the
    run ignored: cases 2 and 3 derive their step size as eps/num_steps, which is
    why a/b/window/gamma are absent from theirs.

    Non-default baseline knobs are APPENDED rather than always present, so a
    default run keeps the directory it already has and its existing checkpoints
    stay resumable (save_dir IS checkpoint_dir; see evaluate_all_transferability
    in eval_imagenet.py).
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

    Anchored on the two tokens with a FIXED shape -- the case is `caseN`, the
    epsilon is the token starting `eps` -- rather than on position. Splitting
    positionally would break on every name whose dataset or model family
    contains an underscore, which is most of them: `cifar100` survives, but
    `robustbench` vs a future `adv_trained` would shift every later field by
    one and silently mislabel the whole table.

    A name that carries no `caseN` token is not a run directory; every field
    comes back empty rather than guessed, so a stray folder under the results
    root is skipped instead of aggregated as a phantom configuration.
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

    Walks up from the file to the first component that parses as a run
    directory, so both `<run>/results.json` and `<run>/images/x.png` resolve to
    the same run.
    """
    for part in reversed(os.path.normpath(path).split(os.sep)):
        meta = parse_run_dir(part)
        if meta["case"]:
            return meta
    return dict.fromkeys(RUN_DIR_FIELDS, "")
