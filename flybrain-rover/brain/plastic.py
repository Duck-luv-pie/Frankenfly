"""
plastic.py -- demo control: wipe the fly's learned synapses back to the raw connectome, or restore them.

    snap = lobotomize(lif)                    # reset every synapse to lif.w0 ("the fly forgot everything")
    restore(lif, snap)                        # undo it, bit-for-bit

For a more dramatic cut, pass zero_lc10a (a LongTensor of LC10a neuron indices, both sides) and it also
lesions the fly's motion-detecting visual channel (lif.lesion(zero_lc10a, mode="both")) -- "the fly can't
see." restore() undoes both the weight reset and the lesion, in the right order, exactly.

Demo use: mid-demo, lobotomize a trained checkpoint to show the naive (or blind) brain next to the
trained one, then restore to bring the learned behavior back for the finale.

CLI (works on saved checkpoints, no running training loop needed):
    python -m brain.plastic --lobotomize --checkpoint checkpoints/X.pt \
        [--out checkpoints/X_lobotomized.pt] [--zero-lc10a] [--brain data/brain.npz]
    python -m brain.plastic --restore --checkpoint checkpoints/X_lobotomized.pt \
        [--out checkpoints/X_lobotomized_restored.pt] [--brain data/brain.npz]
"""
from __future__ import annotations

import argparse
import os

import torch

from brain.lif import LIF, load_brain


def _resolve_eid(lif, eid):
    if eid is None:
        return torch.arange(lif.nnz, device=lif.device)
    return torch.as_tensor(eid, dtype=torch.long, device=lif.device)


def snapshot_plastic(lif: object, eid: torch.Tensor | None = None) -> dict:
    """{"eid": edge ids (every edge if eid is None), "w": lif.w[eid] at call time, on cpu}."""
    eid = _resolve_eid(lif, eid)
    return {"eid": eid.clone(), "w": lif.w[eid].clone().cpu()}


def lobotomize(lif: object, eid: torch.Tensor | None = None, zero_lc10a: torch.Tensor | None = None) -> dict:
    """Reset the plastic synapses (eid, default every edge) to their connectome values lif.w0[eid].
    If zero_lc10a is a LongTensor of LC10a neuron indices (both sides), also lesion them via
    lif.lesion(zero_lc10a, mode="both") -- the dramatic "the fly is blind" demo version.
    Returns a snapshot taken BEFORE any of this, with the lesion snapshot nested under key "lesion"
    (or None if zero_lc10a wasn't given), so restore() can undo both exactly."""
    eid = _resolve_eid(lif, eid)
    snap = snapshot_plastic(lif, eid)
    lif.set_w_subset(eid, lif.w0[eid])
    snap["lesion"] = lif.lesion(zero_lc10a, mode="both") if zero_lc10a is not None else None
    return snap


def restore(lif: object, snapshot: dict) -> None:
    """Undo lobotomize() exactly: the lesion first (if any), then the weight reset."""
    if snapshot.get("lesion") is not None:
        lif.restore(snapshot["lesion"])
    lif.set_w_subset(snapshot["eid"], snapshot["w"].to(lif.device))


# ------------------------------------------------------------------ CLI
def _out_path(path, suffix):
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def _build_cpu_lif(brain_path):
    d, N, groups = load_brain(brain_path)
    lif = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, 1,
              device="cpu", engine="event")
    return lif, groups


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Lobotomize a checkpoint (reset learned synapses to the raw connectome, "
                     "optionally lesion LC10a) or restore one that was lobotomized.")
    ap.add_argument("--lobotomize", action="store_true", help="reset synapses to connectome values")
    ap.add_argument("--restore", action="store_true", help="undo a --lobotomize checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--zero-lc10a", action="store_true", help="also lesion LC10a_L/LC10a_R (--lobotomize only)")
    ap.add_argument("--brain", default="data/brain.npz")
    args = ap.parse_args(argv)

    if args.lobotomize == args.restore:
        ap.error("pass exactly one of --lobotomize or --restore")

    lif, groups = _build_cpu_lif(args.brain)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    lif.set_w(ck["w"].to("cpu"))

    if args.lobotomize:
        out = args.out or _out_path(args.checkpoint, "_lobotomized")
        if os.path.abspath(out) == os.path.abspath(args.checkpoint):
            ap.error("--out must not overwrite --checkpoint")
        zero_lc10a = torch.cat([groups["LC10a_L"], groups["LC10a_R"]]) if args.zero_lc10a else None

        snap = lobotomize(lif, eid=None, zero_lc10a=zero_lc10a)

        snap_path = out + ".snapshot.pt"
        torch.save(snap, snap_path)

        new_ck = dict(ck)
        new_ck["w"] = lif.w.cpu()
        new_ck["lobotomized"] = True
        new_ck["lobotomized_from"] = args.checkpoint
        new_ck["snapshot_path"] = snap_path
        torch.save(new_ck, out)

        n_syn = int(snap["eid"].numel())
        msg = f"lobotomized {args.checkpoint} -> {out}: reset {n_syn} synapses to connectome values"
        if zero_lc10a is not None:
            msg += f", lesioned {int(zero_lc10a.numel())} neurons (LC10a_L + LC10a_R)"
        msg += f" [snapshot {snap_path}]"
        print(msg)
    else:
        snap_path = ck.get("snapshot_path")
        if not snap_path or not os.path.exists(snap_path):
            ap.error(f"checkpoint has no usable snapshot_path: {snap_path!r}")
        snap = torch.load(snap_path, map_location="cpu", weights_only=True)

        restore(lif, snap)

        out = args.out or _out_path(args.checkpoint, "_restored")
        if os.path.abspath(out) == os.path.abspath(args.checkpoint):
            ap.error("--out must not overwrite --checkpoint")

        new_ck = dict(ck)
        new_ck["w"] = lif.w.cpu()
        for k in ("lobotomized", "lobotomized_from", "snapshot_path"):
            new_ck.pop(k, None)
        torch.save(new_ck, out)

        n_syn = int(snap["eid"].numel())
        n_lesioned = int(snap["lesion"]["n_neurons"]) if snap.get("lesion") is not None else 0
        msg = f"restored {args.checkpoint} -> {out}: reset {n_syn} synapses"
        if n_lesioned:
            msg += f", un-lesioned {n_lesioned} neurons"
        print(msg)


if __name__ == "__main__":
    main()
