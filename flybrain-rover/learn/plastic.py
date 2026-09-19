"""
plastic.py -- which connectome synapses are allowed to change.

    eid = plastic_edges(lif, groups, "kc_mbon")     # LongTensor of edge ids into lif.w

  kc_mbon : pre in KC and post in MBON            (the fly's own learning site; ~61k edges)
  dn_in   : post in a descending-neuron group      (synapses onto DNa02/DNa01/DNp09/GF; ~4.5k edges)
  lc_dn   : pre in an eye group OR post in a DN group   (~195k edges)
  both    : kc_mbon + lc_dn
The sparsity mask never changes: only the weights of these existing edges do, and each keeps its sign.
"""
from __future__ import annotations

import torch

EYE_PREFIXES = ("LC", "LPLC2", "THERMO")
DN_KEYS = ["DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "GF"]


def _cat(groups, keys):
    return torch.cat([groups[k] for k in keys]) if keys else torch.zeros(0, dtype=torch.long)


def plastic_edges(lif: object, groups: dict[str, torch.Tensor], which: str = "kc_mbon") -> torch.Tensor:
    # masks are built on CPU: torch.isin on MPS materialises an (nnz x group) matrix (9.5 GB for KC)
    pre, post = lif.pre_idx.cpu(), lif.post_idx.cpu()
    cpu = lambda k: groups[k].cpu()
    eye = _cat({k: cpu(k) for k in groups}, [k for k in groups if k.startswith(EYE_PREFIXES)])
    dn = _cat({k: cpu(k) for k in groups}, [k for k in DN_KEYS if k in groups])
    kc_mbon = torch.isin(pre, cpu("KC")) & torch.isin(post, cpu("MBON"))
    dn_in = torch.isin(post, dn)
    lc_dn = torch.isin(pre, eye) | dn_in
    mask = {"kc_mbon": kc_mbon, "dn_in": dn_in, "lc_dn": lc_dn, "both": kc_mbon | lc_dn}[which]
    return mask.nonzero().flatten().to(lif.device)


def check_dale(lif: object, eid: torch.Tensor | None = None) -> bool:
    """True if every plastic edge still has the sign of its initial weight (or is zero)."""
    eid = torch.arange(lif.nnz, device=lif.device) if eid is None else eid
    w, w0 = lif.w[eid], lif.w0[eid]
    return bool(((w * w0 >= 0)).all())
