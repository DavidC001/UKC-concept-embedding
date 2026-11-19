"""RelGeoE: a Euclidean KGE whose transformation and score depend on the relation's family.

Each UKC relation is assigned to the geometry that studies of LLM representations report for it
(evidence table in the "Relation geometry penalties plan" doc). The geometry is the scoring
function itself rather than a penalty on top of a generic model, and it is a property of the edge,
not of the query direction: an inverse query (?, r, t) scores the same function with the roles
swapped, so one set of parameters serves both directions. The score is still directed for
directed relations (f_r(h, t) != f_r(t, h) for IS_A, PART_OF, ...).

Built for the canonical datasets (data_build/canonicalize_relations.py, e.g. UKC_lex_noroot_canon):
every relation has one stored direction, RELATED_FORM and VALUE_OF are split by POS pair
("31:NV" = noun -> verb, "34:NA" = attribute noun -> value adjective), and symmetric.pickle lists
the symmetric relations. The family is looked up by the relation's base id (the part before ':').
For a symmetric relation in a directed family (offset, affine, hier, attr) the score is
symmetrised, max(f_r(h, t), f_r(t, h)): canonical storage puts the smaller entity id first, so the
stored order carries no meaning (an offset relation then means "differ by +/- beta_r").

All vectors are taken relative to a learned global centre mu (x' = x - mu). Every score is a
negative (weighted) squared distance.

  hier     IS_A (src = parent). Park et al. 2025: child - parent is orthogonal to the parent
           vector, so the child projects onto the parent direction at exactly |parent|.
           Score: -(a_r * par^2 + b_r * perp^2), with par = <c', p'^> - |p'| (distance of the child
           from the hyperplane through p' orthogonal to p') and perp^2 = |c' - p'|^2 - par^2
           (spread inside that hyperplane), a_r >= b_r > 0. A perfect fit gives
           |c'|^2 = |p'|^2 + perp^2, so specificity grows with norm and siblings differ by
           vectors orthogonal to the parent (the simplex picture).
  offset   RELATED_FORM, PARTICIPLE_OF_VERB, PERTAINS_TO, PERTAINYM_OF. Park, Choe & Veitch 2023 and
           the 2026 linearity study: inflection and derivation are one shared offset per relation
           (parallelograms). The canonical dataset gives each derivation type (POS pair) its own
           relation, so one offset beta_r per relation. Score: -|h + beta_r - t|^2.
  affine   PART_OF, MEMBER_OF, SUBSTANCE_OF, CAUSE_OF, LEXICAL ENTAILMENT, METONYM and any relation
           without a specific result. Hernandez et al. 2024 (linear relational embeddings) and
           Christ et al. 2025: relation decoding is an affine map whose relation-specific part is
           low rank. Map: W_r x' + b_r with W_r = s_r I + U_r V_r^T (rank k). Score: -|W_r h' + b_r - t'|^2.
           With relgeo_affine=tucker the linear part comes from a TuckER-style core shared by all
           relations: W_r = s_r I + G x_2 w_r, G in R^{d x d_r x d}, w_r in R^{d_r}. The maps of all
           relations live in one d_r-dim space of linear maps (any linear map if d_r = d^2), which is the
           shared operator basis Christ et al. 2025 report; the score stays a distance, not TuckER's dot.
  reflect  ANTONYM. No static antonym offset exists (2026 relation-geometry papers), but antonymy
           is one operation (Todd et al. 2024 function vectors) and antonyms share every feature
           except the attribute they oppose. Map: a reflection of a k-dim attribute subspace Q
           about a centre c_r, R(x) = x - 2 Q Q^T (x - c_r). It is an involution (antonymy is
           symmetric), it preserves the distance to c_r, and t - h = -2 Q Q^T (h - c_r) lies in the
           attribute subspace but points in a different direction for every pair.
  attr     VALUE_OF (attribute noun <-> value adjective). Engels et al. 2024: the values of an
           attribute lie on a circle or line inside a low-dimensional subspace. With z = v' - a' - b_r:
           score -(w_perp * |P_perp z|^2 + w_rad * (|Q^T z| - rho_r)^2), i.e. values sit on a sphere of
           radius rho_r around the attribute, inside the attribute subspace Q. The head is the
           attribute ("34:NA" is stored noun -> adjective).
           By default Q is shared with the antonym reflection, so antonyms are opposite values of
           the same attribute subspace.
  near     SIMILAR_TO, ALSO_SEE, VERB_GROUP, COGNATE. The 2026 relation-geometry paper finds no
           dedicated geometry for similarity, only closeness. Score: -s_r |h - t|^2; in ranking the
           query entity itself is masked out.

Options (read from args with defaults, see run.py): relgeo_families, relgeo_affine_rank,
relgeo_affine, relgeo_tucker_dim, relgeo_attr_rank, relgeo_attr_separate.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import euc_sqdistance

RELGEO_MODELS = ["RelGeoE"]

FAMILIES = ["hier", "offset", "affine", "reflect", "attr", "near"]

# UKC relation_type -> family. Relation names are the keys of relation_to_id.pickle.
DEFAULT_FAMILY = {
    "20": "hier",      # IS_A
    "47": "hier",      # TROPONYMY (verb hierarchy)
    "31": "offset",    # RELATED_FORM
    "32": "offset",    # PARTICIPLE_OF_VERB
    "33": "offset",    # PERTAINS_TO
    "46": "offset",    # PERTAINYM_OF
    "22": "affine",    # PART_OF
    "36": "affine",    # SUBSTANCE_OF
    "37": "affine",    # MEMBER_OF
    "38": "affine",    # CAUSE_OF
    "39": "affine",    # LEXICAL ENTAILMENT
    "101": "affine",   # METONYM
    "30": "reflect",   # ANTONYM
    "34": "attr",      # VALUE_OF
    "35": "near",      # SIMILAR_TO
    "40": "near",      # VERB_GROUP
    "41": "near",      # ALSO_SEE
    "100": "near",     # COGNATE
}
FALLBACK_FAMILY = "affine"

SYMMETRIC_FAMILIES = {"reflect", "near"}   # symmetric by construction
_MASK = -1e6


def _inv_softplus(y):
    return float(np.log(np.expm1(y)))


class RelGeoE(KGModel):
    """Relation-family Euclidean KGE (see module docstring)."""

    def __init__(self, args):
        super().__init__(args.sizes, args.rank, args.sym, args.dropout, args.gamma, args.bias, args.init_size)
        d, dt = self.rank, self.data_type
        self.n_rel = self.sizes[1] // 2
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], d), dtype=dt)
        # rel: offset beta_r (offset), translation b_r (affine, attr), reflection centre c_r (reflect)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], d), dtype=dt)
        self.centre = nn.Parameter(torch.zeros(d, dtype=dt))

        # relation -> family (DEFAULT_FAMILY, overridden by --relgeo_families), by relation name
        data_dir = Path(os.environ.get("DATA_PATH", "")) / getattr(args, "dataset", "")
        names = self._relation_names(data_dir)
        fam = self._families(names, getattr(args, "relgeo_families", None))
        self.register_buffer("fam_of", torch.tensor([FAMILIES.index(f) for f in fam], dtype=torch.long))
        self.family_names = {names[i]: fam[i] for i in range(self.n_rel)}
        # symmetric relations (args.sym, as in base.compute_metrics) whose family is directed get the
        # symmetrised score max(f_r(h, t), f_r(t, h))
        sym = sorted(int(r) for r in (args.sym or []))
        sym_mask = torch.zeros(self.n_rel, dtype=torch.bool)
        sym_mask[sym] = True
        self.register_buffer("symmetrise", sym_mask & torch.tensor([f not in SYMMETRIC_FAMILIES for f in fam]))
        self.sym_names = [names[i] for i in sym]

        R = self.n_rel
        # hier: a_r = b_r + softplus(alpha_r), b_r = softplus(beta_r); a ~ 1, b ~ 0.1 at init
        self.hier_alpha = nn.Parameter(torch.full((R,), _inv_softplus(0.9), dtype=dt))
        self.hier_beta = nn.Parameter(torch.full((R,), _inv_softplus(0.1), dtype=dt))
        # affine: W_r = s_r I + U_r V_r^T
        k = int(getattr(args, "relgeo_affine_rank", 8))
        self.affine_rank = k
        self.affine_s = nn.Parameter(torch.ones(R, dtype=dt))
        self.affine_type = getattr(args, "relgeo_affine", "lowrank") or "lowrank"
        if self.affine_type not in ("lowrank", "tucker"):
            raise ValueError(f"relgeo_affine must be lowrank or tucker, got {self.affine_type!r}")
        if self.affine_type == "lowrank":
            self.affine_U = nn.Parameter(torch.zeros((R, d, k), dtype=dt))
            self.affine_V = nn.Parameter(torch.randn((R, d, k), dtype=dt) / np.sqrt(d))
        else:
            dr = int(getattr(args, "relgeo_tucker_dim", 32))
            self.tucker_dim = dr
            # G x_2 w_r starts as a map of spectral norm ~0.1 (entries ~N(0, (0.05/sqrt(d))^2))
            self.tucker_core = nn.Parameter(torch.randn((d, dr, d), dtype=dt) * 0.05 / np.sqrt(d * dr))
            self.tucker_rel = nn.Parameter(torch.randn((R, dr), dtype=dt))
        # attribute subspace (reflect and attr); one shared basis unless relgeo_attr_separate
        ka = int(getattr(args, "relgeo_attr_rank", 8))
        self.attr_rank = ka
        self.attr_separate = bool(getattr(args, "relgeo_attr_separate", False))
        self.attr_basis = nn.Parameter(torch.randn((R if self.attr_separate else 1, d, ka), dtype=dt))
        self.attr_rho = nn.Parameter(torch.full((R,), _inv_softplus(1.0), dtype=dt))
        self.attr_wperp = nn.Parameter(torch.full((R,), _inv_softplus(1.0), dtype=dt))
        self.attr_wrad = nn.Parameter(torch.full((R,), _inv_softplus(1.0), dtype=dt))
        # near: s_r
        self.near_s = nn.Parameter(torch.full((R,), _inv_softplus(1.0), dtype=dt))

    # ------------------------------------------------------------------ setup helpers
    def _relation_names(self, data_dir):
        f = data_dir / "relation_to_id.pickle"
        if f.is_file():
            rel2id = pickle.load(open(f, "rb"))
            names = {v: str(k) for k, v in rel2id.items()}
            return [names.get(i, str(i)) for i in range(self.n_rel)]
        return [str(i) for i in range(self.n_rel)]

    def _families(self, names, spec):
        table = dict(DEFAULT_FAMILY)
        if spec:  # "20:hier,31:offset,..." overrides the default table
            for item in spec.split(","):
                k, v = item.rsplit(":", 1)
                if v not in FAMILIES:
                    raise ValueError(f"unknown relation family {v!r}; choose from {FAMILIES}")
                table[k.strip()] = v.strip()
        return [table.get(n, table.get(n.split(":")[0], FALLBACK_FAMILY)) for n in names]

    def describe(self):
        by = {}
        for n, f in self.family_names.items():
            by.setdefault(f, []).append(n)
        return "; ".join(f"{f}: {','.join(by[f])}" for f in FAMILIES if f in by) + \
            f" | symmetric: {','.join(self.sym_names) or 'none'}, " \
            f"affine {'tucker d_r=%d' % self.tucker_dim if self.affine_type == 'tucker' else 'rank %d' % self.affine_rank}, " \
            f"attribute rank {self.attr_rank} ({'per relation' if self.attr_separate else 'shared'})"

    # ------------------------------------------------------------------ KGModel interface
    def get_queries(self, queries):
        return (self.entity(queries[:, 0]), queries[:, 0], queries[:, 1]), self.bh(queries[:, 0])

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return (self.entity.weight, None), self.bt.weight
        return (self.entity(queries[:, 2]), queries[:, 2]), self.bt(queries[:, 2])

    def get_factors(self, queries):
        r0 = queries[:, 1] % self.n_rel
        return self.entity(queries[:, 0]), self.rel(r0), self.entity(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        x, xid, r = lhs_e
        Y, yid = rhs_e
        if yid is None:
            yid = torch.arange(Y.shape[0], device=Y.device)
        r0, inv = r % self.n_rel, r >= self.n_rel
        fam = self.fam_of[r0]
        out = x.new_zeros((x.shape[0], Y.shape[0] if eval_mode else 1))
        for f_id, f_name in enumerate(FAMILIES):
            rows = (fam == f_id).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            Yr, yr = (Y, yid) if eval_mode else (Y[rows], yid[rows])
            fn = getattr(self, f"_score_{f_name}")
            sym = self.symmetrise[r0[rows]]
            if sym.any():   # max over both readings of the edge
                s = torch.maximum(fn(x[rows], xid[rows], r0[rows], torch.zeros_like(sym), Yr, yr, eval_mode),
                                  fn(x[rows], xid[rows], r0[rows], torch.ones_like(sym), Yr, yr, eval_mode))
                if not sym.all():
                    s = torch.where(sym[:, None], s, fn(x[rows], xid[rows], r0[rows], inv[rows], Yr, yr, eval_mode))
            else:
                s = fn(x[rows], xid[rows], r0[rows], inv[rows], Yr, yr, eval_mode)
            out = out.index_copy(0, rows, s.to(out.dtype))
        return out

    # ------------------------------------------------------------------ helpers
    def _c(self, z):
        return z - self.centre

    def _attr_Q(self, r):
        """Orthonormal basis [d, k] of the attribute subspace of relation r (shared by default)."""
        Q, _ = torch.linalg.qr(self.attr_basis[r if self.attr_separate else 0])
        return Q

    # ------------------------------------------------------------------ family scores
    # Each takes the query rows (x, xid, r0, inv) and the candidates (Y, yid; all entities in
    # eval mode, else the aligned tails) and returns [b, N] or [b, 1].

    def _score_hier(self, x, xid, r0, inv, Y, yid, eval_mode):
        b = F.softplus(self.hier_beta[r0])[:, None]
        a = b + F.softplus(self.hier_alpha[r0])[:, None]
        Xc, Yc = self._c(x), self._c(Y)
        out = x.new_zeros((x.shape[0], Y.shape[0] if eval_mode else 1))
        for m in (~inv, inv):
            if not m.any():
                continue
            X = Xc[m]
            C = Yc if eval_mode else Yc[m]
            if m is not inv:   # query is the parent, candidates are children
                pn = X.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                cu = (X / pn) @ C.t() if eval_mode else (C * X).sum(-1, keepdim=True) / pn
                par = cu - pn
            else:              # query is the child, candidates are parents
                pn = C.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                if eval_mode:
                    pn = pn.t()
                cp = X @ C.t() if eval_mode else (X * C).sum(-1, keepdim=True)
                par = cp / pn - pn
            d2 = euc_sqdistance(X, C, eval_mode)
            perp2 = (d2 - par ** 2).clamp_min(0)
            out[m] = -(a[m] * par ** 2 + b[m] * perp2)
        return out

    def _score_offset(self, x, xid, r0, inv, Y, yid, eval_mode):
        sign = (1.0 - 2.0 * inv.to(x.dtype))[:, None]              # inverse query: h = t - beta
        return -euc_sqdistance(x + sign * self.rel.weight[r0], Y, eval_mode)

    def _tucker_W(self, r):
        """Linear parts G x_2 w_r + s_r I for relation ids r [b] -> [b, d, d]."""
        W = torch.einsum("bk,ikj->bij", self.tucker_rel[r], self.tucker_core)
        return W + self.affine_s[r][:, None, None] * torch.eye(self.rank, dtype=W.dtype, device=W.device)

    def _affine_map(self, Z, r0):
        """W_r z + b_r for z' rows [b, d] with per-row relations."""
        if self.affine_type == "tucker":
            uniq, inv_idx = torch.unique(r0, return_inverse=True)
            W = self._tucker_W(uniq)[inv_idx]
            return torch.bmm(W, Z[:, :, None])[:, :, 0] + self.rel.weight[r0]
        U, V = self.affine_U[r0], self.affine_V[r0]
        low = torch.bmm(U, torch.bmm(V.transpose(1, 2), Z[:, :, None]))[:, :, 0]
        return self.affine_s[r0][:, None] * Z + low + self.rel.weight[r0]

    def _score_affine(self, x, xid, r0, inv, Y, yid, eval_mode):
        Xc, Yc = self._c(x), self._c(Y)
        out = x.new_zeros((x.shape[0], Y.shape[0] if eval_mode else 1))
        fwd = ~inv
        if fwd.any():
            q = self._affine_map(Xc[fwd], r0[fwd])
            out[fwd] = -euc_sqdistance(q, Yc if eval_mode else Yc[fwd], eval_mode)
        if inv.any():
            if not eval_mode:   # candidates are heads: map them, compare to the query tail
                out[inv] = -euc_sqdistance(self._affine_map(Yc[inv], r0[inv]), Xc[inv], False)
            else:
                idx = inv.nonzero(as_tuple=True)[0]
                for r in torch.unique(r0[idx]).tolist():
                    rows = idx[r0[idx] == r]
                    M = self._affine_map_all(Yc, r)
                    out[rows] = -euc_sqdistance(Xc[rows], M, True)
        return out

    def _affine_map_all(self, Z, r):
        if self.affine_type == "tucker":
            W = self._tucker_W(torch.tensor([r], device=Z.device))[0]
            return Z @ W.t() + self.rel.weight[r]
        U, V = self.affine_U[r], self.affine_V[r]
        return self.affine_s[r] * Z + (Z @ V) @ U.t() + self.rel.weight[r]

    def _score_reflect(self, x, xid, r0, inv, Y, yid, eval_mode):
        # involution: the same map serves both directions
        q = torch.empty_like(x)
        for r in torch.unique(r0).tolist():
            rows = r0 == r
            Q, c = self._attr_Q(r), self.rel.weight[r]
            q[rows] = x[rows] - 2 * ((x[rows] - c) @ Q) @ Q.t()
        return -euc_sqdistance(q, Y, eval_mode)

    def _score_attr(self, x, xid, r0, inv, Y, yid, eval_mode):
        q_is_attr = ~inv                                          # head = attribute
        Xc, Yc = self._c(x), self._c(Y)
        b = self.rel.weight[r0]
        # z = v' - a' - b. Query attribute: m = a' + b, z = Y - m. Query value: m = v' - b, z = m - Y.
        m = torch.where(q_is_attr[:, None], Xc + b, Xc - b)
        out = x.new_zeros((x.shape[0], Y.shape[0] if eval_mode else 1))
        for r in torch.unique(r0).tolist():
            rows = r0 == r
            Q = self._attr_Q(r)
            mr = m[rows]
            C = Yc if eval_mode else Yc[rows]
            zu2 = euc_sqdistance(mr @ Q, C @ Q, eval_mode).clamp_min(0)
            perp2 = (euc_sqdistance(mr, C, eval_mode) - zu2).clamp_min(0)
            rho = F.softplus(self.attr_rho[r])
            wp, wr = F.softplus(self.attr_wperp[r]), F.softplus(self.attr_wrad[r])
            out[rows] = -(wp * perp2 + wr * (torch.sqrt(zu2 + 1e-12) - rho) ** 2)
        return out

    def _score_near(self, x, xid, r0, inv, Y, yid, eval_mode):
        s = F.softplus(self.near_s[r0])[:, None]
        out = -s * euc_sqdistance(x, Y, eval_mode)
        if eval_mode:   # a concept is not its own synonym
            out = out.clone()
            out[torch.arange(x.shape[0], device=x.device), xid] = _MASK
        return out
