# Notice

The code repository is forked from https://github.com/tensorflow/neural-structured-learning/tree/master/research/kg_hyp_emb,
which is the repository if the Publication "Low-Dimensional Hyperbolic Knowledge Graph Embeddings"

# RelGeoE: relation-family geometry

`models/relgeo.py` adds `--model RelGeoE`, a Euclidean KGE whose transformation and score depend on the
relation, following the geometry that LLM-representation studies report for each relation type
(details and references in the module docstring):

| Family | UKC relations | Transformation / score |
|---|---|---|
| `hier` | IS_A | child on the hyperplane through the parent orthogonal to it: `-(a·par² + b·perp²)` |
| `offset` | RELATED_FORM, PARTICIPLE_OF_VERB, PERTAINS_TO, PERTAINYM_OF | one shared offset per relation (`31:NV`, `31:NA`, ... are separate relations in the canonical dataset) |
| `affine` | PART_OF, MEMBER_OF, SUBSTANCE_OF, CAUSE_OF, LEXICAL ENTAILMENT, METONYM (and unknown); HAS_ASPECT is folded into IS_A by the canonical datasets | `t ≈ (s I + U Vᵀ) h + b`, low-rank relation-specific part |
| `reflect` | ANTONYM | reflection of a k-dim attribute subspace about a centre (involution) |
| `attr` | VALUE_OF | values on a sphere around the attribute inside the attribute subspace (shared with ANTONYM) |
| `near` | SIMILAR_TO, VERB_GROUP, ALSO_SEE | plain closeness, `-s·‖h − t‖²` |

Use it on a canonical dataset (`data_build/canonicalize_relations.py`, e.g. `UKC_lex_noroot_canon`). Families are
matched on the relation's base id (`31:NV` → `31`). The score is a property of the edge: inverse queries use
the same parameters with the roles swapped, and directed relations stay directed. Relations in
`symmetric.pickle` whose family is directed (`31:NN`, `31:AA`, `34:NN`, ...) are scored as
`max(f(h, t), f(t, h))`, since canonical storage order is arbitrary.
`--relgeo_affine tucker` replaces the per-relation low-rank part with a TuckER-style core shared by all affine
relations, `W_r = s_r I + G ×₂ w_r` (`G`: d × d_r × d, `--relgeo_tucker_dim` d_r, default 32), still scored as a distance.
Flags: `--relgeo_families 22:offset,...` (override the table), `--relgeo_affine_rank`, `--relgeo_attr_rank`,
`--relgeo_attr_separate`.
