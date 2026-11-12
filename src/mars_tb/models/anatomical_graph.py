"""Anatomical Graph Reasoning Network (AGRN), manuscript Section 2.4.4, Eqs. 9-10.

12 computational lung-zone nodes are mean-pooled from final-layer patch tokens, refined by three
graph-attention layers (8 heads) over three edge families (adjacency / hematogenous / lymphatic),
and read out as (i) 12 regional activations and (ii) a 768-d graph feature.

The regional activations are an UNSUPERVISED learned readout (the manuscript reports no node-level
supervision). They are not calibrated lesion probabilities and must not be reported as such.
Zone boxes and the edge list are implementation proposals; see resources/anatomical_graph.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from mars_tb.config import AGRNConfig

PROVENANCE = {"figure4_interpreted", "proposal"}


@dataclass(frozen=True)
class GraphSpec:
    node_names: tuple[str, ...]
    boxes: tuple[tuple[float, float, float, float], ...]  # (y0, y1, x0, x1)
    relations: tuple[str, ...]
    edges: tuple[tuple[int, int, int], ...]  # (src, dst, relation_index)
    edge_provenance: tuple[str, ...]
    self_loops: bool
    directed: bool
    laterality: str

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)


def parse_graph_spec(data: dict[str, Any]) -> GraphSpec:
    nodes = sorted(data["nodes"], key=lambda n: n["id"])
    ids = [n["id"] for n in nodes]
    if ids != list(range(len(nodes))):
        raise ValueError(f"node ids must be 0..{len(nodes) - 1}, got {ids}")
    relations = tuple(data["relations"])
    boxes = []
    for n in nodes:
        y0, y1, x0, x1 = (float(v) for v in n["box"])
        if not (0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1):
            raise ValueError(f"node {n['name']}: box {n['box']} outside [0,1] or empty")
        boxes.append((y0, y1, x0, x1))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if min(a[1], b[1]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[2], b[2]):
                raise ValueError(f"zone boxes {nodes[i]['name']} and {nodes[j]['name']} overlap")
    edges, prov = [], []
    for e in data["edges"]:
        s, d, rel = int(e["src"]), int(e["dst"]), e["relation"]
        if not (0 <= s < len(nodes) and 0 <= d < len(nodes)):
            raise ValueError(f"edge {e}: node index out of range")
        if s == d:
            raise ValueError(f"edge {e}: explicit self-edge; use self_loops instead")
        if rel not in relations:
            raise ValueError(f"edge {e}: unknown relation {rel!r}")
        if e.get("provenance") not in PROVENANCE:
            raise ValueError(f"edge {e}: provenance must be one of {sorted(PROVENANCE)}")
        edges.append((s, d, relations.index(rel)))
        prov.append(e["provenance"])
    if len({(s, d, r) for s, d, r in edges}) != len(edges):
        raise ValueError("duplicate edges in graph spec")
    return GraphSpec(
        node_names=tuple(n["name"] for n in nodes),
        boxes=tuple(boxes),
        relations=relations,
        edges=tuple(edges),
        edge_provenance=tuple(prov),
        self_loops=bool(data.get("self_loops", True)),
        directed=bool(data.get("directed", False)),
        laterality=str(data.get("laterality", "figure")),
    )


def load_graph_spec(source: str = "builtin") -> GraphSpec:
    if source == "builtin":
        raw = resources.files("mars_tb.resources").joinpath("anatomical_graph.yaml").read_text("utf-8")
    else:
        raw = Path(source).read_text("utf-8")
    spec = parse_graph_spec(yaml.safe_load(raw))
    if spec.laterality == "patient":  # mirror boxes left<->right
        mirrored = tuple((y0, y1, 1 - x1, 1 - x0) for (y0, y1, x0, x1) in spec.boxes)
        spec = GraphSpec(**{**spec.__dict__, "boxes": mirrored})
    return spec


def zone_pool_matrix(spec: GraphSpec, grid: int) -> torch.Tensor:
    """Row-normalised (n_nodes, grid*grid) token-to-zone averaging matrix."""
    centers = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
    cy, cx = torch.meshgrid(centers, centers, indexing="ij")
    cy, cx = cy.reshape(-1), cx.reshape(-1)
    rows = []
    for name, (y0, y1, x0, x1) in zip(spec.node_names, spec.boxes, strict=True):
        inside = ((cy >= y0) & (cy < y1) & (cx >= x0) & (cx < x1)).float()
        if inside.sum() == 0:
            raise ValueError(
                f"zone {name} contains no tokens on a {grid}x{grid} grid; use a larger image or coarser zone boxes"
            )
        rows.append(inside / inside.sum())
    return torch.stack(rows)


def relation_tensor(spec: GraphSpec) -> torch.Tensor:
    """(N, N, R+1) bool; [i, j] = relations by which node i may attend to node j.

    The last relation slot is the self-loop. Undirected graphs add both directions.
    """
    n, r = spec.n_nodes, len(spec.relations)
    rel = torch.zeros(n, n, r + 1, dtype=torch.bool)
    for s, d, k in spec.edges:
        rel[d, s, k] = True
        if not spec.directed:
            rel[s, d, k] = True
    if spec.self_loops:
        idx = torch.arange(n)
        rel[idx, idx, r] = True
    return rel


class RelationalGATLayer(nn.Module):
    """Dense multi-head graph attention with a learned per-relation, per-head logit bias.

    e_ij = LeakyReLU(a_dst . Wh_i + a_src . Wh_j) + sum_{r in rel(i,j)} b_{r,head}
    (Veličković et al. 2018, with relation biases added as an implementation proposal; the
    manuscript names the edge types but not how attention distinguishes them.)
    """

    def __init__(self, in_dim: int, head_dim: int, heads: int, n_rel_slots: int, dropout: float):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.lin = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, head_dim))
        self.rel_bias = nn.Parameter(torch.zeros(n_rel_slots, heads))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, h: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        b, n, _ = h.shape
        wh = self.lin(h).view(b, n, self.heads, self.head_dim)
        e_dst = (wh * self.a_dst).sum(-1)  # (B,N,H)
        e_src = (wh * self.a_src).sum(-1)
        logits = torch.nn.functional.leaky_relu(e_dst.unsqueeze(2) + e_src.unsqueeze(1), 0.2)  # (B,N,N,H)
        bias = torch.einsum("ijr,rh->ijh", rel.to(logits.dtype), self.rel_bias)
        logits = logits + bias.unsqueeze(0)
        mask = rel.any(-1)  # (N,N)
        logits = logits.masked_fill(~mask[None, :, :, None], float("-inf"))
        att = self.drop(torch.softmax(logits, dim=2))
        self.last_attention = att.detach().permute(0, 3, 1, 2)  # (B,H,N,N)
        out = torch.einsum("bijh,bjhd->bihd", att, wh)
        return out.reshape(b, n, self.heads * self.head_dim)


class AGRN(nn.Module):
    def __init__(self, dim: int, grid: int, cfg: AGRNConfig):
        super().__init__()
        if cfg.layers != 3:
            raise ValueError("AGRN is specified with exactly three graph-attention layers")
        spec = load_graph_spec(cfg.graph_spec)
        self.spec = spec
        self.n_nodes = spec.n_nodes
        self.register_buffer("pool", zone_pool_matrix(spec, grid), persistent=False)
        rel = relation_tensor(spec)
        self.register_buffer("rel", rel, persistent=False)
        n_slots = int(rel.shape[-1])
        head_dim = cfg.hidden // cfg.heads
        self.gat1 = RelationalGATLayer(dim, head_dim, cfg.heads, n_slots, cfg.dropout)
        self.gat2 = RelationalGATLayer(cfg.hidden, head_dim, cfg.heads, n_slots, cfg.dropout)
        self.gat3 = RelationalGATLayer(cfg.hidden, head_dim, cfg.heads, n_slots, cfg.dropout)
        self.res1 = nn.Linear(dim, cfg.hidden)  # residual mapping: dims differ (768 -> 512)
        self.norm1 = nn.LayerNorm(cfg.hidden)
        self.norm2 = nn.LayerNorm(cfg.hidden)
        self.out3 = nn.Linear(cfg.hidden, dim)  # layer 3: 512 -> 768 projection, no residual
        self.norm3 = nn.LayerNorm(dim)
        self.act = nn.ELU()
        self.node_head = nn.Linear(dim, 1)
        self.readout = nn.Linear(self.n_nodes * dim, dim)
        self.readout_act = nn.GELU()
        self.readout_norm = nn.LayerNorm(dim)

    def forward(self, z12: torch.Tensor) -> dict[str, torch.Tensor]:
        h0 = torch.einsum("zn,bnd->bzd", self.pool.to(z12.dtype), z12)  # Eq. 9
        h1 = self.norm1(self.res1(h0) + self.act(self.gat1(h0, self.rel)))  # Eq. 10, l=1
        h2 = self.norm2(h1 + self.act(self.gat2(h1, self.rel)))  # Eq. 10, l=2
        h3 = self.norm3(self.out3(self.gat3(h2, self.rel)))
        node_logit = self.node_head(h3).squeeze(-1)
        graph_feat = self.readout_norm(self.readout_act(self.readout(h3.flatten(1))))
        return {
            "z_agrn": graph_feat,
            "node_logit": node_logit,
            "node_prob": torch.sigmoid(node_logit),
            "node_features": h3,
            "attention": torch.stack(
                [a for a in (g.last_attention for g in (self.gat1, self.gat2, self.gat3)) if a is not None],
                dim=1,
            ),  # (B, 3 layers, H, N, N)
        }
