import copy

import pytest
import torch
import yaml

from mars_tb.config import AGRNConfig
from mars_tb.models.anatomical_graph import (
    AGRN,
    RelationalGATLayer,
    load_graph_spec,
    parse_graph_spec,
    relation_tensor,
    zone_pool_matrix,
)


@pytest.fixture()
def spec():
    return load_graph_spec()


def test_spec_has_twelve_nodes_and_three_relations(spec):
    assert spec.n_nodes == 12 and spec.relations == ("adjacency", "hematogenous", "lymphatic")
    assert spec.node_names[0] == "N0" and spec.node_names[11] == "N11"


def test_every_edge_has_provenance_and_proposals_are_marked(spec):
    assert set(spec.edge_provenance) == {"figure4_interpreted", "proposal"}
    assert spec.edge_provenance.count("proposal") == 2  # only the two lower-zone adjacencies


def test_figure4_hematogenous_and_lymphatic_edges_present(spec):
    rel = relation_tensor(spec)
    h, lym = spec.relations.index("hematogenous"), spec.relations.index("lymphatic")
    assert rel[0, 6, h] and rel[6, 0, h]
    assert rel[4, 10, lym] and rel[11, 5, lym]


@pytest.mark.parametrize("grid", [16, 32])
def test_zone_pooling_rows_are_means_over_disjoint_nonempty_regions(spec, grid):
    pool = zone_pool_matrix(spec, grid)
    assert pool.shape == (12, grid * grid)
    assert torch.allclose(pool.sum(1), torch.ones(12))
    membership = (pool > 0).sum(0)
    assert membership.max() == 1  # no token belongs to two zones
    assert (pool > 0).sum(1).min() >= 1


def test_zone_too_small_for_grid_is_an_error(spec):
    with pytest.raises(ValueError, match="no tokens"):
        zone_pool_matrix(spec, 8)


def _raw():
    from importlib import resources

    return yaml.safe_load(resources.files("mars_tb.resources").joinpath("anatomical_graph.yaml").read_text("utf-8"))


def test_overlapping_boxes_rejected():
    raw = _raw()
    raw["nodes"][1]["box"] = raw["nodes"][0]["box"]
    with pytest.raises(ValueError, match="overlap"):
        parse_graph_spec(raw)


@pytest.mark.parametrize(
    "edit,match",
    [
        (
            lambda r: r["edges"].append({"src": 0, "dst": 99, "relation": "adjacency", "provenance": "proposal"}),
            "range",
        ),
        (lambda r: r["edges"].append({"src": 0, "dst": 0, "relation": "adjacency", "provenance": "proposal"}), "self"),
        (lambda r: r["edges"].append({"src": 1, "dst": 7, "relation": "bogus", "provenance": "proposal"}), "relation"),
        (lambda r: r["edges"].append({"src": 1, "dst": 7, "relation": "adjacency"}), "provenance"),
        (lambda r: r["edges"].append(copy.deepcopy(r["edges"][0])), "duplicate"),
    ],
)
def test_invalid_edges_rejected(edit, match):
    raw = _raw()
    edit(raw)
    with pytest.raises(ValueError, match=match):
        parse_graph_spec(raw)


def test_laterality_patient_mirrors_boxes():
    a = load_graph_spec()
    raw = _raw()
    raw["laterality"] = "patient"
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh)
    b = load_graph_spec(fh.name)
    assert b.boxes[0][2] == pytest.approx(1 - a.boxes[0][3])


def test_gat_attention_is_a_distribution_over_neighbours_only(spec):
    rel = relation_tensor(spec)
    layer = RelationalGATLayer(8, 4, 2, rel.shape[-1], 0.0)
    layer(torch.randn(3, 12, 8), rel)
    att = layer.last_attention  # (B,H,N,N)
    assert torch.allclose(att.sum(-1), torch.ones(3, 2, 12), atol=1e-5)
    assert (att[:, :, ~rel.any(-1)] == 0).all()


def test_gat_is_permutation_equivariant(spec):
    rel = relation_tensor(spec)
    layer = RelationalGATLayer(8, 4, 2, rel.shape[-1], 0.0).eval()
    h = torch.randn(2, 12, 8)
    perm = torch.randperm(12)
    out = layer(h, rel)
    out_perm = layer(h[:, perm], rel[perm][:, perm])
    assert torch.allclose(out[:, perm], out_perm, atol=1e-5)


def test_agrn_shapes_and_probability_range():
    m = AGRN(32, 16, AGRNConfig(heads=4, hidden=16)).eval()
    out = m(torch.randn(2, 256, 32))
    assert out["z_agrn"].shape == (2, 32) and out["node_prob"].shape == (2, 12)
    assert ((out["node_prob"] > 0) & (out["node_prob"] < 1)).all()
    assert out["attention"].shape == (2, 3, 4, 12, 12)


def test_agrn_manuscript_dimensions():
    m = AGRN(768, 32, AGRNConfig())
    assert m.res1.out_features == 512 and m.out3.in_features == 512 and m.out3.out_features == 768
    assert m.readout.in_features == 9216 and m.readout.out_features == 768  # 12 x 768 -> 768
    assert m.gat1.heads == 8 and m.gat1.head_dim == 64


def test_agrn_requires_three_layers():
    with pytest.raises(ValueError):
        AGRN(32, 16, AGRNConfig(layers=2, heads=4, hidden=16))


def test_agrn_backward_is_finite():
    m = AGRN(32, 16, AGRNConfig(heads=4, hidden=16))
    x = torch.randn(2, 256, 32, requires_grad=True)
    out = m(x)
    (out["z_agrn"].sum() + out["node_prob"].sum()).backward()
    assert torch.isfinite(x.grad).all()
