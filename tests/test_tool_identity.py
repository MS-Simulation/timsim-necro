"""The timsim-predict tools are fingerprinted by `--identity`, not by their launcher bytes.

`timsim-ccs` on PATH is a pip stub that never changes; the CCS it writes depends on pepdl, mscorepy
and the weights. Hashing the stub let pepdl's padding-mask fix (2+ CCS moved ~35 A^2) through with
every fingerprint unchanged. These tests run the flow against fake tools in a private TIMSIM_BIN.
"""
import importlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest
from necroflow import DAG, Pipeline

FLOW = Path(__file__).resolve().parents[1] / "flow"
sys.path.insert(0, str(FLOW))
CONF = FLOW / "configs"


def _script(path: Path, body: str):
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _fake_predict_tool(bin_dir: Path, name: str, payload: dict | None):
    """A launcher that answers --identity with `payload` (or fails, if None)."""
    if payload is None:
        body = "import sys; sys.exit(2)\n"
    else:
        body = f"import json, sys\nif '--identity' in sys.argv: print(json.dumps({payload!r}))\n"
    _script(bin_dir / name, body)


def _load_flow(monkeypatch, bin_dir: Path):
    """Import timsim_flow afresh with TIMSIM_BIN = bin_dir (templates are stamped at import)."""
    monkeypatch.setenv("TIMSIM_BIN", str(bin_dir))
    monkeypatch.setenv("PATH", os.environ["PATH"])
    sys.modules.pop("timsim_flow", None)
    return importlib.import_module("timsim_flow")


def _fingerprints(tf, tmp_path) -> dict[str, str]:
    cfg = dict(proteome_spec=str(CONF / "hela_proteome.toml"), mods=str(CONF / "mods_basic.toml"),
               design_spec=str(CONF / "design_hela.toml"), max_peptides=5000, sample="A_R1")
    dag = DAG(tmp_path / "out")
    P = Pipeline(dag)
    tf.job(P, cfg)
    out = {}
    for n in dag.nodes:
        assert n.output_name not in out, f"{n.output_name} twice; key the fingerprints differently"
        out[n.output_name] = str(n.relative_path)  # <rule>/<provenance hash>
    return out


def _bin(tmp_path, ccs=None, rt=None, frag=None):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    for name, payload in (("timsim-ccs", ccs), ("timsim-rt", rt), ("timsim-fragments", frag)):
        _fake_predict_tool(d, name, payload if payload is not False else None)
    return d


def ident(tool, pepdl="a"):
    return {"schema": 1, "tool": tool, "packages": {"pepdl": {"files_sha256": pepdl}}}


@pytest.fixture(autouse=True)
def _restore_flow():
    yield
    sys.modules.pop("timsim_flow", None)


def test_predict_tools_are_stamped_by_identity_not_launcher_bytes(monkeypatch, tmp_path):
    d = _bin(tmp_path, ident("timsim-ccs"), ident("timsim-rt"), ident("timsim-fragments"))
    tf = _load_flow(monkeypatch, d)
    one = tf._tool_sha("timsim-ccs")
    assert one.startswith("id-")
    # Same identity, different launcher bytes: the stamp must not move.
    (d / "timsim-ccs").write_text((d / "timsim-ccs").read_text() + "# a comment\n")
    tf2 = _load_flow(monkeypatch, d)
    assert tf2._tool_sha("timsim-ccs") == one


def test_a_changed_identity_moves_the_tool_and_every_downstream_fingerprint(monkeypatch, tmp_path):
    before = _fingerprints(_load_flow(monkeypatch, _bin(
        tmp_path, ident("timsim-ccs"), ident("timsim-rt"), ident("timsim-fragments"))), tmp_path)
    after = _fingerprints(_load_flow(monkeypatch, _bin(
        tmp_path, ident("timsim-ccs", pepdl="b"), ident("timsim-rt"), ident("timsim-fragments"))),
        tmp_path)
    assert before.keys() == after.keys()
    moved = {k for k in before if before[k] != after[k]}
    assert "precursor_ccs" in moved, moved
    assert moved - {"precursor_ccs"}, "nothing downstream of the CCS moved with it"
    # Everything that does not consume the CCS stays put: the structure upstream of it, and RT.
    for k in ("proteome", "peptides", "modforms", "precursors", "peptide_rt"):
        assert k in before and k not in moved, k


def test_an_rt_identity_change_moves_the_rt_node(monkeypatch, tmp_path):
    ids = {t: ident(t) for t in ("timsim-ccs", "timsim-rt", "timsim-fragments")}
    before = _fingerprints(_load_flow(monkeypatch, _bin(tmp_path, *ids.values())), tmp_path)
    ids["timsim-rt"] = ident("timsim-rt", pepdl="b")
    after = _fingerprints(_load_flow(monkeypatch, _bin(tmp_path, *ids.values())), tmp_path)
    assert before["peptide_rt"] != after["peptide_rt"]
    assert before["precursor_ccs"] == after["precursor_ccs"]


def test_a_fragments_identity_change_reaches_the_fragments_rule(monkeypatch, tmp_path):
    """The default job has no fragments node; check the stamped template the rule hashes."""
    ids = {t: ident(t) for t in ("timsim-ccs", "timsim-rt", "timsim-fragments")}
    one = _load_flow(monkeypatch, _bin(tmp_path, *ids.values())).fragments
    ids["timsim-fragments"] = ident("timsim-fragments", pepdl="b")
    two = _load_flow(monkeypatch, _bin(tmp_path, *ids.values())).fragments

    def template(rule):
        t = vars(rule).get("command") or getattr(rule, "template", None) or repr(vars(rule))
        return str(t)

    assert "timsim-fragments=id-" in template(one)
    assert template(one) != template(two)


def test_unchanged_identity_reuses_every_fingerprint(monkeypatch, tmp_path):
    ids = (ident("timsim-ccs"), ident("timsim-rt"), ident("timsim-fragments"))
    one = _fingerprints(_load_flow(monkeypatch, _bin(tmp_path, *ids)), tmp_path)
    two = _fingerprints(_load_flow(monkeypatch, _bin(tmp_path, *ids)), tmp_path)
    assert one == two


@pytest.mark.parametrize("bad", [False, {"schema": 1, "tool": "timsim-rt"}])
def test_a_tool_without_a_valid_identity_is_an_error_not_a_launcher_hash(monkeypatch, tmp_path, bad):
    """False: --identity exits non-zero (a timsim-predict that predates it). Dict: it answers, but
    as a different tool (a wrong or broken install)."""
    d = _bin(tmp_path, bad, ident("timsim-rt"), ident("timsim-fragments"))
    with pytest.raises(RuntimeError, match="no identity"):
        _load_flow(monkeypatch, d)


def test_native_binaries_keep_byte_hashing(monkeypatch, tmp_path):
    d = _bin(tmp_path, ident("timsim-ccs"), ident("timsim-rt"), ident("timsim-fragments"))
    (d / "timsim-render").write_bytes(b"\x7fELF fake binary v1")
    tf = _load_flow(monkeypatch, d)
    v1 = tf._tool_sha("timsim-render")
    assert not v1.startswith("id-")
    (d / "timsim-render").write_bytes(b"\x7fELF fake binary v2")
    assert _load_flow(monkeypatch, d)._tool_sha("timsim-render") != v1
