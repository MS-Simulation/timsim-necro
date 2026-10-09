"""The job-TOML entry point (`timsim_flow.job`) must build the same experiments the CLI builds, and
refuse what the CLI refuses. Pipelines are only constructed and resolved here, never run."""
import sys
from pathlib import Path

import pytest
from necroflow import DAG, Pipeline

FLOW = Path(__file__).resolve().parents[1] / "flow"
sys.path.insert(0, str(FLOW))
import timsim_flow as tf  # noqa: E402

CONF = FLOW / "configs"
REF = "/media/hd02/data/raw/dia/blanks/blanks-dia-PASEF/G241217_011_Slot2-2_1_16312.d"
needs_ref = pytest.mark.skipif(not Path(REF).exists(), reason="reference blank not on this machine")


def base(**over):
    cfg = dict(proteome_spec=str(CONF / "hela_proteome.toml"), mods=str(CONF / "mods_basic.toml"),
               design_spec=str(CONF / "design_hela.toml"), max_peptides=5000, sample="A_R1")
    cfg.update(over)
    return cfg


def commands(config, tmp_path):
    dag = DAG(tmp_path)
    P = Pipeline(dag)
    tf.job(P, config)
    dag.require([n for n in vars(P).values() if hasattr(n, "rule_call")] or list(dag.nodes))
    return [c for c in (call.resolve() for call in tf._calls(dag)) if c]


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(SystemExit, match="unknown config key"):
        commands(base(noise_mz_pmm=6.5), tmp_path)


@pytest.mark.parametrize("key,value,why", [
    ("max_peptides", "5000", "not a string"),          # a quoted number is a typo, not an int
    ("max_peptides", 2.5, "expected an integer"),
    ("noise_mz_ppm", True, "not a boolean"),           # bool is an int subclass; must still be refused
    ("noise_real_data", "yes", "expected true/false"),
    ("samples", "A_R1", "expected a non-empty list"),
])
def test_values_are_type_checked(tmp_path, key, value, why):
    with pytest.raises(SystemExit, match=why):
        commands(base(**{key: value}), tmp_path)


def test_toml_integer_for_float_option_is_accepted(tmp_path):
    # TOML `noise_mz_ppm = 6` is an int; the CLI would have produced 6.0. Same command either way.
    assert commands(base(noise_mz_ppm=6), tmp_path) == commands(base(noise_mz_ppm=6.0), tmp_path)


def test_several_samples_without_sample_is_refused(tmp_path):
    cfg = base(samples=["A_R1", "B_R1"])
    del cfg["sample"]
    with pytest.raises(SystemExit, match="a job runs one sample"):
        commands(cfg, tmp_path)


def test_seed_is_configurable_and_defaults_to_41(tmp_path):
    default = commands(base(), tmp_path)
    assert default == commands(base(seed=41), tmp_path)
    assert default != commands(base(seed=7), tmp_path)


@needs_ref
def test_quant_job_builds_the_joint_pipeline(tmp_path):
    cfg = dict(quant=True, proteome_spec=str(CONF / "hye.toml"), design_spec=str(CONF / "design.toml"),
               samples=["A_R1", "B_R1"], max_peptides=5000, mods=str(CONF / "mods_basic.toml"),
               bruker_reference=REF, search_fasta=str(CONF / "hela_subset.fasta"))
    cmds = commands(cfg, tmp_path)
    renders = [c for c in cmds if "timsim-render" in c]
    assert {("--sample A_R1" in c, "--sample B_R1" in c) for c in renders} == {(True, False), (False, True)}
    assert any("search_bruker_joint" in c for c in cmds), "no joint search"
    assert any("v2_quant_eval" in c for c in cmds), "no fold-change evaluation"


@needs_ref
def test_quant_job_refuses_reversed_conditions(tmp_path):
    cfg = dict(quant=True, proteome_spec=str(CONF / "hye.toml"), design_spec=str(CONF / "design.toml"),
               samples=["B_R1", "A_R1"], max_peptides=5000, mods=str(CONF / "mods_basic.toml"),
               bruker_reference=REF, search_fasta=str(CONF / "hela_subset.fasta"))
    with pytest.raises(SystemExit, match="FIRST sample must be the design reference"):
        commands(cfg, tmp_path)


@needs_ref
def test_gradient_reaches_the_render(tmp_path):
    cmds = commands(base(bruker_reference=REF, gradient_s=3600), tmp_path)
    assert any("--n-frames 34141" in c for c in cmds if "timsim-render" in c)


@needs_ref
def test_shipped_example_job_still_loads(tmp_path):
    import tomllib
    cfg = {k: v for k, v in tomllib.load(open(CONF / "job_example.toml", "rb")).items() if not k.startswith(".")}
    for k in ("proteome_spec", "mods", "design_spec", "search_fasta"):  # relative to configs/ in the example
        cfg[k] = str(CONF / cfg[k])
    assert any("timsim-render" in c for c in commands(cfg, tmp_path))
