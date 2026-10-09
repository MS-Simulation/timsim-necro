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


@needs_ref
def test_toml_integer_for_float_option_is_accepted(tmp_path):
    # TOML `noise_mz_ppm = 6` is an int; the CLI would have produced 6.0. On the Bruker path, where the
    # value reaches the render command, both must resolve to `--noise-mz-ppm 6.0`.
    as_int = commands(base(bruker_reference=REF, noise_mz_ppm=6), tmp_path)
    assert as_int == commands(base(bruker_reference=REF, noise_mz_ppm=6.0), tmp_path)
    assert any("--noise-mz-ppm 6.0" in c for c in as_int if "timsim-render" in c)


@pytest.mark.parametrize("key,value", [("mods", None), ("samples", [None]), ("samples", [["A_R1"]])])
def test_null_and_nested_values_are_refused(tmp_path, key, value):
    with pytest.raises(SystemExit, match="null is not allowed|not a list"):
        commands(base(**{key: value}), tmp_path)


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


def test_quant_job_refuses_reversed_conditions(tmp_path):
    # Needs no reference: the condition check runs before anything is built.
    cfg = dict(quant=True, proteome_spec=str(CONF / "hye.toml"), design_spec=str(CONF / "design.toml"),
               samples=["B_R1", "A_R1"], max_peptides=5000, mods=str(CONF / "mods_basic.toml"))
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


# ── job ≡ CLI: the same experiment written as a job and as a command line resolves to the same commands ──
# Compared with fingerprint directories masked. necroflow folds each output's type name, module included,
# into the fingerprint (rule_call.py), and the CLI defines the types in `__main__` while job() (and
# necroflow's own job runner, which names the module after the file) defines them in `timsim_flow`. So the
# two never share cache entries even for identical commands — a known, separate issue. What is compared
# here is what actually runs: every tool, flag and value.
import re  # noqa: E402

_FP = re.compile(r"/([a-z0-9_]+)/[0-9a-f]{64}/")


def masked(cmds):
    return sorted(_FP.sub(r"/\1/<fp>/", c) for c in cmds)

def cli_commands(argv, tmp_path):
    import subprocess
    out = subprocess.run([sys.executable, str(FLOW / "timsim_flow.py"), "--outdir", str(tmp_path), *argv,
                          "--dry-run"], capture_output=True, text=True, check=True, cwd=FLOW).stdout
    return sorted(l.strip() for l in out.split("resolved commands:", 1)[1].splitlines() if l.strip())


def as_argv(cfg):
    argv = []
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif isinstance(v, list):
            argv += [flag, *map(str, v)]
        elif v is not False:
            argv += [flag, str(v)]
    return argv


@needs_ref
@pytest.mark.parametrize("extra", [
    dict(noise_mz_ppm=6.5, noise_frag_ppm=6.5, noise_real_data=True, gradient_s=3600),
    dict(phospho=True, mods=str(CONF / "mods_phospho_reg.toml")),
])
def test_job_and_cli_resolve_the_same_bruker_commands(tmp_path, extra):
    cfg = base(bruker_reference=REF, search_fasta=str(CONF / "hela_subset.fasta"), **extra)
    sample = cfg.pop("sample")
    job_cmds = masked(commands(dict(cfg, sample=sample), tmp_path))
    assert job_cmds == masked(cli_commands(as_argv(dict(cfg, samples=[sample])), tmp_path))


@needs_ref
def test_job_and_cli_resolve_the_same_quant_commands(tmp_path):
    cfg = dict(quant=True, proteome_spec=str(CONF / "hye.toml"), design_spec=str(CONF / "design.toml"),
               samples=["A_R1", "B_R1"], max_peptides=5000, mods=str(CONF / "mods_basic.toml"),
               bruker_reference=REF, search_fasta=str(CONF / "hela_subset.fasta"))
    assert masked(commands(cfg, tmp_path)) == masked(cli_commands(as_argv(cfg), tmp_path))


@needs_ref
def test_legacy_peak_shape_reaches_both_renders(tmp_path):
    cfg = base(bruker_reference=REF, noise_real_data=True, gradient_s=3601.5, peak_shape="per-peptide-legacy",
               legacy_rt_sigma_mean=0.9, legacy_rt_sigma_var=0.2, legacy_rt_lambda_mean=1.5, legacy_rt_lambda_var=0.01,
               search_fasta=str(CONF / "hela_subset.fasta"))  # a search is what adds the noise-only control
    renders = [c for c in commands(cfg, tmp_path) if "/timsim-render --" in c]
    assert len(renders) == 2 and sum("--noise-only" in c for c in renders) == 1, renders
    for c in renders:
        assert "--peak-shape per-peptide-legacy" in c and "--n-frames 34155" in c
        assert "--legacy-rt-sigma-mean 0.9 --legacy-rt-sigma-var 0.2" in c
        assert "--legacy-rt-lambda-mean 1.5 --legacy-rt-lambda-var 0.01" in c


def test_legacy_peak_shape_refused_for_thermo(tmp_path):
    for shape in ("per-peptide-legacy", "gaussian"):  # any non-default shape would be ignored there
        with pytest.raises(SystemExit, match="Bruker render only"):
            commands(base(thermo_template="/nonexistent/template.raw", peak_shape=shape), tmp_path)


def test_unknown_peak_shape_refused(tmp_path):
    with pytest.raises(SystemExit, match="expected one of"):
        commands(base(peak_shape="legacy"), tmp_path)


@needs_ref
def test_job_and_cli_agree_on_custom_legacy_values(tmp_path):
    cfg = base(bruker_reference=REF, search_fasta=str(CONF / "hela_subset.fasta"), noise_real_data=True,
               gradient_s=3601.5, peak_shape="per-peptide-legacy", legacy_rt_sigma_mean=0.9,
               legacy_rt_sigma_var=0.2, legacy_rt_lambda_mean=1.5, legacy_rt_lambda_var=0.01)
    sample = cfg.pop("sample")
    assert masked(commands(dict(cfg, sample=sample), tmp_path)) == \
        masked(cli_commands(as_argv(dict(cfg, samples=[sample])), tmp_path))
