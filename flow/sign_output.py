"""Sign one rendered run with mzprov, from inside its own necroflow node.

The render's output is never touched. This node links to it and writes the
signature beside the link, in its own directory:

    <sign node>/data.d               -> symlink to the render's data.d (or .raw / .mzML)
    <sign node>/data.provenance.json    the mzprov record: TimSim, renderer build, key
    <sign node>/data.config.toml        mzprov's signed copy of render_config.json
    <sign node>/render_config.json      the run configuration and the answer key's SHA-256

so `mzprov verify <sign node>/data.d` checks the render output in place. Keeping
the signature out of the render's bytes matters here: an embedded signature
carries a timestamp, which would change the render's content hash on every run
and invalidate everything downstream of it.

The answer key (truth parquet) is bound through the hash in the signed config;
`mzprov verify` does not re-hash it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SUFFIX = {"d": ".d", "raw": ".raw", "mzml": ".mzML"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, required=True, help="the render's output")
    ap.add_argument("--kind", choices=sorted(SUFFIX), required=True)
    ap.add_argument("--truth", type=Path, required=True, help="the render's answer key")
    ap.add_argument("--sidecar", type=Path, required=True, help="<node>/<stem>.provenance.json")
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--tool-version", required=True, help="renderer binary and its content hash")
    ap.add_argument("--config-json", required=True, help="the run configuration, as JSON")
    ap.add_argument("--key", default="", help="signing key or its directory; empty = mzprov's default")
    a = ap.parse_args(argv)

    from mzprov import sign_mzml_output, sign_raw_output, sign_simulation_output

    node = a.sidecar.parent
    node.mkdir(parents=True, exist_ok=True)
    stem = a.sidecar.name[: -len(".provenance.json")]
    link = node / (stem + SUFFIX[a.kind])
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(a.artifact.resolve(), link)

    config = node / "render_config.json"
    config.write_text(json.dumps({
        "renderer": a.tool_version,
        "run": json.loads(a.config_json),
        "artifact": str(a.artifact.resolve()),
        "truth_sha256": _sha256(a.truth),
    }, sort_keys=True, indent=2) + "\n")

    key = a.key or None
    if a.kind == "d":
        out = sign_simulation_output(
            d_path=link, ground_truth_path=None, config_path=config,
            experiment_name=a.experiment_name, simulator_name="TimSim",
            simulator_version=a.tool_version, sidecar_path=a.sidecar,
            private_key_path=key,
        )
    elif a.kind == "mzml":
        out = sign_mzml_output(
            mzml_path=link, config_path=config, experiment_name=a.experiment_name,
            tool_name="TimSim", tool_version=a.tool_version, sidecar_path=a.sidecar,
            private_key_path=key,
        )
    else:
        out = sign_raw_output(
            raw_path=link, config_path=config, experiment_name=a.experiment_name,
            tool_name="TimSim", tool_version=a.tool_version, sidecar_path=a.sidecar,
            private_key_path=key,
        )
    print(f"signed: {out}")
    return Path(out)


if __name__ == "__main__":
    main()
