"""Site paths for hep-any2any.

Every data, checkpoint and output location used by the package and the configs
resolves through this module. Resolution order for each variable:

  1. the process environment,
  2. the ``.env`` file at the repository root (or ``$HEP4M_ENV_FILE``),
  3. the defaults below, built from ``HEP4M_DATA`` and ``HEP4M_WORK``.

Normally only two variables need setting:

  HEP4M_DATA  downloaded data: the Zenodo files (``release/``), the tokenised store
              (``Tokenized_7mod/``) and the frozen tokeniser checkpoints (``Checkpoints/``).
  HEP4M_WORK  a writable directory for training runs, inference outputs and the
              released model checkpoints (``checkpoints/ckpt.<row>/``).

YAML configs refer to these as ``${VAR}``; they are expanded on load (see
``safe_load_expanded``). Shell scripts get them with
``eval "$(python -m hep4m.paths --export)"``. Python code uses the module
constants, e.g. ``hep4m.paths.DATA``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("HEP4M_REPO", Path(__file__).resolve().parent.parent))


def _load_dotenv(path: Path) -> None:
    """Set variables from a ``KEY=VALUE`` file without overriding the environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export "):
            s = s[len("export "):]
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(Path(os.environ.get("HEP4M_ENV_FILE", REPO / ".env")))

# (variable, default). Defaults may reference earlier variables with ${...}.
_DEFAULTS: list[tuple[str, str]] = [
    ("HEP4M_REPO", str(REPO)),
    ("HEP4M_DATA", "${HEP4M_REPO}/data"),
    ("HEP4M_WORK", "${HEP4M_REPO}/work"),
    ("HEP4M_TOKENIZED", "${HEP4M_DATA}/Tokenized_7mod"),
    ("HEP4M_TOKENIZERS", "${HEP4M_DATA}/Checkpoints"),
    ("HEP4M_RELEASE", "${HEP4M_DATA}/release"),
    ("HEP4M_RAW_TEST", "${HEP4M_RELEASE}/eval.raw_test_singlejet_0_100kseg_bw100.0.root"),
    # raw COCOA training sample, needed only to retrain the tokenisers (not in the record)
    ("HEP4M_RAW_TRAIN", "${HEP4M_DATA}/raw_train"),
    ("HEP4M_CKPT", "${HEP4M_WORK}/checkpoints"),
    ("HEP4M_RUNS", "${HEP4M_WORK}/runs"),
    ("HEP4M_OUTPUT_ROOT", "${HEP4M_WORK}/outputs"),
    ("HEP4M_LOGGER", "csv"),
]
for _k, _v in _DEFAULTS:
    os.environ.setdefault(_k, os.path.expandvars(_v))
    os.environ[_k] = os.path.expandvars(os.environ[_k])

VARIABLES = [k for k, _ in _DEFAULTS]

DATA = os.environ["HEP4M_DATA"]
WORK = os.environ["HEP4M_WORK"]
TOKENIZED = os.environ["HEP4M_TOKENIZED"]
TOKENIZERS = os.environ["HEP4M_TOKENIZERS"]
RELEASE = os.environ["HEP4M_RELEASE"]
RAW_TEST = os.environ["HEP4M_RAW_TEST"]
CKPT = os.environ["HEP4M_CKPT"]
RUNS = os.environ["HEP4M_RUNS"]
OUTPUT_ROOT = os.environ["HEP4M_OUTPUT_ROOT"]
REPO = str(REPO)


def expand(s: str) -> str:
    """Expand ``${VAR}`` references in one string."""
    return os.path.expandvars(s)


def safe_load_expanded(stream):
    """``yaml.safe_load`` followed by ``${VAR}`` expansion of every string leaf."""
    import yaml

    from hep4m.utility.config_utils import expand_env_vars

    return expand_env_vars(yaml.safe_load(stream))


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", action="store_true", help="print shell export lines")
    a = ap.parse_args(argv)
    for k in VARIABLES:
        v = os.environ[k]
        print(f"export {k}='{v}'" if a.export else f"{k}={v}")
    if a.export:
        print(f"export PYTHON='{sys.executable}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
