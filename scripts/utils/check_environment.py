#!/usr/bin/env python3
"""Validate the local environment for the SWMI pipeline.

The default checks are deterministic and do not make live API requests. They
verify Python/package availability, required configuration files, scientific
invariants, expected directories, API credential presence, model-file policy,
and free disk space.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from swmi.utils import config
from swmi.utils.config import load_config, validate_scientific_invariants


@dataclass(frozen=True)
class CheckResult:
    """Result of one environment check."""

    name: str
    ok: bool
    message: str
    required: bool = True


_REQUIRED_IMPORTS = {
    "apexpy": "apexpy",
    "boto3": "boto3",
    "cdasws": "cdasws",
    "dask": "dask",
    "lightgbm": "lightgbm",
    "matplotlib": "matplotlib",
    "netCDF4": "netCDF4",
    "numpy": "numpy",
    "pandas": "pandas",
    "ppigrf": "ppigrf",
    "pyarrow": "pyarrow",
    "requests": "requests",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "viresclient": "viresclient",
    "xarray": "xarray",
    "xgboost": "xgboost",
    "yaml": "pyyaml",
}

_CONFIG_FILES = (
    "configs/data_retrieval.yaml",
    "configs/feature_engineering.yaml",
    "configs/model_baseline.yaml",
)


def check_python_version() -> CheckResult:
    minimum = (3, 12)
    current = sys.version_info[:3]
    ok = current >= minimum
    return CheckResult(
        "python",
        ok,
        f"Python {current[0]}.{current[1]}.{current[2]} detected; requires >= {minimum[0]}.{minimum[1]}",
    )


def check_imports(required_imports: dict[str, str] | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    for import_name, package_name in sorted((required_imports or _REQUIRED_IMPORTS).items()):
        try:
            importlib.import_module(import_name)
        except Exception as exc:
            results.append(CheckResult(f"dependency:{package_name}", False, str(exc)))
        else:
            results.append(CheckResult(f"dependency:{package_name}", True, "import ok"))
    return results


def check_configs(project_root: str | Path = _PROJECT_ROOT) -> list[CheckResult]:
    root = Path(project_root)
    results: list[CheckResult] = []
    loaded: dict[str, dict] = {}

    for relative_path in _CONFIG_FILES:
        path = root / relative_path
        if not path.exists():
            results.append(CheckResult(f"config:{relative_path}", False, "missing config file"))
            continue
        try:
            cfg = load_config(path)
            validate_scientific_invariants(cfg, relative_path)
        except Exception as exc:
            results.append(CheckResult(f"config:{relative_path}", False, str(exc)))
        else:
            loaded[relative_path] = cfg
            results.append(CheckResult(f"config:{relative_path}", True, "loaded and invariants ok"))

    data_cfg = loaded.get("configs/data_retrieval.yaml")
    if data_cfg is not None:
        cadence = data_cfg.get("master_cadence")
        results.append(
            CheckResult(
                "invariant:MASTER_CADENCE",
                cadence == config.MASTER_CADENCE,
                f"YAML={cadence!r}, Python={config.MASTER_CADENCE!r}",
            )
        )

    return results


def check_directories(project_root: str | Path = _PROJECT_ROOT) -> list[CheckResult]:
    root = Path(project_root)
    required_dirs = [
        root / config.RAW_DATA_DIR,
        root / config.PROCESSED_DIR,
        root / config.FEATURES_DIR,
        root / config.SEQUENCES_DIR,
        root / config.MODELS_DIR,
        root / config.LOGS_DIR,
        root / config.ARTIFACTS_DIR,
        root / config.DASK_TEMP_DIR,
        root / "data" / "external" / "station_metadata",
    ]
    return [
        CheckResult(f"directory:{path.relative_to(root)}", path.is_dir(), "exists" if path.is_dir() else "missing")
        for path in required_dirs
    ]


def check_api_credentials(env: dict[str, str] | None = None, *, strict_api: bool = True) -> list[CheckResult]:
    env = os.environ if env is None else env
    supermag_user = env.get("SUPERMAG_USERNAME", "").strip()
    vires_token_present = any(
        env.get(name, "").strip()
        for name in ("VIRES_TOKEN", "VIRES_ACCESS_TOKEN", "SWARM_TOKEN")
    )

    return [
        CheckResult(
            "credential:SUPERMAG_USERNAME",
            bool(supermag_user),
            "set" if supermag_user else "missing SUPERMAG_USERNAME",
            required=strict_api,
        ),
        CheckResult(
            "credential:VirES token",
            vires_token_present,
            "token env var present" if vires_token_present else "no VIRES_TOKEN/VIRES_ACCESS_TOKEN/SWARM_TOKEN env var; viresclient may still use its local token store",
            required=False,
        ),
    ]


def check_model_files(project_root: str | Path = _PROJECT_ROOT) -> list[CheckResult]:
    root = Path(project_root)
    chaos_candidates = [
        root / "models" / "CHAOS-8.5.mat",
        root / "data" / "models" / "CHAOS-8.5.mat",
    ]
    chaos_available = any(path.exists() for path in chaos_candidates)
    reference_field = config.REFERENCE_FIELD.upper()
    if reference_field == "CHAOS":
        return [
            CheckResult(
                "model:CHAOS-8.5",
                chaos_available,
                "found" if chaos_available else "required because REFERENCE_FIELD=CHAOS",
            )
        ]
    return [
        CheckResult(
            "model:CHAOS-8.5",
            True,
            "optional because REFERENCE_FIELD=IGRF" + ("; found local file" if chaos_available else ""),
            required=False,
        )
    ]


def check_disk_space(project_root: str | Path = _PROJECT_ROOT, min_free_gb: float = 5.0) -> CheckResult:
    usage = shutil.disk_usage(project_root)
    free_gb = usage.free / (1024**3)
    return CheckResult(
        "disk:project_root",
        free_gb >= min_free_gb,
        f"{free_gb:.1f} GiB free; requires >= {min_free_gb:.1f} GiB",
    )


def run_checks(
    project_root: str | Path = _PROJECT_ROOT,
    *,
    min_free_gb: float = 5.0,
    strict_api: bool = True,
) -> list[CheckResult]:
    """Run all environment checks and return structured results."""
    return [
        check_python_version(),
        *check_imports(),
        *check_configs(project_root),
        *check_directories(project_root),
        *check_api_credentials(strict_api=strict_api),
        *check_model_files(project_root),
        check_disk_space(project_root, min_free_gb=min_free_gb),
    ]


def environment_ok(results: list[CheckResult]) -> bool:
    """Return True if all required checks passed."""
    return all(result.ok or not result.required for result in results)


def print_report(results: list[CheckResult]) -> None:
    for result in results:
        if result.ok:
            status = "PASS"
        elif result.required:
            status = "FAIL"
        else:
            status = "WARN"
        print(f"[{status}] {result.name}: {result.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check SWMI pipeline environment.")
    parser.add_argument("--project-root", default=str(_PROJECT_ROOT))
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Report API credential status as non-blocking.",
    )
    args = parser.parse_args(argv)

    results = run_checks(
        args.project_root,
        min_free_gb=args.min_free_gb,
        strict_api=not args.skip_api,
    )
    print_report(results)
    return 0 if environment_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
