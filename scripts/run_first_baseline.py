from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def git_rev_parse(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def ensure_results_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_command(cfg: dict, baseline: str, smoke: bool, dev: bool, reuse: str | None) -> tuple[list[str], Path, Path]:
    run_cfg = cfg["runs"][baseline]
    workspace = ROOT / cfg["paths"]["workspace_root"] / run_cfg["workspace"]
    if smoke:
        config_name = run_cfg["smoke_config"]
        work_dir_name = run_cfg["smoke_work_dir"]
    elif dev:
        config_name = run_cfg["dev_config"]
        work_dir_name = run_cfg["dev_work_dir"]
    else:
        config_name = run_cfg["config"]
        work_dir_name = run_cfg["work_dir"]
    work_dir = ROOT / cfg["paths"]["results_root"] / work_dir_name
    ensure_results_dir(work_dir)
    command = [
        cfg["python"][baseline],
        "run.py",
        config_name,
        "-w",
        str(work_dir.resolve()),
    ]
    if reuse:
        command.extend(["-r", reuse])
    return command, workspace, work_dir


def write_run_metadata(metadata_path: Path, payload: dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first vanilla or Sparse-dLLM baseline with exact command capture.")
    parser.add_argument("baseline", choices=["vanilla", "sparse"], help="Which first-baseline run to launch.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "first_working_baseline.json"),
        help="Path to the first baseline JSON config.",
    )
    parser.add_argument("--dev", action="store_true", help="Run the 128-example GSM8K dev config instead of the full baseline config.")
    parser.add_argument("--smoke", action="store_true", help="Run the 4-example GSM8K smoke config instead of the full baseline config.")
    parser.add_argument("--reuse", type=str, default=None, help="Reuse an existing OpenCompass timestamped run directory, e.g. 20260417_224243.")
    parser.add_argument("--dry-run", action="store_true", help="Print and record the command without executing it.")
    args = parser.parse_args()

    if args.dev and args.smoke:
        raise SystemExit("Use only one of --dev or --smoke.")

    cfg = load_config(Path(args.config))
    command, workspace, work_dir = build_command(cfg, args.baseline, args.smoke, args.dev, args.reuse)

    if not workspace.exists():
        raise RuntimeError(
            f"Workspace {workspace} does not exist. Run scripts/prepare_first_baseline.py first."
        )

    metadata = {
        "baseline": args.baseline,
        "smoke": args.smoke,
        "dev": args.dev,
        "reuse": args.reuse,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace.resolve()),
        "work_dir": str(work_dir.resolve()),
        "command": command,
        "model_path": cfg["model_path"],
        "source_repos": {
            "llada": {
                "path": str((ROOT / cfg["paths"]["llada_repo"]).resolve()),
                "commit": git_rev_parse(ROOT / cfg["paths"]["llada_repo"]),
            },
            "sparse_dllm": {
                "path": str((ROOT / cfg["paths"]["sparse_repo"]).resolve()),
                "commit": git_rev_parse(ROOT / cfg["paths"]["sparse_repo"]),
            },
        },
    }

    metadata_path = work_dir / "run_metadata.json"

    if args.dry_run:
        metadata["status"] = "dry_run"
        write_run_metadata(metadata_path, metadata)
        print("Dry run command:")
        print(" ".join(command))
        print(f"Metadata written to {metadata_path.resolve()}")
        return

    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=workspace, check=False)
    finished = datetime.now(timezone.utc).isoformat()

    metadata["status"] = "completed" if completed.returncode == 0 else "failed"
    metadata["started_utc"] = started
    metadata["finished_utc"] = finished
    metadata["return_code"] = completed.returncode
    write_run_metadata(metadata_path, metadata)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    print(f"Run finished successfully. Metadata written to {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
