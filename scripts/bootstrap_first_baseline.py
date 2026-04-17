from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def git_head(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def ensure_repo(repo_dir: Path, url: str, commit: str, force_sync: bool) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        run(["git", "clone", url, str(repo_dir)])
    elif not (repo_dir / ".git").exists():
        raise RuntimeError(f"{repo_dir} exists but is not a git repository.")

    if force_sync:
        run(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
        run(["git", "checkout", commit], cwd=repo_dir)
        return

    head = git_head(repo_dir)
    if head == commit:
        return

    run(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
    run(["git", "checkout", commit], cwd=repo_dir)


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' is not on PATH.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone and pin the upstream repos needed for the first SAPS baseline.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "first_working_baseline.json"),
        help="Path to the first baseline JSON config.",
    )
    parser.add_argument(
        "--force-sync",
        action="store_true",
        help="Always fetch and reset the external repos to the pinned commits.",
    )
    args = parser.parse_args()

    ensure_tool("git")
    config = load_config(Path(args.config))

    llada_dir = ROOT / config["paths"]["llada_repo"]
    sparse_dir = ROOT / config["paths"]["sparse_repo"]

    ensure_repo(
        llada_dir,
        config["upstreams"]["llada"]["url"],
        config["upstreams"]["llada"]["commit"],
        args.force_sync,
    )
    ensure_repo(
        sparse_dir,
        config["upstreams"]["sparse_dllm"]["url"],
        config["upstreams"]["sparse_dllm"]["commit"],
        args.force_sync,
    )

    print("Bootstrap complete.")
    print(f"LLaDA:        {llada_dir} @ {git_head(llada_dir)}")
    print(f"Sparse-dLLM:  {sparse_dir} @ {git_head(sparse_dir)}")
    print("Next steps:")
    print("1. python -m pip install modal")
    print("2. modal setup")
    print("3. huggingface-cli login")
    print("4. python scripts/prepare_first_baseline.py")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
