from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import modal


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "first_working_baseline.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)

    embedded = os.environ.get("FIRST_BASELINE_CONFIG_JSON")
    if embedded:
        return json.loads(embedded)

    raise FileNotFoundError(f"Could not find config file at {CONFIG_PATH} or FIRST_BASELINE_CONFIG_JSON in environment.")


CFG = load_config()
MODAL_CFG = CFG["modal"]
RUNS_CFG = CFG["runs"]
LOCAL_WORKSPACE_ROOT = ROOT / CFG["paths"]["workspace_root"]
REMOTE_WORKSPACE_ROOT = PurePosixPath(MODAL_CFG["workspace_root"])
RUNTIME_REQUIREMENTS = ROOT / "external" / "LLaDA" / "opencompass" / "requirements" / "runtime.txt"
HF_HOME = PurePosixPath(MODAL_CFG["hf_home"])
RESULTS_ROOT = PurePosixPath(MODAL_CFG["results_root"])

hf_cache_volume = modal.Volume.from_name(MODAL_CFG["hf_cache_volume"], create_if_missing=True)
results_volume = modal.Volume.from_name(MODAL_CFG["results_volume"], create_if_missing=True)


def ensure_prepared() -> None:
    missing = [name for name, run in RUNS_CFG.items() if not (LOCAL_WORKSPACE_ROOT / run["workspace"]).exists()]
    if missing:
        command = [sys.executable, str(ROOT / "scripts" / "prepare_first_baseline.py"), "--config", str(CONFIG_PATH)]
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to prepare workspaces with {' '.join(command)}")


if CONFIG_PATH.exists():
    ensure_prepared()


def build_image(baseline: str) -> modal.Image:
    packages = MODAL_CFG[f"{baseline}_packages"]
    workspace_name = RUNS_CFG[baseline]["workspace"]
    local_workspace = LOCAL_WORKSPACE_ROOT / workspace_name
    remote_workspace = REMOTE_WORKSPACE_ROOT / workspace_name

    return (
        modal.Image.debian_slim(python_version=MODAL_CFG["python_version"])
        .apt_install("git")
        .pip_install_from_requirements(str(RUNTIME_REQUIREMENTS))
        .pip_install(*packages)
        .env(
            {
                "FIRST_BASELINE_CONFIG_JSON": json.dumps(CFG),
                "HF_HOME": str(HF_HOME),
                "HF_HUB_CACHE": str(HF_HOME / "hub"),
                "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            }
        )
        .add_local_dir(
            str(local_workspace),
            str(remote_workspace),
            copy=True,
            ignore=[".git", "__pycache__", "outputs", ".pytest_cache"],
        )
        .add_local_dir(
            str(ROOT / "saps"),
            "/opt/saps_src/saps",
            copy=True,
            ignore=["__pycache__"],
        )
        .add_local_file(str(ROOT / "scripts" / "profile_saps.py"), "/opt/profile_saps.py", copy=True)
        .run_commands(f"cd {remote_workspace.as_posix()} && python -m pip install -e .")
        .run_commands(f"cd {remote_workspace.as_posix()} && [ -d saps ] && python -m pip install -e saps/ || true")
    )


app = modal.App(MODAL_CFG["app_name"])
vanilla_image = build_image("vanilla")
sparse_image = build_image("sparse")
saps_image = build_image("saps")


def build_remote_command(
    baseline: str,
    smoke: bool = False,
    dev: bool = False,
    full: bool = False,
    reuse: str | None = None,
) -> tuple[list[str], PurePosixPath, PurePosixPath]:
    run_cfg = RUNS_CFG[baseline]
    workspace = REMOTE_WORKSPACE_ROOT / run_cfg["workspace"]
    if smoke:
        config_name = run_cfg["smoke_config"]
        work_dir_name = run_cfg["smoke_work_dir"]
    elif dev:
        config_name = run_cfg["dev_config"]
        work_dir_name = run_cfg["dev_work_dir"]
    elif full:
        config_name = run_cfg["full_config"]
        work_dir_name = run_cfg["full_work_dir"]
    else:
        config_name = run_cfg["config"]
        work_dir_name = run_cfg["work_dir"]
    work_dir = RESULTS_ROOT / work_dir_name
    command = ["python", "run.py", config_name, "-w", str(work_dir)]
    if reuse:
        command.extend(["-r", reuse])
    return command, workspace, work_dir


def _tee_stream(stream, destination: Path, prefix: str) -> None:
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        for line in iter(stream.readline, ""):
            handle.write(line)
            handle.flush()
            print(f"[{prefix}] {line.rstrip()}", flush=True)
    stream.close()


def _extract_progress(lines: list[str]) -> dict | None:
    progress_pattern = re.compile(r"(\d+)%.*?(\d+)\s*/\s*(\d+).*(it/s|\?\sit/s)")

    for line in reversed(lines):
        match = progress_pattern.search(line)
        if not match:
            continue
        percent = int(match.group(1))
        completed = int(match.group(2))
        total = int(match.group(3))
        return {
            "completed": completed,
            "total": total,
            "percent": percent,
            "line": line,
        }
    return None


def _extract_gsm8k_answer(text: str) -> str | None:
    """Extract the numeric answer from a GSM8K gold or prediction string."""
    import re
    # Gold format: '#### 42' or '#### 1,234'
    m = re.search(r"####\s*([\d,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last standalone integer in the text
    nums = re.findall(r"\b\d[\d,]*\b", text)
    return nums[-1].replace(",", "") if nums else None


def _compute_checkpoint_accuracy(work_dir: Path) -> dict | None:
    """Read all completed prediction shards and compute partial GSM8K accuracy."""
    pred_dirs = list(work_dir.glob("*/predictions"))
    if not pred_dirs:
        return None
    pred_dir = pred_dirs[-1]  # latest run
    shard_files = sorted(pred_dir.rglob("gsm8k*.json"))
    # exclude the eval results file (which lives under results/, not predictions/)
    shard_files = [p for p in shard_files if "predictions" in p.parts]
    if not shard_files:
        return None

    total, correct = 0, 0
    for shard_path in shard_files:
        try:
            shard = json.loads(shard_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for entry in shard.values():
            gold_raw = entry.get("gold", "")
            pred_raw = entry.get("prediction", "")
            if not gold_raw:
                continue
            gold_ans = _extract_gsm8k_answer(str(gold_raw))
            pred_ans = _extract_gsm8k_answer(str(pred_raw))
            total += 1
            if gold_ans and pred_ans and gold_ans == pred_ans:
                correct += 1

    if total == 0:
        return None
    return {
        "examples_scored": total,
        "correct": correct,
        "partial_accuracy": round(correct / total * 100, 2),
        "shards_read": len(shard_files),
    }


# Checkpoint thresholds: log accuracy at 25 / 50 / 75 / 100 %
_CHECKPOINT_PCTS = [25, 50, 75, 100]
_FULL_DATASET_SIZE = 1319


def _maybe_record_checkpoint(
    checkpoint_data: dict,
    existing_checkpoints: list[dict],
    total_examples: int = _FULL_DATASET_SIZE,
) -> list[dict]:
    """Append a checkpoint entry if we've crossed a new percentage threshold."""
    scored = checkpoint_data.get("examples_scored", 0)
    pct_done = scored / total_examples * 100
    recorded_pcts = {c["target_pct"] for c in existing_checkpoints}
    new_checkpoints = list(existing_checkpoints)
    for threshold in _CHECKPOINT_PCTS:
        if pct_done >= threshold and threshold not in recorded_pcts:
            new_checkpoints.append({
                "target_pct": threshold,
                "examples_scored": scored,
                "partial_accuracy": checkpoint_data["partial_accuracy"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(
                f"[checkpoint] {threshold}% ({scored}/{total_examples} examples): "
                f"accuracy={checkpoint_data['partial_accuracy']:.2f}%",
                flush=True,
            )
    return new_checkpoints


def _collect_progress_snapshot(work_dir: Path) -> dict:
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "work_dir_exists": work_dir.exists(),
    }
    if not work_dir.exists():
        return snapshot

    infer_logs = sorted(work_dir.glob("*/logs/infer/**/*.out"))
    stderr_log = work_dir / "stderr.log"
    stdout_log = work_dir / "stdout.log"

    snapshot["top_level_entries"] = sorted(path.name for path in work_dir.iterdir())
    snapshot["stdout_log_bytes"] = stdout_log.stat().st_size if stdout_log.exists() else 0
    snapshot["stderr_log_bytes"] = stderr_log.stat().st_size if stderr_log.exists() else 0
    run_dirs = sorted(path.name for path in work_dir.iterdir() if re.fullmatch(r"\d{8}_\d{6}", path.name))
    snapshot["run_dirs"] = run_dirs
    if run_dirs:
        snapshot["latest_run_dir"] = run_dirs[-1]

    if infer_logs:
        snapshot["infer_log_count"] = len(infer_logs)
        latest = infer_logs[-1]
        try:
            latest_lines = latest.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            latest_lines = []
        tail_lines = latest_lines[-10:]
        snapshot["latest_infer_log"] = str(latest)
        snapshot["latest_infer_log_tail"] = tail_lines
        progress = _extract_progress(latest_lines[-100:])
        if progress:
            snapshot["latest_infer_progress"] = progress

        progress_summaries = []
        for infer_log in infer_logs[-8:]:
            try:
                lines = infer_log.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            parsed = _extract_progress(lines[-100:])
            if parsed:
                progress_summaries.append(
                    {
                        "log": str(infer_log),
                        "completed": parsed["completed"],
                        "total": parsed["total"],
                        "percent": parsed["percent"],
                    }
                )
        if progress_summaries:
            snapshot["infer_progress"] = progress_summaries

    return snapshot


def _heartbeat_loop(
    process: subprocess.Popen,
    work_dir: Path,
    heartbeat_path: Path,
    interval_seconds: int = 60,
    full_run: bool = False,
) -> None:
    checkpoints: list[dict] = []
    checkpoint_path = work_dir / "accuracy_checkpoints.json"
    while process.poll() is None:
        snapshot = _collect_progress_snapshot(work_dir)
        if full_run:
            checkpoint_data = _compute_checkpoint_accuracy(work_dir)
            if checkpoint_data:
                snapshot["checkpoint_accuracy"] = checkpoint_data
                checkpoints = _maybe_record_checkpoint(checkpoint_data, checkpoints)
                snapshot["accuracy_checkpoints"] = checkpoints
                checkpoint_path.write_text(
                    json.dumps(checkpoints, indent=2), encoding="utf-8", newline="\n"
                )
        heartbeat_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8", newline="\n")
        results_volume.commit()
        print(f"[heartbeat] {json.dumps(snapshot)}", flush=True)
        time.sleep(interval_seconds)


def run_remote_baseline(
    baseline: str,
    smoke: bool = False,
    dev: bool = False,
    full: bool = False,
    reuse: str | None = None,
) -> dict:
    command, workspace, work_dir = build_remote_command(baseline, smoke=smoke, dev=dev, full=full, reuse=reuse)
    output_dir = Path(str(work_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = output_dir / "stdout.log"
    stderr_log = output_dir / "stderr.log"
    heartbeat_log = output_dir / "heartbeat.json"

    metadata = {
        "baseline": baseline,
        "smoke": smoke,
        "dev": dev,
        "reuse": reuse,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "work_dir": str(work_dir),
        "command": command,
        "model_path": CFG["model_path"],
        "hf_home": str(HF_HOME),
        "modal_gpu": MODAL_CFG["gpu"],
    }

    (output_dir / "remote_run_request.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    stdout_log.write_text("", encoding="utf-8", newline="\n")
    stderr_log.write_text("", encoding="utf-8", newline="\n")
    heartbeat_log.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "starting",
                "command": command,
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    results_volume.commit()

    process = subprocess.Popen(
        command,
        cwd=workspace,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_thread = threading.Thread(
        target=_tee_stream,
        args=(process.stdout, stdout_log, "stdout"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_tee_stream,
        args=(process.stderr, stderr_log, "stderr"),
        daemon=True,
    )
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(process, output_dir, heartbeat_log),
        kwargs={"full_run": full},
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()
    heartbeat_thread.start()

    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    heartbeat_thread.join(timeout=1)

    metadata["return_code"] = return_code
    metadata["status"] = "completed" if return_code == 0 else "failed"
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    heartbeat_log.write_text(
        json.dumps(
            {
                **_collect_progress_snapshot(output_dir),
                "status": metadata["status"],
                "return_code": return_code,
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "remote_run_result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")
    results_volume.commit()

    if return_code != 0:
        stderr_tail = "\n".join(stderr_log.read_text(encoding="utf-8", errors="replace").splitlines()[-50:])
        raise RuntimeError(f"{baseline} baseline failed with return code {process.returncode}\n{stderr_tail}")

    return {
        "baseline": baseline,
        "smoke": smoke,
        "work_dir": str(work_dir),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "heartbeat_log": str(heartbeat_log),
        "result_metadata": str(output_dir / "remote_run_result.json"),
    }


def probe_remote_baseline(
    baseline: str,
    smoke: bool = False,
    dev: bool = False,
    reuse: str | None = None,
) -> dict:
    command, workspace, work_dir = build_remote_command(baseline, smoke=smoke, dev=dev, reuse=reuse)
    config_path = Path(str(workspace)) / command[2]
    return {
        "baseline": baseline,
        "smoke": smoke,
        "dev": dev,
        "reuse": reuse,
        "workspace": str(workspace),
        "workspace_exists": Path(str(workspace)).exists(),
        "config_exists": config_path.exists(),
        "config_path": str(config_path),
        "hf_home": str(HF_HOME),
        "hf_home_exists": Path(str(HF_HOME)).exists(),
        "results_mount_exists": Path("/vol").exists(),
        "results_dir": str(work_dir),
        "python": sys.version,
    }


def build_profile_command(baseline: str, dataset_size: str = "smoke") -> tuple[list[str], PurePosixPath, PurePosixPath]:
    """Build profiling command with configurable dataset size.
    
    Args:
        baseline: vanilla, sparse, or saps
        dataset_size: "smoke" (4), "dev" (128), or "full" (all)
    """
    run_cfg = RUNS_CFG[baseline]
    workspace = REMOTE_WORKSPACE_ROOT / run_cfg["workspace"]
    
    # Determine prompt limit based on dataset size
    if dataset_size == "smoke":
        prompt_limit = 4
        size_tag = "smoke"
    elif dataset_size == "dev":
        prompt_limit = 128
        size_tag = "dev"
    else:  # full
        prompt_limit = 999999  # effectively unlimited
        size_tag = "full"
    
    output_path = RESULTS_ROOT / "profiling" / f"{baseline}_gsm8k_{size_tag}_profile.json"
    command = [
        "python",
        "/opt/profile_saps.py",
        "--baseline",
        baseline,
        "--workspace",
        str(workspace),
        "--model-path",
        CFG["model_path"],
        "--gsm8k-smoke",
        "--prompt-limit",
        str(prompt_limit),
        "--apply-chat-template",
        "--steps",
        "256",
        "--gen-length",
        "256",
        "--block-length",
        "32",
        "--output",
        str(output_path),
    ]
    
    # Add baseline-specific tuning parameters
    if baseline == "sparse":
        command.extend([
            "--keep-ratio", "0.5",
        ])
    elif baseline == "saps":
        command.extend([
            "--r-max", "0.7",
            "--r-min", "0.1",
            "--decay-type", "exp",
        ])
    
    return command, workspace, output_path



def run_remote_profile(baseline: str, dataset_size: str = "smoke") -> dict:
    command, workspace, output_path = build_profile_command(baseline, dataset_size=dataset_size)
    output_dir = Path(str(RESULTS_ROOT / "profiling" / baseline))
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = output_dir / "stdout.log"
    stderr_log = output_dir / "stderr.log"

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    extra_pythonpath = "/opt/saps_src"
    env["PYTHONPATH"] = f"{extra_pythonpath}:{existing_pythonpath}" if existing_pythonpath else extra_pythonpath

    process = subprocess.Popen(
        command,
        cwd=workspace,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_thread = threading.Thread(target=_tee_stream, args=(process.stdout, stdout_log, "stdout"), daemon=True)
    stderr_thread = threading.Thread(target=_tee_stream, args=(process.stderr, stderr_log, "stderr"), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    if return_code != 0:
        stderr_tail = "\n".join(stderr_log.read_text(encoding="utf-8", errors="replace").splitlines()[-50:])
        raise RuntimeError(f"{baseline} profiling failed with return code {return_code}\n{stderr_tail}")

    report_path = Path(str(output_path))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs = report.get("runs", [])
    peak_total_memories = [
        run.get("peak_total_gib", run.get("peak_memory_gb"))
        for run in runs
        if run.get("peak_total_gib", run.get("peak_memory_gb")) is not None
    ]
    avg_kv_memories = [run["avg_kv_cache_gib"] for run in runs if run.get("avg_kv_cache_gib") is not None]
    elapsed = [run["elapsed_seconds"] for run in runs]
    stability_values = [
        run["profile"]["summary"]["stability"]["avg_consecutive_jaccard"]
        for run in runs
        if run.get("profile") is not None and run["profile"]["summary"]["stability"]["avg_consecutive_jaccard"] is not None
    ]
    anchor_values = [
        run["profile"]["summary"]["stability"]["avg_early_anchor_share"]
        for run in runs
        if run.get("profile") is not None and run["profile"]["summary"]["stability"]["avg_early_anchor_share"] is not None
    ]
    late_values = [
        run["profile"]["summary"]["stability"]["avg_late_token_share"]
        for run in runs
        if run.get("profile") is not None and run["profile"]["summary"]["stability"]["avg_late_token_share"] is not None
    ]

    def _avg(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "baseline": baseline,
        "workspace": str(workspace),
        "report_path": str(report_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "avg_peak_total_gib": _avg(peak_total_memories),
        "avg_peak_memory_gb": _avg(peak_total_memories),
        "avg_kv_cache_gib": _avg(avg_kv_memories),
        "avg_elapsed_seconds": _avg(elapsed),
        "avg_consecutive_jaccard": _avg(stability_values),
        "avg_early_anchor_share": _avg(anchor_values),
        "avg_late_token_share": _avg(late_values),
    }


@app.function(
    image=vanilla_image,
    gpu=MODAL_CFG["gpu"],
    timeout=MODAL_CFG["timeout_seconds"],
    startup_timeout=MODAL_CFG["startup_timeout_seconds"],
    volumes={
        HF_HOME: hf_cache_volume,
        PurePosixPath("/vol"): results_volume,
    },
)
def run_vanilla(smoke: bool = False, dev: bool = False, full: bool = False, reuse: str | None = None) -> dict:
    return run_remote_baseline("vanilla", smoke=smoke, dev=dev, full=full, reuse=reuse)


@app.function(
    image=sparse_image,
    gpu=MODAL_CFG["gpu"],
    timeout=MODAL_CFG["timeout_seconds"],
    startup_timeout=MODAL_CFG["startup_timeout_seconds"],
    volumes={
        HF_HOME: hf_cache_volume,
        PurePosixPath("/vol"): results_volume,
    },
)
def run_sparse(smoke: bool = False, dev: bool = False, full: bool = False, reuse: str | None = None) -> dict:
    return run_remote_baseline("sparse", smoke=smoke, dev=dev, full=full, reuse=reuse)


@app.function(
    image=saps_image,
    gpu=MODAL_CFG["gpu"],
    timeout=MODAL_CFG["timeout_seconds"],
    startup_timeout=MODAL_CFG["startup_timeout_seconds"],
    volumes={
        HF_HOME: hf_cache_volume,
        PurePosixPath("/vol"): results_volume,
    },
)
def run_saps(smoke: bool = False, dev: bool = False, full: bool = False, reuse: str | None = None) -> dict:
    return run_remote_baseline("saps", smoke=smoke, dev=dev, full=full, reuse=reuse)


@app.function(
    image=vanilla_image,
    timeout=600,
    volumes={
        HF_HOME: hf_cache_volume,
        PurePosixPath("/vol"): results_volume,
    },
)
def probe_vanilla(smoke: bool = False, dev: bool = False, reuse: str | None = None) -> dict:
    return probe_remote_baseline("vanilla", smoke=smoke, dev=dev, reuse=reuse)


@app.function(
    image=sparse_image,
    timeout=600,
    volumes={
        HF_HOME: hf_cache_volume,
        PurePosixPath("/vol"): results_volume,
    },
)
def probe_sparse(smoke: bool = False, dev: bool = False, reuse: str | None = None) -> dict:
    return probe_remote_baseline("sparse", smoke=smoke, dev=dev, reuse=reuse)


@app.function(
    image=saps_image,
    timeout=600,
    volumes={
        HF_HOME: hf_cache_volume,
        PurePosixPath("/vol"): results_volume,
    },
)
def probe_saps(smoke: bool = False, dev: bool = False, reuse: str | None = None) -> dict:
    return probe_remote_baseline("saps", smoke=smoke, dev=dev, reuse=reuse)


@app.function(
    image=vanilla_image,
    gpu=MODAL_CFG["gpu"],
    timeout=MODAL_CFG["timeout_seconds"],
    startup_timeout=MODAL_CFG["startup_timeout_seconds"],
    volumes={
        HF_HOME: hf_cache_volume,
        PurePosixPath("/vol"): results_volume,
    },
)
def profile_vanilla(dataset_size: str = "smoke") -> dict:
    return run_remote_profile("vanilla", dataset_size=dataset_size)


@app.function(
    image=sparse_image,
    gpu=MODAL_CFG["gpu"],
    timeout=MODAL_CFG["timeout_seconds"],
    startup_timeout=MODAL_CFG["startup_timeout_seconds"],
    volumes={
        HF_HOME: hf_cache_volume,
        PurePosixPath("/vol"): results_volume,
    },
)
def profile_sparse(dataset_size: str = "smoke") -> dict:
    return run_remote_profile("sparse", dataset_size=dataset_size)


@app.function(
    image=saps_image,
    gpu=MODAL_CFG["gpu"],
    timeout=MODAL_CFG["timeout_seconds"],
    startup_timeout=MODAL_CFG["startup_timeout_seconds"],
    volumes={
        HF_HOME: hf_cache_volume,
        PurePosixPath("/vol"): results_volume,
    },
)
def profile_saps(dataset_size: str = "smoke") -> dict:
    return run_remote_profile("saps", dataset_size=dataset_size)


@app.local_entrypoint()
def main(
    baseline: str = "vanilla",
    dry_run: bool = False,
    prepare: bool = True,
    smoke: bool = False,
    dev: bool = False,
    full: bool = False,
    reuse: str | None = None,
    probe: bool = False,
    profile: bool = False,
    profile_dataset: str = "smoke",
) -> None:
    if prepare:
        ensure_prepared()

    if baseline not in RUNS_CFG:
        raise ValueError(f"Unknown baseline {baseline!r}. Expected one of {sorted(RUNS_CFG)}")
    mode_flags = sum([smoke, dev, full])
    if mode_flags > 1:
        raise ValueError("Use only one of --smoke, --dev, or --full.")
    if profile and (smoke or dev or full or reuse is not None):
        raise ValueError("Profiling mode does not use smoke/dev/full/reuse flags. Use --profile-dataset instead.")
    if profile_dataset not in ("smoke", "dev", "full"):
        raise ValueError(f"Invalid profile_dataset: {profile_dataset!r}. Expected one of: smoke, dev, full.")
    if full and baseline == "vanilla":
        print(
            "[warn] vanilla --full will run all 1319 GSM8K examples. "
            "Vanilla uses steps=512 vs sparse/SAPS steps=256, so expect ~30h on A100. "
            "Consider skipping vanilla full eval for the final report."
        )

    command, workspace, work_dir = build_remote_command(baseline, smoke=smoke, dev=dev, full=full, reuse=reuse)
    payload = {
        "baseline": baseline,
        "smoke": smoke,
        "dev": dev,
        "full": full,
        "reuse": reuse,
        "local_workspace": str((LOCAL_WORKSPACE_ROOT / RUNS_CFG[baseline]["workspace"]).resolve()),
        "remote_workspace": str(workspace),
        "command": command,
        "remote_results_dir": str(work_dir),
        "hf_cache_volume": MODAL_CFG["hf_cache_volume"],
        "results_volume": MODAL_CFG["results_volume"],
        "gpu": MODAL_CFG["gpu"],
        "packages": MODAL_CFG[f"{baseline}_packages"],
    }

    print(json.dumps(payload, indent=2))

    if dry_run:
        volume_relative = work_dir.relative_to("/vol").as_posix()
        print(
            f"To fetch results later: modal volume get {MODAL_CFG['results_volume']} "
            f"{volume_relative} <local-path>"
        )
        return

    if profile:
        if baseline == "vanilla":
            result = profile_vanilla.remote(dataset_size=profile_dataset)
        elif baseline == "sparse":
            result = profile_sparse.remote(dataset_size=profile_dataset)
        else:
            result = profile_saps.remote(dataset_size=profile_dataset)
        print(json.dumps(result, indent=2))
        return

    if probe:
        if baseline == "vanilla":
            result = probe_vanilla.remote(smoke=smoke, dev=dev, reuse=reuse)
        elif baseline == "sparse":
            result = probe_sparse.remote(smoke=smoke, dev=dev, reuse=reuse)
        else:
            result = probe_saps.remote(smoke=smoke, dev=dev, reuse=reuse)
        print(json.dumps(result, indent=2))
        return

    if baseline == "vanilla":
        result = run_vanilla.remote(smoke=smoke, dev=dev, full=full, reuse=reuse)
    elif baseline == "sparse":
        result = run_sparse.remote(smoke=smoke, dev=dev, full=full, reuse=reuse)
    else:
        result = run_saps.remote(smoke=smoke, dev=dev, full=full, reuse=reuse)

    print(json.dumps(result, indent=2))
