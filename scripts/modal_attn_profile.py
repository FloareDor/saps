"""
Standalone Modal script: per-step, per-layer attention entropy profiling.

Runs SAPS-exp inference on 5 GSM8K (math) + 5 HumanEval (code) examples
and records H(t, l) — attention entropy at each denoising step t and
transformer layer l — to characterize whether code and math tasks show
qualitatively different attention patterns across the denoising trajectory.

Usage:
    modal run -d scripts/modal_attn_profile.py
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import modal


ROOT = Path(__file__).resolve().parents[1]
SAPS_WORKSPACE = ROOT / "workspaces" / "opencompass_saps"
REMOTE_WORKSPACE = PurePosixPath("/workspace/opencompass_saps")
RUNTIME_REQUIREMENTS = (
    ROOT / "external" / "LLaDA" / "opencompass" / "requirements" / "runtime.txt"
)
HF_HOME = PurePosixPath("/cache/huggingface")
RESULTS_ROOT = PurePosixPath("/vol/results/first_working_baseline")

hf_cache_volume = modal.Volume.from_name("saps-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "saps-first-baseline-results", create_if_missing=True
)

profile_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install_from_requirements(str(RUNTIME_REQUIREMENTS))
    .pip_install(
        "torch==2.6.0",
        "transformers==4.46.3",
        "datasets",
        "hf_xet",
    )
    .env(
        {
            "HF_HOME": str(HF_HOME),
            "HF_HUB_CACHE": str(HF_HOME / "hub"),
            "TRANSFORMERS_CACHE": str(HF_HOME / "hub"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    .add_local_dir(
        str(SAPS_WORKSPACE),
        str(REMOTE_WORKSPACE),
        copy=True,
        ignore=[".git", "__pycache__", "outputs", ".pytest_cache"],
    )
    .add_local_dir(
        str(ROOT / "saps"),
        "/opt/saps_src/saps",
        copy=True,
        ignore=["__pycache__"],
    )
    .run_commands(f"cd {REMOTE_WORKSPACE} && python -m pip install -e .")
)

app = modal.App("saps-attn-profile")


@app.function(
    image=profile_image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={
        str(HF_HOME.parent): hf_cache_volume,
        "/vol": results_volume,
    },
)
def run_attn_profile():
    import sys
    import json
    import torch
    from pathlib import Path
    from transformers import AutoConfig, AutoTokenizer
    from datasets import load_dataset

    sys.path.insert(0, "/workspace/opencompass_saps")
    sys.path.insert(0, "/opt/saps_src")

    from opencompass.models.sparse_dllm.modeling_llada import LLaDAModelLM
    from opencompass.models.sparse_dllm.llada_generate import generate
    from saps.schedule import SAPSScheduleConfig
    from saps.ratio_controller import RatioController

    MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"
    N_EXAMPLES = 5
    STEPS = 256
    GEN_LENGTH = 256
    BLOCK_LENGTH = 32

    print("Loading model...", flush=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config.block_len = BLOCK_LENGTH
    config.kernel_size = 3
    config.keep_ratio = 0.5
    model = LLaDAModelLM.from_pretrained(
        MODEL_PATH,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    n_layers = getattr(model.config, "n_layers", 32)
    print(f"Model loaded. n_layers={n_layers}", flush=True)

    # Load examples
    gsm8k_rows = list(load_dataset("gsm8k", "main", split=f"test[:{N_EXAMPLES}]"))
    gsm8k_prompts = [row["question"] for row in gsm8k_rows]

    humaneval_rows = list(load_dataset("openai_humaneval", split=f"test[:{N_EXAMPLES}]"))
    humaneval_prompts = [row["prompt"] for row in humaneval_rows]

    tasks = [
        ("gsm8k", gsm8k_prompts),
        ("humaneval", humaneval_prompts),
    ]

    all_examples = []

    for task_name, prompts in tasks:
        for ex_id, prompt in enumerate(prompts):
            print(f"\n[{task_name}] example {ex_id}", flush=True)
            print(f"  prompt: {prompt[:80].strip()}...", flush=True)

            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            input_ids = tokenizer(rendered, return_tensors="pt")[
                "input_ids"
            ].to(model.device)

            # profile_attention=True records H(t,l) for every filter_cache call
            saps_cfg = SAPSScheduleConfig(
                r_max=0.7, r_min=0.1, decay_type="exp", profile_attention=True
            )
            ratio_controller = RatioController(saps_cfg, n_layers=n_layers)

            torch.cuda.empty_cache()
            with torch.no_grad():
                generate(
                    model,
                    input_ids,
                    steps=STEPS,
                    gen_length=GEN_LENGTH,
                    block_length=BLOCK_LENGTH,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=126336,
                    ratio_controller=ratio_controller,
                )

            log = ratio_controller.get_attention_log()
            print(f"  → {len(log)} entropy records", flush=True)
            all_examples.append(
                {
                    "task": task_name,
                    "example_id": ex_id,
                    "prompt_snippet": prompt[:120],
                    "n_records": len(log),
                    "records": log,
                }
            )

    # Compute summary: mean H per (task, step) and mean H per (task, layer)
    summary = {}
    for task_name in ("gsm8k", "humaneval"):
        task_examples = [e for e in all_examples if e["task"] == task_name]
        from collections import defaultdict

        by_step: dict[int, list[float]] = defaultdict(list)
        by_layer: dict[int, list[float]] = defaultdict(list)
        for ex in task_examples:
            for rec in ex["records"]:
                by_step[rec["step"]].append(rec["entropy"])
                by_layer[rec["layer"]].append(rec["entropy"])
        summary[task_name] = {
            "mean_H_by_step": {
                str(s): sum(v) / len(v) for s, v in sorted(by_step.items())
            },
            "mean_H_by_layer": {
                str(l): sum(v) / len(v) for l, v in sorted(by_layer.items())
            },
        }

    out_path = Path(
        "/vol/results/first_working_baseline/attn_profile/attn_entropy_profile.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "model": MODEL_PATH,
            "saps_config": {"r_max": 0.7, "r_min": 0.1, "decay_type": "exp"},
            "steps": STEPS,
            "gen_length": GEN_LENGTH,
            "block_length": BLOCK_LENGTH,
            "n_examples_per_task": N_EXAMPLES,
        },
        "summary": summary,
        "examples": all_examples,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote profile to {out_path}", flush=True)
    results_volume.commit()


@app.local_entrypoint()
def main():
    run_attn_profile.remote()
