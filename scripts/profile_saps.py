from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saps import RatioController, SAPSProfiler, SAPSScheduleConfig


def load_prompts(prompt: str | None, prompt_file: str | None) -> list[str]:
    if prompt_file is not None:
        return [line.strip() for line in Path(prompt_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if prompt is not None:
        return [prompt]
    raise ValueError("Provide either --prompt or --prompt-file")


def load_gsm8k_questions(limit: int) -> list[str]:
    dataset = load_dataset("gsm8k", "main", split=f"test[:{limit}]")
    return [str(row["question"]).strip() for row in dataset]


def build_ratio_controller(method: str, keep_ratio: float, r_max: float, r_min: float, decay_type: str) -> tuple[RatioController | None, dict]:
    profiler = SAPSProfiler()
    if method == "sparse":
        cfg = SAPSScheduleConfig.fixed(keep_ratio)
        return RatioController(cfg, profiler=profiler), cfg.to_dict()
    cfg = SAPSScheduleConfig(r_max=r_max, r_min=r_min, decay_type=decay_type)
    return RatioController(cfg, profiler=profiler), cfg.to_dict()


def load_vanilla_runtime(workspace: Path):
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from generate import generate as vanilla_generate

    return vanilla_generate


def load_sparse_runtime(workspace: Path):
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from opencompass.models.sparse_dllm.llada_generate import generate as sparse_generate
    from opencompass.models.sparse_dllm.modeling_llada import LLaDAModelLM

    return sparse_generate, LLaDAModelLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile vanilla/sparse/SAPS memory and pruning stability on shared prompts.")
    parser.add_argument("--baseline", choices=["vanilla", "sparse", "saps"], default="saps")
    parser.add_argument("--workspace")
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--gsm8k-smoke", action="store_true", help="Use the first 4 GSM8K test questions as prompts.")
    parser.add_argument("--prompt-limit", type=int, default=4)
    parser.add_argument("--output", default=str(ROOT / "results" / "profiling" / "saps_profile.json"))
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--r-max", type=float, default=0.8)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--decay-type", choices=["constant", "linear", "cosine", "exp"], default="exp")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--apply-chat-template", action="store_true")
    args = parser.parse_args()

    default_workspace = ROOT / "workspaces" / {
        "vanilla": "opencompass_llada_vanilla",
        "sparse": "opencompass_sparse_dllm",
        "saps": "opencompass_saps",
    }[args.baseline]
    workspace = Path(args.workspace).resolve() if args.workspace else default_workspace.resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace does not exist: {workspace}")
    if args.gsm8k_smoke:
        prompts = load_gsm8k_questions(args.prompt_limit)
    else:
        prompts = load_prompts(args.prompt, args.prompt_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    if args.baseline == "vanilla":
        generate = load_vanilla_runtime(workspace)
        model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        schedule_cfg = None
    else:
        generate, llada_model_cls = load_sparse_runtime(workspace)
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config.kernel_size = args.kernel_size
        config.keep_ratio = args.keep_ratio
        config.block_len = args.block_length
        model = llada_model_cls.from_pretrained(
            args.model_path,
            config=config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        schedule_cfg = None

    runs: list[dict] = []
    for prompt in prompts:
        rendered = prompt
        if args.apply_chat_template:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        encoded = tokenizer(rendered, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        ratio_controller = None
        if args.baseline in {"sparse", "saps"}:
            ratio_controller, schedule_cfg = build_ratio_controller(
                args.baseline,
                keep_ratio=args.keep_ratio,
                r_max=args.r_max,
                r_min=args.r_min,
                decay_type=args.decay_type,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        started = time.time()
        with torch.no_grad():
            if args.baseline == "vanilla":
                _ = generate(
                    model,
                    input_ids,
                    attention_mask=attention_mask,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=126336,
                )
            elif args.baseline == "sparse":
                _ = generate(
                    model,
                    input_ids,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=126336,
                    ratio_controller=ratio_controller,
                )
            else:
                _ = generate(
                    model,
                    input_ids,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=126336,
                    ratio_controller=ratio_controller,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_total_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak_total_gib = None

        profile_dict = ratio_controller.profiler.to_dict() if ratio_controller is not None else None
        kv_summary = (profile_dict or {}).get("summary", {}).get("kv_cache_memory", {})
        avg_kv_cache_gib = kv_summary.get("avg_total_kv_cache_gib")

        runs.append(
            {
                "prompt": prompt,
                "elapsed_seconds": time.time() - started,
                "peak_total_gib": peak_total_gib,
                "peak_memory_gb": peak_total_gib,
                "avg_kv_cache_gib": avg_kv_cache_gib,
                "schedule_config": schedule_cfg,
                "profile": profile_dict,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "workspace": str(workspace),
                    "model_path": args.model_path,
                    "baseline": args.baseline,
                    "steps": args.steps,
                    "gen_length": args.gen_length,
                    "block_length": args.block_length,
                    "kernel_size": args.kernel_size,
                    "keep_ratio": args.keep_ratio,
                    "r_max": args.r_max,
                    "r_min": args.r_min,
                    "decay_type": args.decay_type,
                    "apply_chat_template": args.apply_chat_template,
                    "gsm8k_smoke": args.gsm8k_smoke,
                    "prompt_limit": args.prompt_limit,
                },
                "runs": runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote profiling report to {output_path}")


if __name__ == "__main__":
    main()
