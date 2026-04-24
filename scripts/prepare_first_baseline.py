from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SPARSE_GSM8K_CONFIG = """from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import Sparse_dLLM_LLaDACausalLM

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc_maxoutlen_256 import gsm8k_datasets

datasets = []
datasets += gsm8k_datasets
max_seq_len = 2048
max_out_len = 256

num_gpus = {
    'llada_8b_chat': 1,
}

path_dict = {
    'llada_8b_chat': '__MODEL_PATH__',
}

models = [
    ('llada_8b_chat-sparse_dllm', {}, {'steps': 256, 'block_length': 32}, 3, 0.5),
]

models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]],
        kernel_size=kernel_size, keep_ratio=keep_ratio,
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        max_seq_len=max_seq_len, max_out_len=max_out_len, batch_size=1,
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, kernel_size, keep_ratio in models
]

work_dir = './outputs/sparse_dllm/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)
"""

VANILLA_GSM8K_SMOKE_CONFIG = """from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.models.dllm.llada_instruct_8b import models as llada_instruct_8b_models

datasets = gsm8k_datasets
for dataset in datasets:
    dataset['reader_cfg'] = dict(dataset['reader_cfg'])
    dataset['reader_cfg']['test_range'] = '[0:4]'

models = llada_instruct_8b_models
eval_cfg = {
    'gen_blocksize': 512,
    'gen_length': 512,
    'gen_steps': 512,
    'batch_size': 1,
    'batch_size_': 1,
    'diff_confidence_eos_eot_inf': True,
    'diff_logits_eos_inf': False,
}
for model in models:
    model.update(eval_cfg)

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=4,
        num_split=None,
        min_task_size=1,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask),
        retry=1,
    ),
)
"""

VANILLA_GSM8K_DEV_CONFIG = """from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.models.dllm.llada_instruct_8b import models as llada_instruct_8b_models

datasets = gsm8k_datasets
for dataset in datasets:
    dataset['reader_cfg'] = dict(dataset['reader_cfg'])
    dataset['reader_cfg']['test_range'] = '[0:128]'

models = llada_instruct_8b_models
eval_cfg = {
    'gen_blocksize': 512,
    'gen_length': 512,
    'gen_steps': 512,
    'batch_size': 1,
    'batch_size_': 1,
    'diff_confidence_eos_eot_inf': True,
    'diff_logits_eos_inf': False,
}
for model in models:
    model.update(eval_cfg)

from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=8,
        num_split=None,
        min_task_size=1,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask),
        retry=1,
    ),
)
"""

SPARSE_GSM8K_SMOKE_CONFIG = """from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import Sparse_dLLM_LLaDACausalLM

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc_maxoutlen_256 import gsm8k_datasets

datasets = []
datasets += gsm8k_datasets
for dataset in datasets:
    dataset['reader_cfg'] = dict(dataset['reader_cfg'])
    dataset['reader_cfg']['test_range'] = '[0:4]'
max_seq_len = 2048
max_out_len = 256

num_gpus = {
    'llada_8b_chat': 1,
}

path_dict = {
    'llada_8b_chat': '__MODEL_PATH__',
}

models = [
    ('llada_8b_chat-sparse_dllm', {}, {'steps': 256, 'block_length': 32}, 3, 0.5),
]

models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]],
        kernel_size=kernel_size, keep_ratio=keep_ratio,
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        max_seq_len=max_seq_len, max_out_len=max_out_len, batch_size=1,
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, kernel_size, keep_ratio in models
]

work_dir = './outputs/sparse_dllm/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)
"""

SPARSE_GSM8K_DEV_CONFIG = """from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import Sparse_dLLM_LLaDACausalLM

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc_maxoutlen_256 import gsm8k_datasets

datasets = []
datasets += gsm8k_datasets
for dataset in datasets:
    dataset['reader_cfg'] = dict(dataset['reader_cfg'])
    dataset['reader_cfg']['test_range'] = '[0:128]'
max_seq_len = 2048
max_out_len = 256

num_gpus = {
    'llada_8b_chat': 1,
}

path_dict = {
    'llada_8b_chat': '__MODEL_PATH__',
}

models = [
    ('llada_8b_chat-sparse_dllm', {}, {'steps': 256, 'block_length': 32}, 3, 0.5),
]

models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]],
        kernel_size=kernel_size, keep_ratio=keep_ratio,
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        max_seq_len=max_seq_len, max_out_len=max_out_len, batch_size=1,
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, kernel_size, keep_ratio in models
]

work_dir = './outputs/sparse_dllm/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)
"""

SPARSE_IMPORTS = [
    "from .sparse_dllm.llada_wrapper import Sparse_dLLM_LLaDACausalLM",
    "from .sparse_dllm.dream_wrapper import Sparse_dLLM_DreamCausalLM",
    "from .sparse_dllm.dream_wrapper_instruct import Sparse_dLLM_DreamCausalLMInstruct",
]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reset_dir(target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)


def copy_tree(src: Path, dst: Path) -> None:
    reset_dir(dst)
    shutil.copytree(src, dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def patch_llada_model_path(opencompass_root: Path, model_path: str) -> Path:
    target = opencompass_root / "opencompass" / "configs" / "models" / "dllm" / "llada_instruct_8b.py"
    text = target.read_text(encoding="utf-8")
    marker = "path='/mnt/oujingyang/assets/model/LLaDA'"
    if marker not in text:
        raise RuntimeError(f"Unexpected llada_instruct_8b.py contents in {target}")
    updated = text.replace(marker, f"path={model_path!r}")
    target.write_text(updated, encoding="utf-8", newline="\n")
    return target


def patch_dllm_workspace_root(workspace: Path) -> Path:
    target = workspace / "opencompass" / "models" / "dllm.py"
    text = target.read_text(encoding="utf-8")
    marker = "llada_root = Path(__file__).resolve().parents[3]"
    if marker not in text:
        raise RuntimeError(f"Unexpected dllm.py contents in {target}")
    updated = text.replace(marker, "llada_root = Path(__file__).resolve().parents[2]")
    target.write_text(updated, encoding="utf-8", newline="\n")
    return target


def stage_llada_generate_helper(llada_repo: Path, workspace: Path) -> Path:
    src = llada_repo / "generate.py"
    dst = workspace / "generate.py"
    shutil.copy2(src, dst)
    return dst


def append_sparse_imports(opencompass_root: Path) -> Path:
    init_path = opencompass_root / "opencompass" / "models" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    missing = [line for line in SPARSE_IMPORTS if line not in text]
    if missing:
        updated = text.rstrip() + "\n" + "\n".join(missing) + "\n"
        init_path.write_text(updated, encoding="utf-8", newline="\n")
    return init_path


def overlay_sparse_files(sparse_repo: Path, sparse_workspace: Path) -> None:
    sparse_opencompass = sparse_repo / "opencompass"
    for relative in [
        Path("configs"),
        Path("datasets"),
        Path("models") / "sparse_dllm",
    ]:
        src = sparse_opencompass / relative
        dst = sparse_workspace / "opencompass" / relative
        if src.is_dir():
            for item in src.rglob("*"):
                destination = dst / item.relative_to(src)
                if item.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, destination)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    myeval_src = sparse_repo / "myeval"
    myeval_dst = sparse_workspace / "myeval"
    if myeval_dst.exists():
        shutil.rmtree(myeval_dst)
    shutil.copytree(myeval_src, myeval_dst)


def build_sparse_gsm8k_config(sparse_workspace: Path, model_path: str) -> Path:
    target = sparse_workspace / "myeval" / "eval_performance" / "eval_sparse_dllm_llada_chat_gsm8k_local.py"
    write_text(target, SPARSE_GSM8K_CONFIG.replace("__MODEL_PATH__", model_path.replace("\\", "\\\\")))
    return target


def build_vanilla_gsm8k_smoke_config(vanilla_workspace: Path) -> Path:
    target = vanilla_workspace / "examples" / "llada_instruct_gen_gsm8k_length512_block512_confidence_smoke.py"
    write_text(target, VANILLA_GSM8K_SMOKE_CONFIG)
    return target


def build_vanilla_gsm8k_dev_config(vanilla_workspace: Path) -> Path:
    target = vanilla_workspace / "examples" / "llada_instruct_gen_gsm8k_length512_block512_confidence_dev.py"
    write_text(target, VANILLA_GSM8K_DEV_CONFIG)
    return target


def build_sparse_gsm8k_smoke_config(sparse_workspace: Path, model_path: str) -> Path:
    target = sparse_workspace / "myeval" / "eval_performance" / "eval_sparse_dllm_llada_chat_gsm8k_smoke.py"
    write_text(target, SPARSE_GSM8K_SMOKE_CONFIG.replace("__MODEL_PATH__", model_path.replace("\\", "\\\\")))
    return target


def build_sparse_gsm8k_dev_config(sparse_workspace: Path, model_path: str) -> Path:
    target = sparse_workspace / "myeval" / "eval_performance" / "eval_sparse_dllm_llada_chat_gsm8k_dev.py"
    write_text(target, SPARSE_GSM8K_DEV_CONFIG.replace("__MODEL_PATH__", model_path.replace("\\", "\\\\")))
    return target


def prepare_vanilla_workspace(cfg: dict) -> dict:
    llada_repo = ROOT / cfg["paths"]["llada_repo"]
    workspace_root = ROOT / cfg["paths"]["workspace_root"]
    workspace = workspace_root / cfg["runs"]["vanilla"]["workspace"]
    copy_tree(llada_repo / "opencompass", workspace)
    patched = patch_llada_model_path(workspace, cfg["model_path"])
    patched_dllm = patch_dllm_workspace_root(workspace)
    generate_helper = stage_llada_generate_helper(llada_repo, workspace)
    dev_cfg = build_vanilla_gsm8k_dev_config(workspace)
    smoke_cfg = build_vanilla_gsm8k_smoke_config(workspace)
    return {
        "workspace": str(workspace.resolve()),
        "patched_model_config": str(patched.resolve()),
        "patched_dllm_wrapper": str(patched_dllm.resolve()),
        "generate_helper": str(generate_helper.resolve()),
        "dev_config": str(dev_cfg.resolve()),
        "smoke_config": str(smoke_cfg.resolve()),
    }


def prepare_sparse_workspace(cfg: dict) -> dict:
    llada_repo = ROOT / cfg["paths"]["llada_repo"]
    sparse_repo = ROOT / cfg["paths"]["sparse_repo"]
    workspace_root = ROOT / cfg["paths"]["workspace_root"]
    workspace = workspace_root / cfg["runs"]["sparse"]["workspace"]
    copy_tree(llada_repo / "opencompass", workspace)
    overlay_sparse_files(sparse_repo, workspace)
    patched_dllm = patch_dllm_workspace_root(workspace)
    generate_helper = stage_llada_generate_helper(llada_repo, workspace)
    imports = append_sparse_imports(workspace)
    patch_modeling_llada(workspace)
    patch_llada_generate(workspace)
    local_cfg = build_sparse_gsm8k_config(workspace, cfg["model_path"])
    dev_cfg = build_sparse_gsm8k_dev_config(workspace, cfg["model_path"])
    smoke_cfg = build_sparse_gsm8k_smoke_config(workspace, cfg["model_path"])
    return {
        "workspace": str(workspace.resolve()),
        "patched_dllm_wrapper": str(patched_dllm.resolve()),
        "generate_helper": str(generate_helper.resolve()),
        "patched_model_imports": str(imports.resolve()),
        "local_sparse_config": str(local_cfg.resolve()),
        "dev_config": str(dev_cfg.resolve()),
        "smoke_config": str(smoke_cfg.resolve()),
    }


SAPS_GSM8K_CONFIG_TEMPLATE = """from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import Sparse_dLLM_LLaDACausalLM

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_17d0dc_maxoutlen_256 import gsm8k_datasets

datasets = []
datasets += gsm8k_datasets
__DATASET_RANGE_BLOCK__
max_seq_len = 2048
max_out_len = 256

num_gpus = {
    'llada_8b_chat': 1,
}

path_dict = {
    'llada_8b_chat': '__MODEL_PATH__',
}

# SAPS parameters: replace r_max, r_min, decay_type as needed
saps_config = {
    'r_max': __R_MAX__,
    'r_min': __R_MIN__,
    'decay_type': '__DECAY_TYPE__',
    'step_granularity': 'global',
}

models = [
    ('llada_8b_chat-saps', {}, {'steps': 256, 'block_length': 32}, 3, saps_config),
]

models = [
    dict(
        type=Sparse_dLLM_LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]],
        kernel_size=kernel_size, saps_config=saps_config,
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        max_seq_len=max_seq_len, max_out_len=max_out_len, batch_size=1,
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, kernel_size, saps_config in models
]

work_dir = './outputs/saps/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)
"""


def patch_modeling_llada(workspace: Path) -> Path:
    """Patch modeling_llada.py to accept and use ratio_controller.
    
    Changes:
    1. Add ratio_controller parameter to CustomCache.__init__()
    2. Store it as instance variable
    3. Use it in filter_cache() instead of fixed keep_ratio
    4. Emit KV-cache memory bytes to profiler
    """
    file_path = workspace / "opencompass" / "models" / "sparse_dllm" / "modeling_llada.py"
    
    if not file_path.exists():
        raise FileNotFoundError(f"modeling_llada.py not found at {file_path}")
    
    content = file_path.read_text(encoding="utf-8")
    original_content = content
    
    # Patch 1: Add ratio_controller parameter to __init__
    init_pattern = r"(def __init__\(self, n_layers: int, device: torch\.device,\s+kernel_size: Optional\[int\] = None, keep_ratio: float = 0\.7)\)"
    init_replacement = r"\1, ratio_controller=None)"
    content = re.sub(init_pattern, init_replacement, content)
    
    # Patch 2: Store ratio_controller as instance variable
    store_pattern = r"(self\.keep_ratios = \[keep_ratio for i in range\(n_layers\)\])"
    store_replacement = r"\1\n        self.ratio_controller = ratio_controller\n        self.profiler = getattr(ratio_controller, 'profiler', None)"
    content = re.sub(store_pattern, store_replacement, content)
    
    # Patch 3: Use ratio_controller in filter_cache()
    filter_pattern = r"keep_num = int\(importance\.size\(-1\) \* self\.keep_ratios\[layer_id\]\)"
    filter_replacement = """if self.ratio_controller is not None:
            keep_num = self.ratio_controller.keep_num(importance.size(-1))
        else:
            keep_num = int(importance.size(-1) * self.keep_ratios[layer_id])"""
    content = re.sub(filter_pattern, filter_replacement, content)

    profiler_old = "        _, keep_indices = torch.topk(importance, k=keep_num, dim=-1)\n        keep_indices = keep_indices.squeeze(0)"
    profiler_new = """        _, keep_indices = torch.topk(importance, k=keep_num, dim=-1)
        keep_indices = keep_indices.squeeze(0)
        if self.profiler is not None and self.ratio_controller is not None:
            self.profiler.on_cache_selection(
                step=self.ratio_controller.step,
                total_steps=self.ratio_controller.total_steps,
                layer_id=layer_id,
                keep_num=keep_num,
                candidate_count=importance.size(-1),
                keep_indices=keep_indices.detach().cpu().tolist(),
                importance_mean=float(importance.mean().item()),
                importance_max=float(importance.max().item()),
            )"""
    if profiler_new not in content:
        content = content.replace(profiler_old, profiler_new)

    kv_memory_old = """        self.cache[layer_id] = {
            "k": filtered_cached_k,
            "v": filtered_cached_v
        }"""
    kv_memory_new = """        self.cache[layer_id] = {
            "k": filtered_cached_k,
            "v": filtered_cached_v
        }
        if self.profiler is not None and self.ratio_controller is not None:
            total_kv_cache_bytes = sum(
                int(layer_cache["k"].numel() * layer_cache["k"].element_size() + layer_cache["v"].numel() * layer_cache["v"].element_size())
                for layer_cache in self.cache.values()
                if layer_cache["k"] is not None and layer_cache["v"] is not None
            )
            layer_kv_cache_bytes = int(
                filtered_cached_k.numel() * filtered_cached_k.element_size()
                + filtered_cached_v.numel() * filtered_cached_v.element_size()
            )
            self.profiler.on_kv_cache_memory(
                step=self.ratio_controller.step,
                total_steps=self.ratio_controller.total_steps,
                layer_id=layer_id,
                layer_kv_cache_bytes=layer_kv_cache_bytes,
                total_kv_cache_bytes=total_kv_cache_bytes,
            )"""
    if "on_kv_cache_memory(" not in content:
        content = content.replace(kv_memory_old, kv_memory_new)
    
    if content != original_content:
        file_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Patched {file_path.name}")
    
    return file_path


def patch_llada_generate(workspace: Path) -> Path:
    """Patch llada_generate.py to compute global steps and call ratio_controller.set_step().
    
    Changes:
    1. Add ratio_controller parameter to generate()
    2. Compute global_t = num_block * steps + i (steps is already divided by num_blocks)
    3. Call ratio_controller.set_step(global_t, total_steps) before model()
    4. Pass ratio_controller to CustomCache init
    """
    file_path = workspace / "opencompass" / "models" / "sparse_dllm" / "llada_generate.py"
    
    if not file_path.exists():
        raise FileNotFoundError(f"llada_generate.py not found at {file_path}")
    
    content = file_path.read_text(encoding="utf-8")
    original_content = content
    
    # Patch 1: Add ratio_controller parameter to generate() - if not already present
    if "ratio_controller=None" not in content:
        gen_pattern = r"(def generate\(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0\.,\s+cfg_scale=0\., remasking='low_confidence', mask_id=126336)\)"
        gen_replacement = r"\1, ratio_controller=None)"
        content = re.sub(gen_pattern, gen_replacement, content)
    
    # Patch 2: Pass ratio_controller to CustomCache - use exact line replacement
    cache_old = "                        kernel_size=model.config.kernel_size, keep_ratio=model.config.keep_ratio)"
    cache_new = "                        kernel_size=model.config.kernel_size, keep_ratio=model.config.keep_ratio, ratio_controller=ratio_controller)"
    if cache_new not in content:
        content = content.replace(cache_old, cache_new)
    
    # Patch 3: Add global step computation and set_step() call in the loop
    # The loop is "for i in range(steps):" where steps has been divided by num_blocks
    if "global_t = num_block * steps + i" not in content:
        loop_pattern = r"(for i in range\(steps\):)\n(\s+# Determine cache state)"
        loop_replacement = r"\1\n            # SAPS: Compute global step across all blocks\n            global_t = num_block * steps + i\n            total_steps = num_blocks * steps\n            \n            # SAPS: Update ratio controller for step-aware pruning\n            if ratio_controller is not None:\n                ratio_controller.set_step(global_t, total_steps)\n            \n\2"
        content = re.sub(loop_pattern, loop_replacement, content)
    
    if content != original_content:
        file_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Patched {file_path.name}")
    
    return file_path


def patch_llada_wrapper(workspace: Path) -> Path:
    """Patch llada_wrapper.py to instantiate and use RatioController.
    
    Changes:
    1. Extract saps_config from **other_kwargs
    2. Store it as instance variable
    3. Instantiate RatioController before generate() call
    4. Pass ratio_controller to generate()
    """
    file_path = workspace / "opencompass" / "models" / "sparse_dllm" / "llada_wrapper.py"
    
    if not file_path.exists():
        raise FileNotFoundError(f"llada_wrapper.py not found at {file_path}")
    
    content = file_path.read_text(encoding="utf-8")
    original_content = content
    
    # Patch 1: Extract saps_config from **other_kwargs (after keep_ratio assignment)
    if "self.saps_config" not in content:
        store_pattern = r"(self\.keep_ratio = keep_ratio)"
        store_replacement = r"\1\n        self.saps_config = other_kwargs.get('saps_config', None)"
        content = re.sub(store_pattern, store_replacement, content)
    
    # Patch 2: Add RatioController instantiation before generate() with sys.path handling
    if "RatioController" not in content:
        generate_pattern = r"(outputs = generate\(self\.model, tokens\['input_ids'\],)"
        generate_replacement = r"""# SAPS: Initialize ratio controller if saps_config provided
            ratio_controller = None
            if self.saps_config is not None:
                import sys
                from pathlib import Path
                # Add workspace root to sys.path to find saps module
                workspace_root = Path(__file__).resolve().parents[3]
                if str(workspace_root) not in sys.path:
                    sys.path.insert(0, str(workspace_root))
                from saps.ratio_controller import RatioController
                from saps.schedule import SAPSScheduleConfig
                saps_cfg = SAPSScheduleConfig(**self.saps_config)
                ratio_controller = RatioController(saps_cfg)
            
            \1
                               ratio_controller=ratio_controller,"""
        content = re.sub(generate_pattern, generate_replacement, content)
    
    if content != original_content:
        file_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Patched {file_path.name}")
    
    return file_path


def apply_saps_patches(workspace: Path) -> dict:
    """Apply all SAPS patches to the workspace automatically.
    
    Returns dict with paths to all patched files.
    """
    print("🔧 Applying SAPS patches to workspace...")
    patched_files = {
        "modeling_llada": patch_modeling_llada(workspace),
        "llada_generate": patch_llada_generate(workspace),
        "llada_wrapper": patch_llada_wrapper(workspace),
    }
    print("✅ All SAPS patches applied automatically!")
    return patched_files


def build_saps_gsm8k_config(
    saps_workspace: Path,
    filename: str,
    model_path: str,
    r_max: float = 0.8,
    r_min: float = 0.1,
    decay_type: str = "exp",
    test_range: str | None = None,
) -> Path:
    """Build a SAPS GSM8K OpenCompass config with schedule parameters."""
    target = saps_workspace / "myeval" / "eval_performance" / filename
    config_text = SAPS_GSM8K_CONFIG_TEMPLATE.replace("__MODEL_PATH__", model_path.replace("\\", "\\\\"))
    config_text = config_text.replace("__R_MAX__", str(r_max))
    config_text = config_text.replace("__R_MIN__", str(r_min))
    config_text = config_text.replace("__DECAY_TYPE__", decay_type)
    if test_range is None:
        dataset_range_block = ""
    else:
        dataset_range_block = (
            "for dataset in datasets:\n"
            "    dataset['reader_cfg'] = dict(dataset['reader_cfg'])\n"
            f"    dataset['reader_cfg']['test_range'] = '{test_range}'"
        )
    config_text = config_text.replace("__DATASET_RANGE_BLOCK__", dataset_range_block)
    write_text(target, config_text)
    return target


def prepare_saps_workspace(cfg: dict, r_max: float = 0.8, r_min: float = 0.1, 
                           decay_type: str = "exp", apply_patches: bool = True) -> dict:
    """Prepare SAPS workspace with step-aware pruning schedule.
    
    Mirrors the sparse workspace and automatically applies SAPS patches to enable
    step-aware pruning in the generate loop.
    
    Args:
        cfg: Configuration dict with paths
        r_max: Maximum retention ratio (early denoising)
        r_min: Minimum retention ratio (late denoising)
        decay_type: Schedule type: linear, cosine, exp, or constant
        apply_patches: If True, automatically apply all SAPS patches.
                      If False, user must apply manually (for debugging).
    
    Returns:
        Dict with workspace info and paths to patched files
    """
    llada_repo = ROOT / cfg["paths"]["llada_repo"]
    sparse_repo = ROOT / cfg["paths"]["sparse_repo"]
    workspace_root = ROOT / cfg["paths"]["workspace_root"]
    saps_workspace_path = cfg.get("runs", {}).get("saps", {}).get("workspace", "opencompass_saps")
    workspace = workspace_root / saps_workspace_path
    
    copy_tree(llada_repo / "opencompass", workspace)
    overlay_sparse_files(sparse_repo, workspace)
    patched_dllm = patch_dllm_workspace_root(workspace)
    generate_helper = stage_llada_generate_helper(llada_repo, workspace)
    imports = append_sparse_imports(workspace)
    full_cfg = build_saps_gsm8k_config(
        workspace,
        "eval_saps_llada_chat_gsm8k.py",
        cfg["model_path"],
        r_max,
        r_min,
        decay_type,
    )
    dev_cfg = build_saps_gsm8k_config(
        workspace,
        "eval_saps_llada_chat_gsm8k_dev.py",
        cfg["model_path"],
        r_max,
        r_min,
        decay_type,
        test_range="[0:128]",
    )
    smoke_cfg = build_saps_gsm8k_config(
        workspace,
        "eval_saps_llada_chat_gsm8k_smoke.py",
        cfg["model_path"],
        r_max,
        r_min,
        decay_type,
        test_range="[0:4]",
    )
    
    # Copy SAPS module into workspace for Modal availability
    saps_src = ROOT / "saps"
    saps_dst = workspace / "saps"
    if saps_src.exists() and not saps_dst.exists():
        import shutil as _shutil
        _shutil.copytree(saps_src, saps_dst)
        print(f"  ✓ Copied saps module to workspace")
    
    # Apply SAPS patches to make generate loop step-aware
    patched_files = {}
    if apply_patches:
        patched_files_raw = apply_saps_patches(workspace)
        patched_files = {k: str(v.resolve()) for k, v in patched_files_raw.items()}
    else:
        print("⚠️  Skipping automatic patching. Follow SAPS_INTEGRATION_GUIDE.md for manual steps.")
    
    return {
        "workspace": str(workspace.resolve()),
        "patched_dllm_wrapper": str(patched_dllm.resolve()),
        "generate_helper": str(generate_helper.resolve()),
        "patched_model_imports": str(imports.resolve()),
        "config": str(full_cfg.resolve()),
        "dev_config": str(dev_cfg.resolve()),
        "smoke_config": str(smoke_cfg.resolve()),
        "patched_files": patched_files,  # Track which files were patched
        "saps_config": {
            "r_max": r_max,
            "r_min": r_min,
            "decay_type": decay_type,
        }
    }


def write_manifest(cfg: dict, vanilla: dict, sparse: dict, saps: dict | None = None) -> Path:
    results_root = ROOT / cfg["paths"]["results_root"]
    results_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_path": cfg["model_path"],
        "vanilla": vanilla,
        "sparse": sparse,
        "source_repos": {
            "llada_repo": str((ROOT / cfg["paths"]["llada_repo"]).resolve()),
            "sparse_repo": str((ROOT / cfg["paths"]["sparse_repo"]).resolve()),
        },
        "notes": [
            "Vanilla workspace is copied from external/LLaDA/opencompass and only patches the llada_instruct_8b model path.",
            "Sparse workspace starts from the same OpenCompass tree, overlays external/Sparse-dLLM/opencompass and myeval, appends Sparse-dLLM model imports, and adds a GSM8K-only config derived from eval_sparse_dllm_llada_chat.py.",
            "SAPS workspace adds step-aware pruning schedule support with configurable r_max, r_min, and decay_type.",
        ],
    }
    if saps is not None:
        manifest["saps"] = saps
    manifest_path = results_root / "prepared_workspaces.json"
    write_text(manifest_path, json.dumps(manifest, indent=2))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare isolated OpenCompass workspaces for the first vanilla and Sparse-dLLM baselines.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "first_working_baseline.json"),
        help="Path to the first baseline JSON config.",
    )
    parser.add_argument(
        "--with-saps",
        action="store_true",
        help="Also prepare SAPS (step-aware pruning schedule) workspace.",
    )
    parser.add_argument(
        "--saps-r-max",
        type=float,
        default=0.8,
        help="SAPS r_max parameter (default: 0.8).",
    )
    parser.add_argument(
        "--saps-r-min",
        type=float,
        default=0.1,
        help="SAPS r_min parameter (default: 0.1).",
    )
    parser.add_argument(
        "--saps-decay-type",
        choices=["linear", "cosine", "exp", "constant"],
        default="exp",
        help="SAPS decay_type parameter (default: exp).",
    )
    parser.add_argument(
        "--skip-patches",
        action="store_true",
        help="Skip automatic SAPS patch application (for manual review/debugging).",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    vanilla = prepare_vanilla_workspace(cfg)
    sparse = prepare_sparse_workspace(cfg)
    saps = None
    
    if args.with_saps:
        saps = prepare_saps_workspace(
            cfg, 
            args.saps_r_max, 
            args.saps_r_min, 
            args.saps_decay_type,
            apply_patches=not args.skip_patches
        )
    
    manifest_path = write_manifest(cfg, vanilla, sparse, saps)

    print("Prepared first baseline workspaces:")
    print(f"  vanilla: {vanilla['workspace']}")
    print(f"  sparse:  {sparse['workspace']}")
    if saps:
        print(f"  saps:    {saps['workspace']}")
    print(f"  manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
