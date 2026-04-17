from __future__ import annotations

import argparse
import json
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


def write_manifest(cfg: dict, vanilla: dict, sparse: dict) -> Path:
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
        ],
    }
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
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    vanilla = prepare_vanilla_workspace(cfg)
    sparse = prepare_sparse_workspace(cfg)
    manifest_path = write_manifest(cfg, vanilla, sparse)

    print("Prepared first baseline workspaces:")
    print(f"  vanilla: {vanilla['workspace']}")
    print(f"  sparse:  {sparse['workspace']}")
    print(f"  manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
