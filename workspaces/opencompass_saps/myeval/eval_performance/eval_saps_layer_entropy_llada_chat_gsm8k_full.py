from mmengine.config import read_base
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

num_gpus = {'llada_8b_chat': 1}
path_dict = {'llada_8b_chat': 'GSAI-ML/LLaDA-8B-Instruct'}

# Layer-aware SAPS: entropy-guided dynamic — budget ∝ per-layer attention entropy (one-step lag)
saps_config = {
    'r_max': 0.7,
    'r_min': 0.1,
    'decay_type': 'exp',
    'step_granularity': 'global',
    'layer_mode': 'entropy',
}

models = [
    ('llada_8b_chat-saps-layer-entropy', {}, {'steps': 256, 'block_length': 32}, 3, saps_config),
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

work_dir = './outputs/saps_layer_entropy/'

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=8, task=dict(type=OpenICLEvalTask, dump_details=True)),
)
