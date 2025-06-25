import os

LAUNCH_CMD = """
python simple_grpo/beaker/launch.py \
    --task_name="{exp_name}" \
    --workspace="ai2/davidh" \
    --cluster="[ai2/jupiter-cirrascale-2]" \
    --budget="ai2/oe-eval" \
    -- \
python simple_grpo/simple_trainer.py \
    --model.model_name_or_path="Qwen/Qwen3-0.6B" \
    --model.checkpoint_save_dir="{checkpoint_save_dir}" \
    --wandb.exp_name="{exp_name}" \
    --wandb.run_name="{run_name}" \
    --train.rollouts_per_prompt="{rollouts_per_prompt}" \
    --train.temperature="{temperature}" \
    --train.lr="{lr}" \
    --train.lr_scheduler_type="{lr_scheduler_type}" \
"""

checkpoint_save_dir = "/oe-eval-default/ai2-llm/checkpoints/davidh/simple-grpo/"
exp_name="qwen3-0.6b"
run_name="qwen3-0.6b"

for rollouts_per_prompt in [2, 4, 8, 16, 32, 64, 128]:
    cmd = LAUNCH_CMD.format(
        checkpoint_save_dir=checkpoint_save_dir + f'rollouts={rollouts_per_prompt}/',
        exp_name=exp_name + f'-rollouts={rollouts_per_prompt}',
        run_name=run_name + f'-rollouts={rollouts_per_prompt}',
        rollouts_per_prompt=rollouts_per_prompt,
        temperature=0.7,
        lr=5e-6,
        lr_scheduler_type="constant",
    )
    os.system(cmd)

for temp in [0, 0.2, 0.7, 1, 1.5, 2]:
    cmd = LAUNCH_CMD.format(
        checkpoint_save_dir=checkpoint_save_dir + f'temp={temp}/',
        exp_name=exp_name + f'-temp={temp}',
        run_name=run_name + f'-temp={temp}',
        rollouts_per_prompt=rollouts_per_prompt,
        temperature=temp,
        lr=5e-6,
        lr_scheduler_type="constant",
    )
    os.system(cmd)

for lr in [1e-5, 5e-5, 1e-6, 5e-6, 1e-7, 5e-7, 1e-8]:
    cmd = LAUNCH_CMD.format(
        checkpoint_save_dir=checkpoint_save_dir + f'lr={lr}/',
        exp_name=exp_name + f'lr={lr}',
        run_name=run_name + f'lr={lr}',
        rollouts_per_prompt=rollouts_per_prompt,
        temperature=0.7,
        lr=lr,
        lr_scheduler_type="constant",
    )
    os.system(cmd)
