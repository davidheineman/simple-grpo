A simple implementation of GRPO. See the core training loop in [`simple_trainer.py`](./simple_grpo/simple_trainer.py#L597-L667).

### Setup

```sh
pip install -e '.[all]'
```

### Quick Start

```sh
# Run the trainer
python simple_grpo/simple_trainer.py

# Evaluate the resulting model
python simple_grpo/simple_eval.py
```

### Launching on Beaker

```sh
# The code after "--" will launch on remote
python simple_grpo/beaker/launch.py \
    --workspace="ai2/davidh" \
    --cluster="[ai2/jupiter-cirrascale-2]" \
    --budget="ai2/oe-eval" \
    --follow=true \
    -- \
python simple_grpo/simple_trainer.py \
    --model.model_name_or_path="Qwen/Qwen3-0.6B" \
    --model.checkpoint_save_dir="/oe-eval-default/ai2-llm/checkpoints/davidh/simple-grpo/" \
    --wandb.exp_name="in-loop-run" \
    --wandb.run_name="in-loop-run"
```

### Local install

```sh
# install nano vllm
git clone https://github.com/davidheineman/nano-vllm
pip install setuptools --link-mode=copy
pip install torch
pip install -e "nano-vllm/." --link-mode=copy --no-build-isolation

# install deepspeed
sudo apt install libmpich-dev # for deepspeed
pip install -e '.[all]'
```
