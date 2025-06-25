### Setup

```sh
pip install -e '.[all]'
```

### Quick Start

```sh
python simple_grpo/simple_trainer.py

python simple_grpo/simple_eval.py
```

### Launching on Beaker

```sh
# (current working command)
python simple_grpo/beaker/launch.py -- python simple_grpo/simple_trainer.py

# Add the prefix before " -- " to launch on remote
python simple_grpo/beaker/launch.py \
    --workspace="ai2/davidh" \
    --cluster="[ai2/jupiter-cirrascale-2]" \
    --budget="ai2/oe-eval" \
    --follow=true \
    -- \
python simple_grpo/simple_trainer.py \
    --model.model_name_or_path="Qwen/Qwen3-0.6B" \
    --model.checkpoint_save_dir="/oe-eval-default/ai2-llm/checkpoints/davidh/simple-grpo/" \
    --wandb.exp_name="first-run" \
    --wandb.run_name="first-run"
```

### More Info

```sh
How long would a training run take on Qwen 3 0.6B?

With 8xA100, assuming 6-7K TPS: 
    -- 1 16-rollout instance per sec (or 1 forward pass every 0.625 sec) 
    -- Minerva pass@1 would take 31.25 sec
    -- 1 epoch on HamishMATH would take 38 hours (56K x 16 samples)

Minimum TODOs:
    [X] Correctness -- Token advantages correct? (yes, they are)
    [X] Beaker launcher (gantry, not mason) -- https://github.com/allenai/beaker-gantry
    [X] Basic wandb support
    [X] Saving / loading (pre-emptible job)
    [ ] In-loop Minerva
    [ ] Fix stop sequences
    [ ] Multi-GPU support (with Ray)

Nice to haves:
    [ ] Pull the Gantry install and default container and push a basic container w/ the install command
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