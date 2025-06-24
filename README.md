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
python simple_grpo/beaker/launch.py \
    --workspace="ai2/davidh" \
    --cluster="ai2/jupiter-cirrascale-2" \
    --budget="oe-eval"

# --model_config.model_name_or_path="Qwen/Qwen3-0.6B"

python simple_grpo/beaker/launch.py \
    -- python simple_grpo/simple_trainer.py
```

### More Info

```sh
How long would a training run take on Qwen 3 0.6B?

With 8xA100, assuming 6-7K TPS: 
    -- 1 16-rollout instance per sec (or 1 forward pass every 0.625 sec) 
    -- Minerva pass@1 would take 31.25 sec
    -- 1 epoch on HamishMATH would take 38 hours (56K x 16 samples)

Minimum TODOs:
    [ ] Correctness -- Token advantages correct?
    [ ] Beaker launcher (gantry, not mason) -- https://github.com/allenai/beaker-gantry
    [ ] Saving / loading (pre-emptible job)
    [ ] In-loop Minerva
    [ ] Wandb support
    [ ] Fix stop sequences
    [ ] Multi-GPU support (with Ray)
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