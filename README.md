### Setup

```sh
# install nano vllm
git clone https://github.com/davidheineman/nano-vllm
pip install setuptools --link-mode=copy
pip install torch
pip install -e "nano-vllm/." --link-mode=copy --no-build-isolation

# install math extraction dependency
pip install sympy antlr4-python3-runtime==4.11

# install datasets
pip install datasets

# install deepspeed
sudo apt install libmpich-dev
pip install deepspeed
pip install mpi4py

# install misc. deps
pip install accelerate
```

### Quick Start

```sh
python src/simple_grpo.py

python src/simple_eval.py
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
    [ ] Beaker launcher -- https://github.com/allenai/open-instruct/blob/main/mason.py
    [ ] Saving / loading (pre-emptible job)
    [ ] In-loop Minerva
    [ ] Wandb support
    [ ] Fix stop sequences
    [ ] Multi-GPU support (with Ray)
```