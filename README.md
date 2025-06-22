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
```

### Quick Start

```sh
python src/simple_grpo.py

python src/simple_eval.py
```