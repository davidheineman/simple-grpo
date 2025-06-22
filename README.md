GRPO training in less than 500 lines of Python

```sh
# install nano vllm
git clone https://github.com/davidheineman/nano-vllm
pip install -e "nano-vllm/."

# install math grader
pip install sympy

# install deepspeed
sudo apt install libmpich-dev
pip install 'deepspeed[cuda118]'
pip install mpi4py

python simple_grpo.py
```