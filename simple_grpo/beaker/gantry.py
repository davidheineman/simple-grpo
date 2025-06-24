"""
Gantry is essentially a wrapper around this python command

pip install beaker-gantry

gantry run --timeout -1 -- python -c 'print("Hello, World!")'

Notes on gantry:
    - Pro: Gantry has support
    - Con: Gantry installs deps on every job

I think we should do *both*. Mini mason has most of the defaults we need, gantry allows pulling the github container. We just need to write a connector to this function.

Finally, to make things faster, we could write a simple Dockerfile which pre-installs the dependencies and pushes through git actions (although not necessary I think? saves like 2 min of startup time). Then, at startup gantry will just pull the current codebase.


Next: How to do code execution? 
    - Sets of Dockerfiles should be a must -- This is standard in related work.
    - Could do (1) pre-built dockerfiles (2) launch execution jobs on beaker (so every tool call spins up a container). Could actually be quick to run maybe... (start with this for now. other options are AWS lambdas, or building your own load balancer)
"""


def launch_experiment(
    args: Sequence[str],
    name: Optional[str] = None,
    description: Optional[str] = None,
    task_name: str = "main",
    workspace: Optional[str] = None,
    group_name: Optional[str] = None,
    clusters: Optional[Sequence[str]] = None,
    gpu_types: Optional[Sequence[str]] = None,
    hostnames: Optional[Sequence[str]] = None,
    beaker_image: Optional[str] = None,
    docker_image: Optional[str] = None,
    cpus: Optional[float] = None,
    gpus: Optional[int] = None,
    memory: Optional[str] = None,
    shared_memory: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
    gh_token_secret: str = constants.GITHUB_TOKEN_SECRET,
    ref: Optional[str] = None,
    branch: Optional[str] = None,
    conda: Optional[PathOrStr] = None,
    pip: Optional[PathOrStr] = None,
    venv: Optional[str] = None,
    env_vars: Optional[Sequence[str]] = None,
    env_secrets: Optional[Sequence[str]] = None,
    dataset_secrets: Optional[Sequence[str]] = None,
    timeout: int = 0,
    task_timeout: Optional[str] = None,
    show_logs: bool = True,
    allow_dirty: bool = False,
    dry_run: bool = False,
    yes: bool = False,
    save_spec: Optional[PathOrStr] = None,
    priority: Optional[str] = None,
    install: Optional[str] = None,
    no_python: bool = False,
    no_conda: bool = False,
    replicas: Optional[int] = None,
    leader_selection: bool = False,
    host_networking: bool = False,
    propagate_failure: Optional[bool] = None,
    propagate_preemption: Optional[bool] = None,
    synchronized_start_timeout: Optional[str] = None,
    mounts: Optional[Sequence[str]] = None,
    weka: Optional[str] = None,
    budget: Optional[str] = None,
    preemptible: Optional[bool] = None,
    retries: Optional[int] = None,
    results: str = constants.RESULTS_DIR,
    skip_tcpxo_setup: bool = False,
):