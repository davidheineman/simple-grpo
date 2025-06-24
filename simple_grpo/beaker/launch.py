from dataclasses import dataclass, field
import os
import re
import secrets
import string
import sys
from typing import Dict, List, Optional
import beaker as bk
from rich.console import Console
from rich.text import Text
from rich.pretty import pprint
from simple_grpo.beaker.config import make_config
from simple_grpo.beaker.defaults import get_env_vars, get_mounts
from simple_grpo.beaker.constants import INTERCONNECT_CLUSTERS, WEKA_CLUSTERS, GCP_CLUSTERS
import beaker as bk
from simple_grpo.beaker.config import make_config
from simple_grpo.beaker.defaults import get_env_vars, get_mounts
from gantry.api import launch_experiment

console = Console()


@dataclass
class BeakerConfig:
    workspace: str
    cluster: List[str]
    budget: str

    # Run configs
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_tags: Optional[str] = None

    # Optional args
    hostname: Optional[List[str]] = None  # specific nodes to run a job
    max_retries: int = 0
    gpus: int = 0
    num_nodes: int = 1
    image: str = "ai2/cuda11.8-cudnn8-dev-ubuntu20.04"
    description: str = "davidh training job 🔥🫡"
    task_name: str = "davidh_task"
    priority: str = "normal"
    preemptible: bool = True
    pure_docker_mode: bool = True  # If false, will cd into os.getcwd()
    no_auto_dataset_cache: bool = False
    auto_output_dir_path: str = "/oe-eval-default/davidh/deletable_checkpoint"
    auto_checkpoint_state_dir: str = "/oe-eval-default/davidh/deletable_checkpoint_states"
    beaker_datasets: List[Dict[str, str]] = field(
        default_factory=list
    )  # TODO: Add parser from mason.py
    env: List[Dict[str, str]] = field(default_factory=list)  # TODO: Add parser from mason.py
    secret: List[Dict[str, str]] = field(default_factory=list)  # TODO: Add parser from mason.py
    no_host_networking: bool = False


def gen_uuid(length: int = 8) -> str:
    """Random base-36 string of `length` digits."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def make_command(cmd: List[str], config: BeakerConfig) -> str:
    # pass through WANDB_ENTITY and WANDB_PROJECT
    if config.wandb_entity is not None:
        cmd = [f"WANDB_ENTITY={config.wandb_entity}"] + cmd
    if config.wandb_project is not None:
        cmd = [f"WANDB_PROJECT={config.wandb_project}"] + cmd
    if config.wandb_tags is not None:
        cmd = [f"WANDB_TAGS={config.wandb_tags}"] + cmd

    # escape the command (e.g., --stop_strings "</answer>")
    for i in range(len(cmd)):
        if "</" in cmd[i]:
            cmd[i] = f"'{cmd[i]}'"

    # special logic to deal with escape like
    # python mason.py ... -- python x.py --dataset_mixer '{"trl-internal-testing/sentiment-trl-style": 1.0}'
    # we need to wrap the json string with single quote
    for idx in range(len(cmd)):
        if "{" in cmd[idx]:
            cmd[idx] = "'" + cmd[idx] + "'"

    setup_cmd = ""
    if not config.pure_docker_mode:
        setup_cmd = f"cd {os.getcwd()} && "

    # override accelerate call
    join_cmd = " ".join(cmd)
    if config.num_nodes > 1:
        if "--num_processes" not in join_cmd and "accelerate" in join_cmd:
            raise ValueError(
                "num_processes must be specified in the command for accelerate-based multi-node jobs."
            )
        join_cmd = re.sub(
            r"--num_processes (\d+)",
            lambda m: (
                f"--num_processes {int(m.group(1)) * config.num_nodes} "
                f"--num_machines {config.num_nodes} "
                "--machine_rank $BEAKER_REPLICA_RANK "
                "--main_process_ip $BEAKER_LEADER_REPLICA_HOSTNAME "
                "--main_process_port 29400 "
            ),
            join_cmd,
        )

    cmd = setup_cmd + join_cmd

    return cmd


def parse_commands() -> List[List[str]]:
    """
    Parse commands separated by '--' into list of command lists.

    E.g.:    launch.py [options] -- cmd1 arg1 -- cmd2 arg2
    Returns: [["cmd1", "arg1"], ["cmd2", "arg2"]]
    """
    if len(sys.argv) < 2:
        raise ValueError("No command provided. Usage: launch.py [options] -- command")

    try:
        first_cmd_idx = sys.argv.index("--")
    except ValueError:
        raise ValueError("No command separator '--' found. Usage: launch.py [options] -- command")

    # Get everything after first --
    remaining_args = sys.argv[first_cmd_idx + 1 :]

    if not remaining_args:
        raise ValueError("No command provided after '--'")

    # Split into separate commands on --
    commands = []
    current_cmd = []

    for arg in remaining_args:
        if arg == "--":
            if current_cmd:
                commands.append(current_cmd)
                current_cmd = []
        else:
            current_cmd.append(arg)

    if current_cmd:
        commands.append(current_cmd)

    if not commands:
        raise ValueError("No valid commands found")

    return commands


def launch_gantry(config: BeakerConfig):
    global_wandb_id = gen_uuid()

    beaker_client = bk.Beaker.from_env(default_workspace=config.workspace)

    # print(beaker_client.workspace.secrets())
    # beaker_secrets = [secret.name for secret in beaker_client.workspace.secrets()]
    beaker_secrets = []

    # whoami = beaker_client.account.whoami().name

    whoami = 'davidh'

    commands = parse_commands()

    full_commands = []
    for command in commands:
        full_commands += [make_command(command, config)]

    assert len(full_commands) == 1, "only one command supported for now"
    full_commands = full_commands[0]

    env_vars, env_secrets = get_env_vars(
        config.cluster,
        beaker_secrets,
        whoami,
        global_wandb_id,
        config.pure_docker_mode,
        config.num_nodes,
        config.env,
        config.secret,
        config.preemptible,
    )
    env_vars = [f"{var.name}={var.value}" for var in env_vars]
    env_secrets = [f"{var.name}={var.value}" for var in env_secrets]

    # mounts = get_mounts(config.beaker_datasets, config.cluster)
    # mounts = [
    #     "oe-adapt-default:/oe-adapt-default"
    #     "oe-training-default:/oe-training-default"
    #     "oe-eval-default:/oe-eval-default"
    # ]
    mounts = None # TODO: How to mount weka??

    ### TODO: Migrate from legacy launcher ???
    # pure_docker_mode
    # no_auto_dataset_cache
    # auto_output_dir_path
    # auto_checkpoint_state_dir
    
    # Launch the experiment
    launch_experiment( # launch_experiment()
        args=full_commands.split(' '),

        workspace=config.workspace,
        clusters=config.cluster,
        budget=config.budget,

        # datasets= # add ability to add this

        name=config.task_name,
        description=config.description,
        hostnames=config.hostname,
        beaker_image=config.image,
        gpus=config.gpus,
        preemptible=config.preemptible,
        retries=config.max_retries,
        mounts=mounts, # need to fix
        replicas=config.num_nodes,
        host_networking=not config.no_host_networking,
        env_vars=env_vars,
        env_secrets=env_secrets,
        yes=True,

        # new stuff
        # allow_dirty=True,
        dry_run=False,
        timeout=99999999, # only way to follow the experiment without canceling
        install="pip install -e '.[all]'",
    )


def main():
    # config = make_config(BeakerConfig(
    #     workspace="ai2/davidh",
    #     cluster=["ai2/jupiter-cirrascale-2"],
    #     budget="ai2/oe-eval"
    # ))
    config = make_config(BeakerConfig(
        workspace="ai2/davidh",
        cluster=["ai2/jupiter-cirrascale-2"],
        budget="ai2/oe-eval",
        # gpus=1
    ))
    launch_gantry(config)


if __name__ == "__main__":
    main()


"""
Notes on gantry:
    - Pro: Gantry has support
    - Con: Gantry installs deps on every job

I think we should do *both*. Mini mason has most of the defaults we need, gantry allows pulling the github container. We just need to write a connector to this function.

Finally, to make things faster, we could write a simple Dockerfile which pre-installs the dependencies and pushes through git actions (although not necessary I think? saves like 2 min of startup time). Then, at startup gantry will just pull the current codebase.


Next: How to do code execution? 
    - Sets of Dockerfiles should be a must -- This is standard in related work.
    - Could do (1) pre-built dockerfiles (2) launch execution jobs on beaker (so every tool call spins up a container). Could actually be quick to run maybe... (start with this for now. other options are AWS lambdas, or building your own load balancer)
"""
