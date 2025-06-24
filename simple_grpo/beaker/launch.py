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
    pure_docker_mode: bool = True # If false, will cd into os.getcwd()
    no_auto_dataset_cache: bool = False
    auto_output_dir_path: str = "/oe-eval-default/davidh/deletable_checkpoint"
    auto_checkpoint_state_dir: str = (
        "/oe-eval-default/davidh/deletable_checkpoint_states"
    )
    beaker_datasets: List[Dict[str, str]] = field(default_factory=list) # TODO: Add parser from mason.py
    env: List[Dict[str, str]] = field(default_factory=list) # TODO: Add parser from mason.py
    secret: List[Dict[str, str]] = field(default_factory=list) # TODO: Add parser from mason.py
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
        if (
            "--num_processes" not in join_cmd
            and "accelerate" in join_cmd
        ):
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


def make_task_spec(
    config: BeakerConfig,
    full_command: str,
    i: int,
    beaker_secrets: str,
    whoami: str,
):
    global_wandb_id = gen_uuid()

    if config.num_nodes > 1 and not all(
        c in INTERCONNECT_CLUSTERS for c in config.cluster
    ):        
        raise ValueError(
            f"Interconnect clusters are required for multi-node jobs; please only use the following clusters: {INTERCONNECT_CLUSTERS}"
        )
    
    if config.image == "ai2/cuda11.8-cudnn8-dev-ubuntu20.04" and any(
        c in GCP_CLUSTERS for c in config.cluster
    ):
        raise ValueError(
            "GCP clusters do not have the dev filesystem, please use a proper image"
        )

    if config.hostname is not None:
        constraints = bk.Constraints(hostname=config.hostname)
    else:
        constraints = bk.Constraints(cluster=config.cluster)

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
    env_vars = env_vars + env_secrets # combine both!

    mounts = get_mounts(config.beaker_datasets, config.cluster)
    
    spec = bk.TaskSpec(
        name=f"{config.task_name}__{i}",
        image=bk.ImageSource(beaker=config.image),
        command=["/bin/bash", "-c"],
        arguments=[full_command],
        result=bk.ResultSpec(path="/output"),
        datasets=mounts,
        context=bk.TaskContext(
            priority=bk.Priority(config.priority), preemptible=config.preemptible
        ),
        constraints=constraints,
        env_vars=env_vars,
        resources=bk.TaskResources(gpu_count=config.gpus),
        replicas=config.num_nodes,
    )
    
    if config.num_nodes > 1:
        spec.leader_selection = True
        spec.propagate_failure = True
        spec.propagate_preemption = True
    if config.no_host_networking:
        spec.host_networking = False
    else:
        spec.host_networking = True

    return spec


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
    remaining_args = sys.argv[first_cmd_idx + 1:]
    
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


def launch(config: BeakerConfig):
    beaker_client = bk.Beaker.from_env(default_workspace=config.workspace)
    beaker_secrets = [secret.name for secret in beaker_client.workspace.secrets()]
    whoami = beaker_client.account.whoami().name

    commands = parse_commands()

    full_commands = []
    for command in commands:
        full_commands += [make_command(command, config)]

    experiment_spec = bk.ExperimentSpec(
        description=config.description,
        tasks=[
            make_task_spec(
                config,
                full_command,
                i,
                beaker_secrets,
                whoami,
            )
            for i, full_command in enumerate(full_commands)
        ],
        budget=config.budget,
        retry=bk.RetrySpec(allowed_task_retries=config.max_retries),
    )

    console.rule(f"[bold pink]Beaker experiment spec: [/bold pink]")
    pprint(experiment_spec)

    console.rule(f"[bold blue]Commands: [/bold blue]")
    for cmd in full_commands:
        console.print(Text(cmd))

    exp = beaker_client.experiment.create(spec=experiment_spec)
    console.log(f"Launched: https://beaker.org/ex/{exp.id}")


def main():
    config = make_config(BeakerConfig(
        workspace="ai2/davidh",
        cluster=["ai2/jupiter-cirrascale-2"],
        budget="ai2/oe-eval"
    ))
    launch(config)


if __name__ == "__main__":
    main()