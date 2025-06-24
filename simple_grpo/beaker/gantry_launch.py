"""
Gantry is essentially a wrapper around this python command

pip install beaker-gantry

gantry run --timeout -1 --allow-dirty --workspace ai2/davidh --budget ai2/oe-eval -- python -c 'print("Hello, World!")'

Notes on gantry:
    - Pro: Gantry has support
    - Con: Gantry installs deps on every job

I think we should do *both*. Mini mason has most of the defaults we need, gantry allows pulling the github container. We just need to write a connector to this function.

Finally, to make things faster, we could write a simple Dockerfile which pre-installs the dependencies and pushes through git actions (although not necessary I think? saves like 2 min of startup time). Then, at startup gantry will just pull the current codebase.


Next: How to do code execution? 
    - Sets of Dockerfiles should be a must -- This is standard in related work.
    - Could do (1) pre-built dockerfiles (2) launch execution jobs on beaker (so every tool call spins up a container). Could actually be quick to run maybe... (start with this for now. other options are AWS lambdas, or building your own load balancer)


Poor man's gantry?
    - Pass env vars: GITHUB_REPO, GIT_REF, GIT_BRANCH (optional), GITHUB_TOKEN (optional)
"""


import beaker as bk
from simple_grpo.beaker.config import make_config
from simple_grpo.beaker.defaults import get_env_vars, get_mounts
from simple_grpo.beaker.launch import BeakerConfig, gen_uuid, make_command, parse_commands
# from gantry.commands.run import run
from gantry.api import launch_experiment


### ???
# pure_docker_mode
# no_auto_dataset_cache
# auto_output_dir_path
# auto_checkpoint_state_dir

    
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
        timeout=99999999 # only way to follow the experiment without canceling
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
        gpus=1
    ))
    launch_gantry(config)


if __name__ == "__main__":
    main()
