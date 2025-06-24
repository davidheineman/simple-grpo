import os
from typing import Dict, List
from constants import WEKA_CLUSTERS, GCP_CLUSTERS
from beaker import EnvVar, DataMount, DataSource

USEFUL_SECRETS = [
    "HF_TOKEN",
    "WANDB_API_KEY",
    "BEAKER_TOKEN",
    "OPENAI_API_KEY",
    # litellm expects these env vars
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "ANTHROPIC_API_KEY",
]


def get_env_vars(
    cluster: List[str],
    beaker_secrets: List[str],
    whoami: str,
    global_wandb_id: str,
    pure_docker_mode: bool = False,
    num_nodes: int = 1,
    additional_env_vars: List[Dict[str, str]] = None,
    additional_secrets: List[Dict[str, str]] = None,
    preemptible: bool = True,
):
    env_vars = []

    # Add user-specified environment variables
    for env_var in additional_env_vars:
        env_vars.append(EnvVar(name=env_var["name"], value=env_var["value"]))

    # Add user-specified secrets
    for secret in additional_secrets:
        env_vars.append(
            EnvVar(
                name=secret["name"],
                secret=secret["value"],
            )
        )

    for useful_secret in USEFUL_SECRETS:
        if f"{whoami}_{useful_secret}" in beaker_secrets:
            env_vars.append(
                EnvVar(
                    name=useful_secret,
                    secret=f"{whoami}_{useful_secret}",
                )
            )
        elif useful_secret in beaker_secrets:
            env_vars.append(
                EnvVar(
                    name=useful_secret,
                    secret=useful_secret,
                )
            )

    # use the user's PATH; including the conda / python PATH
    if not pure_docker_mode:
        env_vars.extend(
            [
                EnvVar(
                    name="PATH",
                    value=os.getenv("PATH"),
                ),
            ]
        )

    # if all cluster is in weka, we mount the weka
    if all(c in WEKA_CLUSTERS for c in cluster):
        env_vars.extend(
            [
                EnvVar(
                    name="HF_HOME",
                    value="/oe-eval-default/allennlp/.cache/huggingface",
                ),
                EnvVar(
                    name="HF_DATASETS_CACHE",
                    value="/oe-eval-default/allennlp/.cache/huggingface",
                ),
                EnvVar(
                    name="HF_HUB_CACHE",
                    value="/oe-eval-default/allennlp/.cache/hub",
                ),
                EnvVar(
                    name="CHECKPOINT_OUTPUT_DIR",
                    value=f"/oe-eval-default/allennlp/deletable_checkpoint_states/{global_wandb_id}",
                ),
            ]
        )
        if num_nodes > 1:
            env_vars.extend(
                [
                    EnvVar(
                        name="NCCL_SOCKET_IFNAME",
                        value="ib",
                    ),
                    EnvVar(
                        name="NCCL_IB_HCA",
                        value="^=mlx5_bond_0",
                    ),
                    EnvVar(
                        name="NCCL_DEBUG",
                        value="INFO",
                    ),
                ]
            )
    # if all cluster is in gcp we add the following env

    elif all(c in GCP_CLUSTERS for c in cluster):
        env_vars.extend(
            [
                EnvVar(
                    name="HF_HOME",
                    value="/filestore/.cache/huggingface",
                ),
                EnvVar(
                    name="HF_DATASETS_CACHE",
                    value="/filestore/.cache/huggingface",
                ),
                EnvVar(
                    name="HF_HUB_CACHE",
                    value="/filestore/.cache/hub",
                ),
                EnvVar(
                    name="HF_HUB_ENABLE_HF_TRANSFER",
                    value="0",  # we disable it because GCP is weird on uploading to the hub
                ),
            ]
        )
        if num_nodes > 1:
            env_vars.extend(
                [
                    EnvVar(
                        name="LD_LIBRARY_PATH",
                        value=r"/var/lib/tcpxo/lib64:${LD_LIBRARY_PATH}",
                    ),
                    EnvVar(
                        name="NCCL_CROSS_NIC",
                        value="0",
                    ),
                    EnvVar(
                        name="NCCL_ALGO",
                        value="Ring,Tree",
                    ),
                    EnvVar(
                        name="NCCL_PROTO",
                        value="Simple",
                    ),
                    EnvVar(
                        name="NCCL_MIN_NCHANNELS",
                        value="4",
                    ),
                    EnvVar(
                        name="NCCL_P2P_NET_CHUNKSIZE",
                        value="524288",
                    ),
                    EnvVar(
                        name="NCCL_P2P_PCI_CHUNKSIZE",
                        value="524288",
                    ),
                    EnvVar(
                        name="NCCL_P2P_NVL_CHUNKSIZE",
                        value="1048576",
                    ),
                    EnvVar(
                        name="NCCL_FASTRAK_NUM_FLOWS",
                        value="2",
                    ),
                    EnvVar(
                        name="NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL",
                        value="0",
                    ),
                    EnvVar(
                        name="NCCL_BUFFSIZE",
                        value="8388608",
                    ),
                    EnvVar(
                        name="NCCL_FASTRAK_USE_SNAP",
                        value="1",
                    ),
                    EnvVar(
                        name="CUDA_VISIBLE_DEVICES",
                        value="0,1,2,3,4,5,6,7",
                    ),
                    EnvVar(
                        name="NCCL_NET_GDR_LEVEL",
                        value="PIX",
                    ),
                    EnvVar(
                        name="NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING",
                        value="0",
                    ),
                    EnvVar(
                        name="NCCL_TUNER_PLUGIN",
                        value="libnccl-tuner.so",
                    ),
                    EnvVar(
                        name="NCCL_TUNER_CONFIG_PATH",
                        value="/var/lib/tcpxo/lib64/a3plus_tuner_config.textproto",
                    ),
                    EnvVar(
                        name="NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE",
                        value="/var/lib/tcpxo/lib64/a3plus_guest_config.textproto",
                    ),
                    EnvVar(
                        name="NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS",
                        value="600000",
                    ),
                    EnvVar(
                        name="NCCL_NVLS_ENABLE",
                        value="0",
                    ),
                    EnvVar(
                        name="NCCL_DEBUG",
                        value="WARN",
                    ),
                    EnvVar(
                        name="NCCL_FASTRAK_CTRL_DEV",
                        value="enp0s12",
                    ),
                    EnvVar(
                        name="NCCL_FASTRAK_IFNAME",
                        value="enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0",
                    ),
                    EnvVar(
                        name="NCCL_SOCKET_IFNAME",
                        value="enp0s12",
                    ),
                    EnvVar(
                        name="NCCL_USE_SNAP",
                        value="1",
                    ),
                    EnvVar(
                        name="NCCL_FASTRAK_USE_LLCM",
                        value="1",
                    ),
                    EnvVar(
                        name="NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY",
                        value="/dev/aperture_devices",
                    ),
                ]
            )
    # don't mount anything; assume no cache
    else:
        pass

    if preemptible:
        env_vars.extend(
            [
                EnvVar(
                    name="WANDB_RUN_ID",
                    value=global_wandb_id,
                ),
                EnvVar(
                    name="WANDB_RESUME",
                    value="allow",
                ),
            ]
        )

    return env_vars


def get_mounts(beaker_datasets, cluster: List[str]):
    res = []

    if all(c in WEKA_CLUSTERS for c in cluster):
        res = [
            DataMount(
                source=DataSource(weka="oe-adapt-default"),
                mount_path="/oe-adapt-default",
            ),
            DataMount(
                source=DataSource(weka="oe-training-default"),
                mount_path="/oe-training-default",
            ),
            DataMount(
                source=DataSource(weka="oe-eval-default"),
                mount_path="/oe-eval-default",
            ),
        ]
    elif all(c in GCP_CLUSTERS for c in cluster):
        res = [
            DataMount(
                source=DataSource(host_path="/mnt/filestore_1"),
                mount_path="/filestore",
            ),
        ]

    for beaker_dataset in beaker_datasets:
        to_append = DataMount(
            source=DataSource(beaker=beaker_dataset["beaker"]),
            mount_path=beaker_dataset["mount_path"],
        )
        res.append(to_append)

    return res
