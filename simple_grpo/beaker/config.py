import sys

from omegaconf import OmegaConf
from rich.pretty import pprint


def apply_overrides(config):
    base = OmegaConf.structured(config)
    try:
        # Ignore everything after the first " -- "
        cmd_sep_idx = sys.argv.index("--")
        cli_args = [arg.lstrip("--") for arg in sys.argv[1:cmd_sep_idx]]
    except ValueError:
        cli_args = [arg.lstrip("--") for arg in sys.argv[1:]]
    overrides = OmegaConf.from_cli(cli_args)
    merged = OmegaConf.merge(base, overrides)
    return OmegaConf.to_object(merged)


def make_config(default_config):
    config = apply_overrides(default_config)
    pprint(config, expand_all=True)
    return config
