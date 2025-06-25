import os
import shutil
from typing import Optional


def calibrate_checkpoint_state_dir(checkpoint_state_dir: str) -> None:
    """
    Find the latest valid checkpoint directory and update the 'latest' file.

    Edge case:
    it's possible sometimes the checkpoint save / upload (1) completely or (2) partially failed (i.e., having incomplete files),
    so we should fall back to a checkpoint that actually exists -- we should pick the latest folder which has the most files.
    The folders look like this:
    checkpoint_state_dir/global_step14
    checkpoint_state_dir/global_step15
    ...
    checkpoint_state_dir/global_step20
    we would then update the `checkpoint_state_dir/latest` file
    with the latest global_step number.
    """
    if not os.path.exists(checkpoint_state_dir):
        return

    # Get all checkpoint directories
    checkpoint_dirs = [
        d
        for d in os.listdir(checkpoint_state_dir)
        if d.startswith("global_step") and os.path.isdir(os.path.join(checkpoint_state_dir, d))
    ]

    if not checkpoint_dirs:
        return

    # Create a list of (dir_name, step_number, file_count) tuples
    checkpoint_info = []
    for dir_name in checkpoint_dirs:
        step_number = int(dir_name.replace("global_step", ""))
        dir_path = os.path.join(checkpoint_state_dir, dir_name)
        # Count files in the directory, not directories
        file_count = len(os.listdir(dir_path))
        checkpoint_info.append((dir_name, step_number, file_count))

    # Find the maximum file count
    max_file_count = max(info[2] for info in checkpoint_info)

    # Filter to only include checkpoints with the maximum file count
    valid_checkpoints = [info for info in checkpoint_info if info[2] >= max_file_count]
    invalid_checkpoints = [info for info in checkpoint_info if info[2] < max_file_count]

    # Remove invalid checkpoint directories
    for dir_name, _, _ in invalid_checkpoints:
        checkpoint_path = os.path.join(checkpoint_state_dir, dir_name)
        print(f"Removing incomplete checkpoint: {dir_name}")
        shutil.rmtree(checkpoint_path)

    # Sort by step number (descending)
    valid_checkpoints.sort(key=lambda x: x[1], reverse=True)

    # Get the latest valid checkpoint
    latest_checkpoint, latest_step, file_count = valid_checkpoints[0]

    # Update the 'latest' file
    with open(os.path.join(checkpoint_state_dir, "latest"), "w") as f:
        f.write(f"global_step{latest_step}")

    print(
        f"Found latest checkpoint: {latest_checkpoint} with {file_count} files, "
        f"updated 'latest' file to global_step{latest_step}"
    )


def clean_last_n_checkpoints_deepspeed(output_dir: str, keep_last_n_checkpoints: int) -> None:
    # Identify checkpoint files that follow the pattern global_step{number}
    all_files = os.listdir(output_dir)
    checkpoint_files = []
    for file in all_files:
        if file.startswith("global_step") and file[len("global_step") :].isdigit():
            checkpoint_files.append(file)

    # Sort checkpoints by step number
    checkpoints = sorted(checkpoint_files, key=lambda x: int(x[len("global_step") :]), reverse=True)

    # Keep the N most recent checkpoints and remove the rest
    if keep_last_n_checkpoints >= 0 and len(checkpoints) > keep_last_n_checkpoints:
        for checkpoint in checkpoints[keep_last_n_checkpoints:]:
            print(f"Removing checkpoint {checkpoint}")
            checkpoint_path = os.path.join(output_dir, checkpoint)
            if os.path.isdir(checkpoint_path):
                shutil.rmtree(checkpoint_path)
            elif os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)

    # Keep special files like zero_to_fp32.py and latest
    print("Remaining files:" + str(os.listdir(output_dir)))


def wandb_login() -> Optional[str]:
    import wandb

    wandb.login()
    api = wandb.Api()
    current_user = api.viewer
    teams = current_user.teams
    if "ai2-llm" in teams:
        return "ai2-llm"
    return None
