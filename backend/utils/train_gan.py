import json
import os

# =============================================================================
# CONSTANTS
# =============================================================================

# Python imports for training script
TRAINING_IMPORTS = """
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from universal_gan import UniversalGAN
import json
import os
"""

# Main training loop code template
# Main training loop code template
TRAINING_CODE_TEMPLATE = """
if __name__ == "__main__":
    # Load configuration
    gan_training_config = {config_json}
    dataset_path = "{dataset_path}"
    
    config = gan_training_config
    
    # Initialize GAN
    gan = UniversalGAN(config)
    
    # Create dataset (NOT dataloader - UniversalGAN.train() creates its own DataLoader)
    dataset = create_dataset(dataset_path)
    
    print(f"Starting GAN training on {{config['modality']}} dataset from {{dataset_path}}")
    
    # Train the GAN (pass dataset, not dataloader)
    gan.train(dataset)
    
    print("Training complete.")
    
    # Generate samples
    generated = gan.generate(5)
    
    # Rescale from [-1, 1] -> [0, 1]
    generated = (generated.clamp(-1, 1) + 1) / 2
    
    # Save generated images
    save_image(generated, "generated_samples.png", nrow=5)
    
    print("Saved generated images to generated_samples.png")
"""

# SLURM script template
SLURM_SCRIPT_TEMPLATE = """
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={remote_dir}/slurm-%j.out
#SBATCH --error={remote_dir}/slurm-%j.err
#SBATCH --time={time_limit}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --cpus-per-task={num_cpus}
#SBATCH --mem={memory}
{allocation_line}

# Load prerequisite modules (required for python/3.11.4)
module purge
module load bzip2
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/{python_version}
module load cuda/{cuda_version}

# Verify python is available
echo "Python location: $(which python)"
python --version

# ALWAYS recreate environment - old one contains broken _bz2
echo "Recreating clean virtual environment..."
rm -rf $HOME/envs/gan_env
python -m venv $HOME/envs/gan_env

# Activate it
source $HOME/envs/gan_env/bin/activate

# Verify activation
echo "Pip location: $(which pip)"
echo "Python location: $(which python)"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install PyTorch with CUDA support (matching CUDA 12.4)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet

# Install other required packages
echo "Installing additional packages..."
pip install {pip_packages} --quiet

# Add current directory to Python path so it can find universal_gan.py
export PYTHONPATH="{remote_dir}:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Change to the job directory
cd {remote_dir}

echo "=========================================="
echo "Starting GAN training job..."
echo "=========================================="
python {remote_dir}/train_gan_job.py

echo "=========================================="
echo "Job completed!"
echo "=========================================="
"""

# Default SLURM configuration
DEFAULT_SLURM_CONFIG = {
    "job_name": "gan_training",
    "time_limit": "24:00:00",
    "num_gpus": 1,
    "num_cpus": 8,
    "memory": "32G",
    "cuda_version": "12.4.1",
    "python_version": "3.11.4",
}

# Default required pip packages
# Default required pip packages (torch installed separately with CUDA support)
DEFAULT_PIP_PACKAGES = [
    "numpy",
    "pandas",
    "pillow",
    "matplotlib",
]

# =============================================================================
# TRAINING SCRIPT GENERATOR
# =============================================================================

def create_gan_training_script(dataset_code, gan_training_config, dataset_path):
    """
    Creates a complete GAN training script.
    
    Args:
        dataloader_code (str): Code defining the create_dataloader() function
        gan_training_config (dict): Configuration dictionary for the GAN
        dataset_path (str): Path to the dataset
    
    Returns:
        str: Complete training script as a string
    """
    # Clean up config - convert float values to int where appropriate
    cleaned_config = {}
    for key, value in gan_training_config.items():
        if isinstance(value, float):
            # Convert floats like 64.0, 4.0 to integers
            if value.is_integer():
                cleaned_config[key] = int(value)
            else:
                cleaned_config[key] = value
        elif isinstance(value, list):
            # Handle lists (like img_shape) - convert floats to ints
            cleaned_config[key] = [
                int(v) if isinstance(v, float) and v.is_integer() else v 
                for v in value
            ]
        else:
            cleaned_config[key] = value
    
    config_json = json.dumps(cleaned_config, indent=4)
    
    script = f"""{TRAINING_IMPORTS.strip()}

{dataset_code.strip()}

{TRAINING_CODE_TEMPLATE.format(config_json=config_json, dataset_path=dataset_path).strip()}
"""
    
    return script


# =============================================================================
# SLURM SCRIPT GENERATOR
# =============================================================================

def create_slurm_script(
    remote_dir,
    job_name=None,
    time_limit=None,
    num_gpus=None,
    num_cpus=None,
    memory=None,
    allocation=None,
    cuda_version=None,
    python_version=None,
    required_pip_packages=None,
    train_script_name="train_gan_job.py"
):
    """
    Creates a SLURM submission script for GAN training.
    
    Args:
        remote_dir (str): Remote directory path where scripts will be located
        job_name (str, optional): Name of the SLURM job
        time_limit (str, optional): Time limit (e.g., "24:00:00")
        num_gpus (int, optional): Number of GPUs to request
        num_cpus (int, optional): Number of CPUs per task
        memory (str, optional): Memory allocation (e.g., "32G")
        allocation (str, optional): Account/allocation name
        cuda_version (str, optional): CUDA module version
        python_version (str, optional): Python module version
        required_pip_packages (list, optional): List of pip packages to install
        train_script_name (str, optional): Name of the training script
    
    Returns:
        str: Complete SLURM script as a string
    
    Example:
        >>> script = create_slurm_script(
        ...     remote_dir="/home/user/gan_project",
        ...     job_name="my_gan",
        ...     allocation="my_account"
        ... )
    """
    # Use defaults if not provided
    config = DEFAULT_SLURM_CONFIG.copy()
    
    if job_name is not None:
        config["job_name"] = job_name
    if time_limit is not None:
        config["time_limit"] = time_limit
    if num_gpus is not None:
        config["num_gpus"] = num_gpus
    if num_cpus is not None:
        config["num_cpus"] = num_cpus
    if memory is not None:
        config["memory"] = memory
    if cuda_version is not None:
        config["cuda_version"] = cuda_version
    if python_version is not None:
        config["python_version"] = python_version
    
    # Handle allocation line
    allocation_line = f"#SBATCH --account={allocation}" if allocation else ""
    
    # Handle pip packages
    if required_pip_packages is None:
        required_pip_packages = DEFAULT_PIP_PACKAGES
    pip_packages = ' '.join(required_pip_packages)
    
    # Generate script
    slurm_script = SLURM_SCRIPT_TEMPLATE.format(
        job_name=config["job_name"],
        remote_dir=remote_dir,
        time_limit=config["time_limit"],
        num_gpus=config["num_gpus"],
        num_cpus=config["num_cpus"],
        memory=config["memory"],
        allocation_line=allocation_line,
        cuda_version=config["cuda_version"],
        python_version=config["python_version"],
        pip_packages=pip_packages
    ).strip()
    
    return slurm_script


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_training_script(script_content, output_path):
    """
    Save training script to file.
    
    Args:
        script_content (str): The script content
        output_path (str): Path where to save the script
    """
    unix_script = script_content.replace("\r\n", "\n").replace("\r", "\n")
    with open(output_path, "w", newline="\n") as f:
        f.write(unix_script)
    print(f"✅ Training script saved to: {output_path}")


def save_slurm_script(script_content, output_path):
    """
    Save SLURM script to file.
    
    Args:
        script_content (str): The script content
        output_path (str): Path where to save the script
    """
    unix_script = script_content.replace("\r\n", "\n").replace("\r", "\n")
    with open(output_path, "w", newline="\n") as f:
        f.write(unix_script)
    # Make executable
    os.chmod(output_path, 0o755)
    print(f"✅ SLURM script saved to: {output_path}")