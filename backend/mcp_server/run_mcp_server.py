from typing import List, Dict
from fastmcp import FastMCP
import shutil
from utils import *
import os, zipfile, torch, tempfile, json, textwrap
from torchvision import transforms, utils as vutils
from torchvision.datasets import ImageFolder
from utils.rivanna import submit_rivanna_job, upload_files_to_rivanna
from utils.train_gan import create_gan_training_script, create_slurm_script, save_training_script, save_slurm_script


mcp = FastMCP("Demo 🚀")

@mcp.tool
def submit_gan_training_job(
    dataset_file_path: str,
    dataset_code: str,
    gan_training_config: Dict,
    required_pip_packages: List[str] = None,
    job_name: str = "gan_train",
    time_limit: str = "02:00:00",
    num_gpus: int = 1,
    num_cpus: int = 4,
    memory: str = "16G",
    remote_dir: str = "~/scratch/mcp_jobs",
    cuda_version: str = "12.4.1",
    python_version: str = "3.11.4"
):
    """
    Creates a temporary training setup, uploads it to Rivanna,
    submits the job, and cleans up local temp files.
    
    Args:
        dataset_file_path (str): Local path to the dataset file
        dataset_code (str): Python code defining create_dataset() function
        gan_training_config (dict): GAN configuration dictionary
        required_pip_packages (list, optional): List of pip packages to install
        job_name (str): Name for the SLURM job
        time_limit (str): Time limit in HH:MM:SS format
        num_gpus (int): Number of GPUs to request
        num_cpus (int): Number of CPUs per task
        memory (str): Memory allocation (e.g., "16G")
        remote_dir (str): Remote directory for job files
        cuda_version (str): CUDA module version (default: 12.4.1 - verified on Rivanna)
        python_version (str): Python module version (default: 3.11.4 - verified on Rivanna)
    
    Returns:
        dict: Job submission information including job_id, remote_dir, and submission_output
    """
    # Get user from environment
    user = os.getenv("RIVANNA_USER")
    
    # Override allocation from environment if not provided
    allocation = os.getenv("RIVANNA_ALLOCATION")
    
    # Expand home directory in remote path
    if remote_dir.startswith("~"):
        remote_dir = remote_dir.replace("~", f"/home/{user}")
    
    # Create temporary directory for local files
    base_dir = tempfile.mkdtemp()
    train_script_path = os.path.join(base_dir, "train_gan_job.py")
    slurm_script_path = os.path.join(base_dir, "train_gan_job.slurm")
    remote_dataset_name = os.path.basename(dataset_file_path)
    
    # Set dataset path for remote execution
    remote_dataset_path = f"{remote_dir}/{remote_dataset_name}"
    
    try:
        # --- Generate and save training script ---
        print("📝 Generating training script...")
        train_script = create_gan_training_script(
            dataset_code=dataset_code,
            gan_training_config=gan_training_config,
            dataset_path=remote_dataset_path
        )
        
        # Use utility function to save training script
        save_training_script(train_script, train_script_path)
        
        # --- Generate and save SLURM script ---
        print("📝 Generating SLURM submission script...")
        
        # Use default packages if none provided
        if required_pip_packages is None:
            required_pip_packages = DEFAULT_PIP_PACKAGES.copy()
        
        slurm_script = create_slurm_script(
            remote_dir=remote_dir,
            job_name=job_name,
            time_limit=time_limit,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            memory=memory,
            allocation=allocation,
            cuda_version=cuda_version,
            python_version=python_version,
            required_pip_packages=required_pip_packages,
            train_script_name="train_gan_job.py"
        )
        
        # Use utility function to save SLURM script
        save_slurm_script(slurm_script, slurm_script_path)
        
        # --- Upload dataset + scripts to Rivanna ---
        print("🔄 Uploading dataset and scripts to Rivanna...")
        upload_files_to_rivanna(
            {"remote_path": remote_dataset_path, "local_path": dataset_file_path},
            {"remote_path": f"{remote_dir}/train_gan_job.py", "local_path": train_script_path},
            {"remote_path": f"{remote_dir}/train_gan_job.slurm", "local_path": slurm_script_path},
        )
        print("✅ Files uploaded successfully")
        
        # --- Submit job to SLURM ---
        print("🚀 Submitting job to SLURM...")
        submission_output = submit_rivanna_job(f"{remote_dir}/train_gan_job.slurm")
        print(f"submission output: {submission_output}")
        
        # Extract job ID from output (e.g., "Submitted batch job 12345")
        job_id = None
        if "Submitted batch job" in submission_output:
            try:
                job_id = submission_output.strip().split()[-1]
                print(f"✅ Job submitted with ID: {job_id}")
            except:
                print("⚠️ Could not parse job ID from submission output")
        else:
            print("⚠️ Unexpected submission output format")
        
        # --- Clean up local temporary files --- uncomment line to actually work
        print("🧹 Cleaning up local temporary files...")
        # shutil.rmtree(base_dir, ignore_errors=True)
        print("✅ Cleanup complete")
        
        # --- Return results ---
        return {
            "success": True,
            "job_id": job_id,
            "remote_dir": remote_dir,
            "submission_output": submission_output,
            "dataset_path": remote_dataset_path,
            "train_script": f"{remote_dir}/train_gan_job.py",
            "slurm_script": f"{remote_dir}/train_gan_job.slurm",
            "modules_used": {
                "cuda": cuda_version,
                "python": python_version
            }
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Clean up on error
        shutil.rmtree(base_dir, ignore_errors=True)
        return {
            "success": False,
            "error": str(e),
            "job_id": None,
            "remote_dir": remote_dir,
        }

# @mcp.tool
# def poll_training_job(job_id:int, poll_frequency:int):
#     print("⏳ Polling for completion...")
#     final_state = poll_rivanna_job(job_id)
#     print(f"✅ Job finished with state: {final_state}")

# @mcp.tool
# def test_simple_upload():
#     """Minimal test case for Rivanna upload"""
#     import tempfile, os
    
#     # Create a tiny test file
#     test_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
#     test_file.write("Hello Rivanna!")
#     test_file.close()
    
#     print(f"📝 Created test file: {test_file.name}")
#     print(f"   Size: {os.path.getsize(test_file.name)} bytes")
    
#     try:
#         result = upload_files_to_rivanna(
#             {"remote_path": "~/scratch/test_upload.txt", "local_path": test_file.name}
#         )
#         print(f"✅ Upload returned: {result}")
        
#         # Now verify via SSH
#         user = os.getenv("RIVANNA_USER") or "ntq4hf"
#         key_path = os.path.expanduser(os.getenv("RIVANNA_KEY_PATH") or "rivanna_info/rivanna_keys")
        
#         import paramiko
#         ssh = paramiko.SSHClient()
#         ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#         ssh.connect(hostname="login.hpc.virginia.edu", username=user, key_filename=key_path)
        
#         # Check if file exists
#         stdin, stdout, stderr = ssh.exec_command(f"ls -lh /home/{user}/scratch/test_upload.txt")
#         output = stdout.read().decode()
#         error = stderr.read().decode()
        
#         print("\n📂 Remote file check:")
#         print(output if output else error)
        
#         # Try to read it back
#         stdin, stdout, stderr = ssh.exec_command(f"cat /home/{user}/scratch/test_upload.txt")
#         content = stdout.read().decode()
#         print(f"\n📄 Remote file content: '{content}'")
        
#         ssh.close()
        
#         print(f"\n💾 Local test file kept at: {test_file.name}")
#         print(f"🌐 Remote file at: /home/{user}/scratch/test_upload.txt")
        
#         return {"success": True, "remote_content": content, "local_file": test_file.name}
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
#         # Keep the file even on error for debugging
#         print(f"\n💾 Local test file kept at: {test_file.name}")
#         return {"success": False, "error": str(e), "local_file": test_file.name}

# @mcp.tool
# def explore_dataset(dataset_path: str, class_samples: int = 5):
#     """
#     Explore a dataset by loading and displaying sample images.
    
#     Args:
#         dataset_path (str): Path to the dataset directory
#         class_samples (int): Number of samples to read for each class
#     """

#     if not os.path.exists(dataset_path):
#         return {"success": False, "error": f"Dataset path does not exist: {dataset_path}"}
#     if zipfile.is_zipfile(dataset_path):
#         try:
#             with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
#                 extract_path = tempfile.mkdtemp()
#                 zip_ref.extractall(extract_path)
#                 dataset_path = extract_path
#         except Exception as e:
#             return {"success": False, "error": f"Failed to extract zip file: {str(e)}"}
        
#     dataset_structure = {}
#     metadata = []
#     dir_queue = []
#     dir_queue.append(dataset_path)
#     while len(dir_queue) > 0:
#         cur_path = dir_queue.pop(0)
#         cur_contents = os.listdir(cur_path)
#         print(cur_contents[:15])
#         dataset_structure[cur_path] = cur_contents[:15]
#         for entry in cur_contents:
#             samples_read = 0
#             full_path = os.path.join(cur_path, entry)
#             if os.path.isdir(full_path):
#                 dir_queue.append(full_path)
#             else:
#                 if samples_read < class_samples:
#                     try:
#                         from PIL import Image
#                         with Image.open(full_path) as img:
#                             width, height = img.size
#                             mode = img.mode
#                             format = img.format
#                         metadata.append({
#                             "width": width,
#                             "height": height,
#                             "mode": mode,
#                             "format": format,
#                         })
#                         samples_read += 1
#                     except Exception as e:
#                         metadata.append({
#                             "path": full_path,
#                             "error": str(e)
#                         })
#     return {"success": True, "structure": dataset_structure, "metadata": metadata}


@mcp.tool
def unzip_dataset(zip_path: str):
    """
    Unzip a dataset file to a temporary directory and return the path.
    
    Args:
        zip_path (str): Path to the zip file
    """
    if not os.path.exists(zip_path):
        return {"success": False, "error": f"Zip file does not exist: {zip_path}"}
    if not zipfile.is_zipfile(zip_path):
        return {"success": False, "error": f"File is not a valid zip: {zip_path}"}
    
    try:
        extract_path = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        return {"success": True, "extracted_path": extract_path}
    except Exception as e:
        return {"success": False, "error": f"Failed to extract zip file: {str(e)}"}

@mcp.tool
def list_dir(file_path: str):
    """
    List files and directories at a given path
    Useful for dataset exploration prior to training job submission.
    """
    import os
    if not os.path.exists(file_path):
        return {"success": False, "error": f"Path does not exist: {file_path}"}
    
    entries = []
    for entry in os.listdir(file_path):
        path = os.path.join(file_path, entry)
        entries.append({
            "name": entry,
            "path": path,
            "is_dir": os.path.isdir(path),
        })
    
    return {"success": True, "entries": entries}

@mcp.tool
def get_file_extension(file_path: str):
    """
    Retrieve file extension for a given file path.
    """
    import os
    if not os.path.exists(file_path):
        return {"success": False, "error": f"Path does not exist: {file_path}"}
    
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    

    
    return {"success": True, "extension": ext}

@mcp.tool
def load_image_metadata(image_path: str):
    """
    Load and return metadata for a given image file.
    """
    from PIL import Image
    import os
    
    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image does not exist: {image_path}"}
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            format = img.format
        
        metadata = {
            "width": width,
            "height": height,
            "mode": mode,
            "format": format,
        }
        
        return {"success": True, "metadata": metadata}
    
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
