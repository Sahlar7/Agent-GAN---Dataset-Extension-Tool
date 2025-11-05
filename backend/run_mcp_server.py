from typing import List, Dict
from fastmcp import FastMCP
from utils import *
import os, zipfile, torch, tempfile, json, textwrap
from torchvision import transforms, utils as vutils
from torchvision.datasets import ImageFolder
from utils.rivanna import submit_rivanna_job, upload_files_to_rivanna


mcp = FastMCP("Demo 🚀")

@mcp.tool
def submit_gan_training_job(
    dataset_file_path: str,
    dataset_loader_code: str,
    gan_training_config: Dict,
    required_pip_packages: List[str],
    job_name: str = "gan_train",
    time_limit: str = "02:00:00",
    num_gpus: int = 1,
    num_cpus: int = 4,
    memory: str = "16G",
    remote_dir: str = "~/mcp_jobs"
):
    """
    Creates a temporary training setup, uploads it to Rivanna,
    submits the job, and cleans up local temp files.
    """

    base_dir = tempfile.mkdtemp()
    train_script_path = os.path.join(base_dir, "train_gan_job.py")
    slurm_script_path = os.path.join(base_dir, "train_gan_job.slurm")
    remote_dataset_name = os.path.basename(dataset_file_path)

    # --- Write training script ---
    train_script = textwrap.dedent(f"""
        import torch
        from torchvision.utils import save_image
        from torch.utils.data import DataLoader
        from universal_gan import UniversalGAN
        import json, tempfile, os

        {dataset_loader_code}

        gan_training_config = {json.dumps(gan_training_config, indent=4)}

        if __name__ == "__main__":
            dataset_path = "{remote_dir}/{remote_dataset_name}"
            config = gan_training_config

            gan = UniversalGAN(config)
            dataloader = create_dataloader(dataset_path, batch_size=config.get("batch_size", 64))

            print(f"🚀 Starting GAN training on {{config['modality']}} dataset from {{dataset_path}}")
            gan.train(dataloader)
            print("✅ Training complete.")
            generated = gan.generate(5)
            # Rescale from [-1, 1] → [0, 1]
            generated = (generated.clamp(-1, 1) + 1) / 2

            # Save to file
            from torchvision.utils import save_image
            save_image(generated, "generated_samples.png", nrow=5)

            print("✅ Saved generated images to generated_samples.png")

    """).strip()
    with open(train_script_path, "w") as f:
        f.write(train_script)

    # --- Write SLURM script ---
    slurm_script = textwrap.dedent(f"""
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --output={remote_dir}/slurm-%j.out
        #SBATCH --error={remote_dir}/slurm-%j.err
        #SBATCH --time={time_limit}
        #SBATCH --partition=gpu
        #SBATCH --gres=gpu:{num_gpus}
        #SBATCH --cpus-per-task={num_cpus}
        #SBATCH --mem={memory}

        module load python/3.10 cuda/12.1
        source ~/envs/gan_env/bin/activate

        pip install {' '.join(required_pip_packages)} --quiet

        echo "Running GAN training job..."
        python {remote_dir}/train_gan_job.py
    """).strip()
    with open(slurm_script_path, "w") as f:
        f.write(slurm_script)

    # --- Upload dataset + scripts ---
    print("🔄 Uploading dataset and scripts to Rivanna...")
    upload_files_to_rivanna(
        {"remote_path": f"{remote_dir}/{remote_dataset_name}", "local_path": dataset_file_path},
        {"remote_path": f"{remote_dir}/train_gan_job.py", "local_path": train_script_path},
        {"remote_path": f"{remote_dir}/train_gan_job.slurm", "local_path": slurm_script_path},
    )

    # --- Submit job ---
    submission_output = submit_rivanna_job(f"{remote_dir}/train_gan_job.slurm")
    job_id = submission_output.strip().split()[-1]

    # --- Clean up local files ---
    # shutil.rmtree(base_dir, ignore_errors=True)

    return {
        "job_id": job_id or 1,
        "remote_dir": remote_dir or 1,
        "submission_output": submission_output or 1,
    }



@mcp.tool
def poll_training_job(job_id:int, poll_frequency:int):
    print("⏳ Polling for completion...")
    final_state = poll_rivanna_job(job_id)
    print(f"✅ Job finished with state: {final_state}")



    # #
    # train_image_gan(
    #     generator=generator,
    #     discriminator=discriminator,
    #     dataset=dataset,
    #     latent_dim=latent_dim,
    #     lr=learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size
    # )
    # print("✅ GAN training complete.")

    # # 6️⃣ Generate synthetic images
    # print(f"🔧 Step 6/7: Generating {num_images_to_generate} synthetic images...")
    # generator.eval()
    # generated_imgs = generate_images(generator, num_images_to_generate, latent_dim)
    # generated_imgs = (generated_imgs * 0.5 + 0.5).clamp(0, 1)
    # print("✅ Image generation complete.")

    # # 7️⃣ Save and zip output
    # print("🔧 Step 7/7: Saving generated images and creating ZIP...")
    # output_root = os.path.join(base_dir, "generated_images")
    # os.makedirs(output_root, exist_ok=True)
    # for class_name in class_names:
    #     os.makedirs(os.path.join(output_root, class_name), exist_ok=True)

    # for i, img_tensor in enumerate(generated_imgs):
    #     cls = class_names[i % len(class_names)]
    #     out_path = os.path.join(output_root, cls, f"gen_{i:04d}.png")
    #     vutils.save_image(img_tensor, out_path)

    # output_zip_path = os.path.join(base_dir, "generated_images.zip")
    # with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    #     for root, _, files in os.walk(output_root):
    #         for file in files:
    #             abs_path = os.path.join(root, file)
    #             rel_path = os.path.relpath(abs_path, start=output_root)
    #             zipf.write(abs_path, arcname=rel_path)
    # print(f"✅ Output ZIP created at: {output_zip_path}")

    # print("🎉 [DONE] train_and_run_image_gan completed successfully.")
    # return {"generated_zip_path": output_zip_path}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
