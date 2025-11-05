import os, time
import paramiko
from paramiko import SSHClient, AutoAddPolicy
from dotenv import load_dotenv

load_dotenv()  # 👈 ensures RIVANNA_USER and RIVANNA_KEY_PATH are loaded

def submit_rivanna_job(job_script_path: str):
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(
        hostname="login.hpc.virginia.edu",
        username=os.getenv("RIVANNA_USER"),
        key_filename=os.getenv("RIVANNA_KEY_PATH"),
    )
    stdin, stdout, stderr = ssh.exec_command(f"sbatch {job_script_path}")
    output = stdout.read().decode().strip()
    ssh.close()
    return output

def poll_rivanna_job(job_id: str, poll_frequency:int=30):
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(
        hostname="login.hpc.virginia.edu",
        username=os.getenv("RIVANNA_USER"),
        key_filename=os.getenv("RIVANNA_KEY_PATH"),
    )
    while True:
        stdin, stdout, stderr = ssh.exec_command(f"sacct -j {job_id} --format=State --noheader")
        state = stdout.read().decode().strip()
        print(state)
        if any(s in state for s in ("COMPLETED", "FAILED", "CANCELLED")):
            break
        time.sleep(poll_frequency)
    ssh.close()
    return state

def upload_files_to_rivanna(*files):
    import os, paramiko

    user = os.getenv("RIVANNA_USER") or "ntq4hf"
    key_path = os.path.expanduser(os.getenv("RIVANNA_KEY_PATH") or "rivanna_info/rivanna_keys")
    base_remote_dir = f"/home/{user}/scratch"   # ✅ Correct root directory

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname="login.hpc.virginia.edu", username=user, key_filename=key_path)
    sftp = ssh.open_sftp()
    print(f"✅ Connected to Rivanna as {user}")

    def ensure_remote_dir(path: str):
        parts = path.strip("/").split("/")
        current = ""
        for p in parts:
            current += "/" + p
            try:
                sftp.stat(current)
            except FileNotFoundError:
                try:
                    sftp.mkdir(current)
                    print(f"📂 Created {current}")
                except OSError:
                    pass

    for f in files:
        remote_path = f.get("remote_path")
        if not remote_path:
            raise ValueError("Each file must include 'remote_path'")

        # Expand and normalize so everything lives under /home/<user>/scratch/
        if remote_path.startswith("~"):
            remote_path = remote_path.replace("~", f"/home/{user}/scratch")
        elif not remote_path.startswith("/home/"):
            remote_path = f"{base_remote_dir}/{remote_path}"

        remote_path = os.path.normpath(remote_path)
        remote_dir = os.path.dirname(remote_path)
        ensure_remote_dir(remote_dir)

        print(f"⬆️  Uploading to {remote_path} ...")

        if f.get("content") is not None:
            data = f["content"]
            if isinstance(data, str):
                data = data.encode("utf-8")
            with sftp.file(remote_path, "wb") as remote_f:
                remote_f.write(data)
        elif f.get("local_path"):
            local_path = os.path.expanduser(f["local_path"])
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            sftp.put(local_path, remote_path)
        else:
            raise ValueError("Each file must have either 'content' or 'local_path'")

        print(f"✅ Uploaded {remote_path}")

    sftp.close()
    ssh.close()
    print("🔒 Connection closed.")
