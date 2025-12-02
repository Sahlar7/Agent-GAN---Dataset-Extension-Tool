# backend/api/services/job_service.py
from typing import Optional, Dict
from datetime import datetime
import os, time
import paramiko
from paramiko import SSHClient, AutoAddPolicy
from dotenv import load_dotenv
import io
import zipfile

class JobService:
    """In-memory job storage (no persistence)"""
    
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
    
    def create_job(self, job_id: str, dataset_path: str, prompt: str) -> dict:
        """Create a new job record"""
        job_data = {
            "job_id": job_id,
            "status": "queued",
            "messages": [],  # Agent conversation history
            "dataset_path": dataset_path,
            "prompt": prompt,
            "slurm_job_id": None,
            "result_url": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self.jobs[job_id] = job_data
        return job_data
    
    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def update_job(self, job_id: str, **updates) -> None:
        """Update job fields"""
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)
            self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
    
    def add_message(self, job_id: str, role: str, content: str) -> None:
        """Add a message to the conversation"""
        if job_id in self.jobs:
            self.jobs[job_id]['messages'].append({
                "role": role,  # "user" or "agent"
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
    
    def delete_job(self, job_id: str) -> bool:
        """Delete job"""
        if job_id in self.jobs:
            del self.jobs[job_id]
            return True
        return False
    
    def poll_rivanna_job(self, job_id: str):
        """Poll Rivanna Job using paramink SSH"""
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        ssh.connect(
            hostname="login.hpc.virginia.edu",
            username=os.getenv("RIVANNA_USER"),
            key_filename=os.getenv("RIVANNA_KEY_PATH"),
        )
        stdin, stdout, stderr = ssh.exec_command(f"sacct -j {job_id} --format=State --noheader")
        state = stdout.read().decode().strip()
        print(state)
        print(stderr.read().decode().strip())
        ssh.close()
        print(state.lower())
        return state.lower()
    
    def get_rivanna_job_results(self, remote_path: str = "~/scratch/mcp_jobs"):
        """Retrieve gan_output.zip from Rivanna and return as bytes"""
        user = os.getenv("RIVANNA_USER")
        if remote_path.startswith("~"):
            remote_path = remote_path.replace("~", f"/home/{user}")

        ssh = SSHClient()
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        ssh.connect(
            hostname="login.hpc.virginia.edu",
            username=os.getenv("RIVANNA_USER"),
            key_filename=os.getenv("RIVANNA_KEY_PATH"),
        )
        sftp = ssh.open_sftp()

        remote_file = os.path.join(remote_path, "gan_output.zip")
        file_buffer = io.BytesIO()
        
        try:
            sftp.getfo(remote_file, file_buffer)
            file_buffer.seek(0)
            result = file_buffer.getvalue()
        except FileNotFoundError:
            print(f"Error: gan_output.zip not found at {remote_file}")
            result = None

        sftp.close()
        ssh.close()
        
        return result

job_service = JobService()