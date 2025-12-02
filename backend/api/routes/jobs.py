# backend/api/routes/jobs.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, Literal, List
import uuid

from api.services import job_service, agent_service
from api.config import UPLOAD_DIR

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

class Message(BaseModel):
    role: Literal["user", "agent"]
    content: str
    timestamp: str

class JobResponse(BaseModel):
    job_id: str
    status: str
    messages: List[Message]
    slurm_job_id: Optional[str] = None

class UserResponseRequest(BaseModel):
    message: str
    conversation_history: List[Message] = []

@router.post("", response_model=JobResponse)
async def initalize_agent(
    dataset: UploadFile = File(...),
    prompt: str = Form(...)
):
    """Create a new augmentation job and kick off agent"""
    if not dataset.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")
    
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    dataset_path = UPLOAD_DIR / f"{job_id}_{dataset.filename}"
    with open(dataset_path, 'wb') as f:
        content = await dataset.read()
        f.write(content)
    
    # Create job
    job_service.create_job(job_id, str(dataset_path), prompt)
    job_service.add_message(job_id, "user", prompt)
    
    # Run agent SYNCHRONOUSLY
    try:
        await agent_service.run_agent_flow(
            job_id,
            str(dataset_path),
            prompt,
            []
        )
        
        # Get updated job with agent's response
        job = job_service.get_job(job_id)
        
        return JobResponse(
            job_id=job_id,
            status=job['status'],
            messages=job['messages'],
            slurm_job_id=job.get('slurm_job_id'),
        )
        
    except Exception as e:
        job_service.update_job(job_id, status="failed")
        job_service.add_message(job_id, "agent", f"Error: {str(e)}")
        
        job = job_service.get_job(job_id)
        return JobResponse(
            job_id=job_id,
            status=job['status'],
            messages=job['messages']
        )


@router.post("/{job_id}/respond")
async def respond_to_agent(
    job_id: str,
    request: UserResponseRequest
):
    """Submit user response and continue agent flow"""
    job = job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] not in ['awaiting_input', 'queued', 'analyzing']:
        raise HTTPException(status_code=400, detail=f"Job is not awaiting input (status: {job['status']})")
    
    # Add user's message
    job_service.add_message(job_id, "user", request.message)
    job_service.update_job(job_id, status="analyzing")
    
    # Run agent SYNCHRONOUSLY (removed background_tasks)
    try:
        await agent_service.run_agent_flow(
            job_id,
            job['dataset_path'],
            job['prompt'],
            request.conversation_history
        )
        
        # Get updated job with agent's response
        job = job_service.get_job(job_id)
        
        return JobResponse(
            job_id=job_id,
            status=job['status'],
            messages=job['messages'],
            slurm_job_id=job.get('slurm_job_id'),
            result_url=job.get('result_url')
        )
        
    except Exception as e:
        job_service.update_job(job_id, status="failed")
        job_service.add_message(job_id, "agent", f"Error: {str(e)}")
        
        job = job_service.get_job(job_id)
        return JobResponse(
            job_id=job_id,
            status=job['status'],
            messages=job['messages']
        )

@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Poll for job status and new messages"""
    job = job_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    state = job_service.poll_rivanna_job(job['slurm_job_id'])

    #Change to real state
    if state:
        job_service.update_job(job_id, status=state)
    
    response = JobResponse(
        job_id=job['job_id'],
        status=job['status'],
        messages=job['messages'],
        slurm_job_id=job.get('slurm_job_id'),
    )

    if any(s in job_service.get_job(job_id)['status'] for s in ("completed", "failed", "cancelled")):
        job_service.delete_job(job_id)

    return response


@router.get("/{job_id}/result")
async def get_job_result(job_id: str):
    """Get job result URL if completed"""
    job = job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Job not completed or result not available")
    
    result_file = job_service.get_rivanna_job_results()

    if result_file is None:
        raise HTTPException(status_code=404, detail="Result file not found")
    
    # Return binary file directly with proper headers
    return Response(
        content=result_file,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=gan_output_{job_id}.zip"
        }
    )
    

@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Delete job and uploaded dataset"""
    job = job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete uploaded file
    import os
    if os.path.exists(job['dataset_path']):
        os.remove(job['dataset_path'])
    
    job_service.delete_job(job_id)
    
    return {"status": "deleted"}