from agent.agent import run_agent
from api.services import job_service

class AgentService:
    """Service for orchestrating the agent workflow"""
    
    async def run_agent_flow(
        self,
        job_id: str,
        dataset_path: str,
        user_message: str,
        conversation_history: list = None
    ):
        """Kick off agent and handle response"""
        try:
            job_service.update_job(job_id, status="analyzing")
            
            # Run agent (it rebuilds messages internally from history)
            result = await run_agent(
                dataset_path=dataset_path,
                user_message=user_message,
                conversation_history=conversation_history or []
            )
            
            # Just append the new agent message - frontend already has conversation
            job_service.add_message(job_id, "agent", result["message"])
            
            # Update status based on action
            if result["action"] == "awaiting_input":
                job_service.update_job(job_id, status="awaiting_input")
                
            elif result["action"] == "job_submitted":
                job_service.update_job(
                    job_id,
                    status="submitted",
                    slurm_job_id=result["job_id"]
                )
                
            elif result["action"] == "error":
                job_service.update_job(job_id, status="failed")
            
        except Exception as e:
            job_service.update_job(job_id, status="failed")
            job_service.add_message(job_id, "agent", f"Error: {str(e)}")

agent_service = AgentService()