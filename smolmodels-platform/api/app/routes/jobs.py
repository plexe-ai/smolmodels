from fastapi import APIRouter, HTTPException
from uuid import UUID

# Use relative imports
from ..models.responses import JobStatusResponse
from ..services.metadata import MetadataService

# Create router without a trailing slash redirect
router = APIRouter(redirect_slashes=False)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: UUID):
    """Get status of a job"""
    job_info = MetadataService.get_job(str(job_id))
    if not job_info:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job_info
