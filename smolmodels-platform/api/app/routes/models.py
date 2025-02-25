from fastapi import APIRouter, HTTPException
from uuid import UUID
import pandas as pd

# Use relative imports since we're within the same package
from ..models.requests import ModelCreateRequest
from ..models.responses import ModelResponse, JobResponse
from ..services.metadata import MetadataService
from ..services.queue import QueueService

# Create router without a trailing slash redirect
router = APIRouter(redirect_slashes=False)


@router.post("", response_model=JobResponse)
async def create_model(request: ModelCreateRequest):
    """Create a new ML model"""
    try:
        # Create model entry in metadata store
        model_id = MetadataService.create_model_entry(request)

        # Convert example data to serializable format if provided
        datasets = None
        if request.example_data:
            # Serialize DataFrame to JSON
            df = pd.DataFrame(request.example_data)
            datasets = {"example_data": df.to_dict(orient="records")}

        # Create job to build the model
        job_id = QueueService.enqueue_job(
            "MODEL_CREATE",
            {
                "model_id": model_id,
                "intent": request.intent,
                "input_schema": request.input_schema,
                "output_schema": request.output_schema,
                "datasets": datasets,
                "timeout": request.timeout,
                "max_iterations": request.max_iterations,
            },
        )

        return {"model_id": model_id, "job_id": job_id, "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create model: {str(e)}")


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: UUID):
    """Get model metadata by ID"""
    model_info = MetadataService.get_model(str(model_id))
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model_info


@router.get("", response_model=list[ModelResponse])
async def list_models():
    """List all models"""
    return MetadataService.list_models()
