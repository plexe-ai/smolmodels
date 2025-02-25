from fastapi import APIRouter, HTTPException
from uuid import UUID

# Use relative imports
from ..models.requests import PredictionRequest
from ..models.responses import PredictionResponse
from ..services.metadata import MetadataService
from ..services.prediction import PredictionService

# Create router without a trailing slash redirect
router = APIRouter(redirect_slashes=False)


@router.post("/models/{model_id}", response_model=PredictionResponse)
async def predict(model_id: UUID, request: PredictionRequest):
    """Get prediction from a model"""
    model_info = MetadataService.get_model(str(model_id))

    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    if model_info["status"] != "ready":
        raise HTTPException(status_code=400, detail=f"Model {model_id} is not ready (status: {model_info['status']})")

    try:
        # Get prediction from model
        result = PredictionService.predict(str(model_id), request.data)
        return {"model_id": str(model_id), "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
