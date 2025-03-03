from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime


class JobResponse(BaseModel):
    """Response for job creation"""

    model_id: str = Field(..., description="ID of the model")
    job_id: str = Field(..., description="ID of the job")
    status: str = Field(..., description="Status of the job")


class JobStatusResponse(BaseModel):
    """Response for job status check"""

    job_id: str = Field(..., description="ID of the job")
    status: str = Field(..., description="Status of the job")
    created_at: datetime = Field(..., description="Creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    model_id: Optional[str] = Field(None, description="ID of the model (if applicable)")
    error: Optional[str] = Field(None, description="Error message (if failed)")


class ModelResponse(BaseModel):
    """Model metadata response"""

    model_id: str = Field(..., description="ID of the model")
    status: str = Field(..., description="Status of the model")
    intent: str = Field(..., description="Natural language description of the model's purpose")
    input_schema: Optional[Dict[str, str]] = Field(None, description="Input schema specification")
    output_schema: Optional[Dict[str, str]] = Field(None, description="Output schema specification")
    created_at: datetime = Field(..., description="Creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Model performance metrics")


class PredictionResponse(BaseModel):
    """Prediction response"""

    model_id: str = Field(..., description="ID of the model")
    prediction: Dict[str, Any] = Field(..., description="Prediction result")
