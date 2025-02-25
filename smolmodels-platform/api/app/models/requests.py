from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class ModelCreateRequest(BaseModel):
    """Request to create a new model"""

    intent: str = Field(..., description="Natural language description of the model's purpose")
    input_schema: Optional[Dict[str, str]] = Field(None, description="Input schema specification")
    output_schema: Optional[Dict[str, str]] = Field(None, description="Output schema specification")
    timeout: Optional[int] = Field(3600, description="Maximum time in seconds for model building")
    max_iterations: Optional[int] = Field(3, description="Maximum iterations for model search")
    example_data: Optional[List[Dict[str, Any]]] = Field(None, description="Example data for training")


class PredictionRequest(BaseModel):
    """Request for model prediction"""

    data: Dict[str, Any] = Field(..., description="Input data for prediction")
