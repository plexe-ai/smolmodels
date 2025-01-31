# internal/data_generation/generator.py
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

from smolmodels.internal.common.providers.provider import Provider


@dataclass
class DataGenerationRequest:
    """Data needed for generation"""

    intent: str
    n_samples: int
    augment_existing: bool = False
    quality_threshold: float = 0.8
    existing_data: Optional[pd.DataFrame] = None
    input_schema: Optional[Dict[str, str]] = None
    output_schema: Optional[Dict[str, str]] = None


def generate_data(provider: Provider, request: DataGenerationRequest) -> pd.DataFrame:
    """Generate synthetic data based on request parameters"""
    from .core.generation.combined import CombinedDataGenerator
    from .config import Config

    generator = CombinedDataGenerator(provider, Config())

    # Convert schemas to generator format
    schema = {
        "column_names": [*request.input_schema.keys(), *request.output_schema.keys()],
        "column_types": [str(t) for t in [*request.input_schema.values(), *request.output_schema.values()]],
        "column_descriptors": [""] * len({**request.input_schema, **request.output_schema}),
        "column_nullable": [False] * len({**request.input_schema, **request.output_schema}),
    }

    output_path = generator.generate(
        problem_description=request.intent,
        n_records_to_generate=request.n_samples,
        schema=schema,
        sample_data_path=request.existing_data.to_csv(index=False) if request.existing_data is not None else None,
    )

    return pd.read_csv(output_path)
