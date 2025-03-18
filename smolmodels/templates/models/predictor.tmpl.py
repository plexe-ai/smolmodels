from typing import List

from pydantic import BaseModel

from smolmodels.internal.models.entities.artifact import Artifact
from smolmodels.internal.models.interfaces.predictor import Predictor


class PredictorImplementation(Predictor):
    def __init__(self, artifacts: List[Artifact]):
        """
        Instantiates the predictor using the provided model artifacts.
        :param artifacts: list of BytesIO or StringIO artifacts
        """
        # TODO: add model loading code here; load from artifacts
        # Example:
        # self.model = load(next(
        #     artifact.path
        #     for artifact in artifacts
        #     if artifact.is_path() and artifact.name == "model"
        # ), None)
        # if self.model is None:
        #     raise ValueError("Model artifact not found")

    def predict(self, inputs: BaseModel) -> BaseModel:
        # TODO: add inference code here
        # Example: return self.model.predict(inputs)
        pass


# REFERENCES:
# The Artifact class is defined as follows:
#
# class Artifact:
#     name: str
#     path: Path | None
#     handle: BinaryIO | None
#     data: bytes | None
#
#     def is_path(self) -> bool:
#         """ True if the artifact is a file path."""
#         return self.path is not None
#
#     def is_handle(self) -> bool:
#         """
#         True if the artifact is file path or file-like object.
#         """
#         return self.handle is not None
#
#     def is_data(self) -> bool:
#         """
#         True if the artifact is a string or bytes object loaded in memory.
#         """
#         return self.data is not None
#
# The Artifact always has a 'name' attribute, which should be used to identify the artifact. If the artifact
# refers to a file, then "is_path" is true, and the path can be accessed via the "path" attribute. The same logic
# applies to the "handle" and "data" attributes.
