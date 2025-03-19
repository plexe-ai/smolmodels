from typing import List

# TODO: add any additional required imports here

from smolmodels.internal.models.entities.artifact import Artifact
from smolmodels.internal.models.interfaces.predictor import Predictor


class PredictorImplementation(Predictor):
    def __init__(self, artifacts: List[Artifact]):
        """
        Instantiates the predictor using the provided model artifacts.
        :param artifacts: list of BytesIO or StringIO artifacts
        """
        # TODO: add model loading code here; use _get_artifact helper to select the artifact by name
        # Example:
        # artifact = self._get_artifact("model", artifacts)
        # with artifact.get_as_handle() as binary_io:
        #     # Load the model from the handle
        #     # self.model = load_model(binary_io)

    def predict(self, inputs: dict) -> dict:
        # TODO: add inference code here
        # Example: return self.model.predict(inputs)
        pass

    @staticmethod
    def _get_artifact(name: str, artifacts: List[Artifact]) -> Artifact:
        for artifact in artifacts:
            if artifact.name == name:
                return artifact
        raise ValueError(f"Artifact {name} not found in the provided artifacts.")


# REFERENCES:
# The Artifact class has the following relevant methods:
#
# class Artifact:
#     name: str
#
#     def get_as_handle(self) -> BinaryIO:
#         """
#         Get the artifact as a file-like object.
#         """
#         ...
#
# The Artifact always has a 'name' attribute, which should be used to identify the artifact. The internal definition
# of the Artifact class is not relevant here, except for the 'get_as_handle' method, which returns a file-like BinaryIO
# object. This should be used to access the artifact's data.
