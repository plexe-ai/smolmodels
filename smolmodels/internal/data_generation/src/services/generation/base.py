from abc import ABC, abstractmethod


class BaseDataGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        problem_description: str,
        n_records_to_generate: int,
        output_path: str = None,
        schema: dict = None,
        sample_data_path: str = None,
    ) -> str:
        """
        Generate synthetic data for a given problem description.
        :param problem_description: natural language description of the problem
        :param n_records_to_generate: number of records to generate
        :param output_path: if provided, specifies the path to save the generated data
        :param schema: if provided, specifies the schema of the data to generate
        :param sample_data_path: if provided, specifies the path to a sample of existing data for the problem
        :return: path to the generated data
        """
        pass
