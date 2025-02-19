import os
import asyncio
import math

import pandas as pd

from .base import BaseDataGenerator
from smolmodels.internal.common.provider import Provider


class SimpleLLMDataGenerator(BaseDataGenerator):
    """
    Implementation of BaseDataGenerator that uses a straightforward LLM prompting mechanism to generate
    synthetic data. The generator relies on a single inference call to a pre-trained LLM model to generate samples.
    """

    def __init__(self, provider: Provider = None):
        self.llm = provider
        self.system_instruction = (
            "You are an expert in data science, data engineering, and any problem domain you encounter. "
            "You are speaking to someone who is, likewise, an expert in all these areas. "
            "Expectations for your performance are extremely high. Mediocrity is not acceptable. "
        )

    def generate(
        self,
        problem_description: str,
        n_records_to_generate: int,
        output_path: str = None,
        schema: dict = None,
        sample_data_path: str = None,
    ) -> str:
        # sample data is optional, as it may not always be available to the user
        if sample_data_path is not None:
            df_real = pd.read_csv(sample_data_path)
        else:
            df_real = None

        # basic problem specification
        base_prompt = (
            f"Give me a dataset of samples for the following ML problem:\n\n"
            f"PROBLEM DESCRIPTION:\n{problem_description}\n\n"
        )
        # the data schema is optional, as it may not always be provided
        if schema is not None:
            base_prompt += f"SCHEMA:\n{schema}\n\n"

        df_generated = pd.DataFrame(columns=df_real.columns if df_real is not None else schema["column_names"])

        # prepare prompts for all batches
        batch_size = 60
        num_batches = math.ceil(n_records_to_generate / batch_size)
        records_left = n_records_to_generate

        prompts = []
        for _ in range(num_batches):
            n_generate_this_iteration = min(records_left, batch_size)
            records_left -= n_generate_this_iteration

            # add sample data to the prompt if available
            sample_str = df_real.sample(5).to_string() if df_real is not None else ""
            prompt = (
                f"{base_prompt}"
                f"SAMPLE DATA:{sample_str}\n\n"
                f"Please give me samples that match the schema and are relevant to solving the problem. "
                f"The data should have an appropriate amount of variance and be representative of the problem. "
                f"The data should be distributed in a way that is consistent with the problem domain. "
                f"Make absolutely sure to give me EXACTLY {n_generate_this_iteration} records. "
                f"You must give me no fewer than and no more than {n_generate_this_iteration} records. "
                f"In your response, only include the dataset as a JSON string, no other text. "
                f"The output must be a raw JSON string with no formatting characters."
                f"Do not give me any code, any descriptions, any explanations, or any other text of any kind. "
                f"Only give me a raw JSON string with the data, and no other information whatsoever. "
            )
            prompts.append((prompt, n_generate_this_iteration))

        # generate data for a prompt
        async def generate_data(prompt):
            loop = asyncio.get_running_loop()
            try:
                return await loop.run_in_executor(None, self.llm.query, self.system_instruction, prompt)
            except Exception as err:
                print(f"Error during generation: {err}")
                return None  # Indicate failure

        # Function to run all tasks asynchronously
        async def run_tasks(p):
            tasks = [generate_data(prompt) for prompt, _ in p]
            return await asyncio.gather(*tasks)

        # generate results asynchronously retry failed batches
        pending_prompts = prompts.copy()
        while pending_prompts:
            print(f"Generating data for {len(pending_prompts)} batches...")
            responses = asyncio.run(run_tasks(pending_prompts))

            failed_prompts = []

            for response, (prompt, n_generate_this_iteration) in zip(responses, pending_prompts):
                if response is not None:
                    try:
                        response_text = response.text.replace("json", "").replace("`", "")
                        # convert the data to pd dataframe and append to the generated data
                        df_generated = pd.concat([df_generated, pd.read_json(str(response_text))], ignore_index=True)
                    except Exception as e:
                        print(f"Error processing data: {e}")
                        # Add the prompt back to failed_prompts for retry
                        failed_prompts.append((prompt, n_generate_this_iteration))
                else:
                    # If response is None, generation failed
                    failed_prompts.append((prompt, n_generate_this_iteration))

            # Update the pending_prompts with failed batches
            pending_prompts = failed_prompts

            if failed_prompts:
                print(f"Retrying {len(failed_prompts)} failed batches...")
            else:
                print("All batches processed successfully.")

        if output_path is None:
            output_path = "outputs/data.csv"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_generated.to_csv(output_path, index=False)

        # Parse the response and return the synthetic data
        return output_path
