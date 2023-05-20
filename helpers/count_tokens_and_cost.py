# examples: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

import tiktoken
import sys
sys.path.append('..')  # Add the parent directory to the Python path

from constants import get_models  # Import the function from models.py

MODELS = get_models()

def get_cost_per_token(model_name: str) -> float:
    """
    Get the cost per token for the specified model name.

    Parameters:
        model_name (str): The name of the model.
    
    Returns:    
        float: The cost per token for the specified model.
    """
    if model_name in MODELS:
        return MODELS[model_name]["cost_per_1k_tokens"] / 1000
    else:
        raise ValueError("Invalid model name. Must be one of ['gpt-4', 'gpt-3.5-turbo', 'text-embedding-ada-002', 'text-davinci-002', 'text-davinci-003']")


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def count_tokens_and_cost(model_name: str, inputs: list[str]) -> dict:
    """
    Count the number of tokens in the provided inputs using the specified OpenAI model and calculate the cost.

    Parameters:
        model_name (str): The name of the model to use for token counting.
        inputs (list[str]): The text inputs for which to count the tokens. Must be a list of strings.

    Returns:
        dict: A dictionary containing the 'token_count', 'cost', and 'currency' for the job.
    """

    tokens = 0

    # get num of tokens
    if isinstance(inputs, list):
        tokens = sum([num_tokens_from_string(input, model_name) for input in inputs])
    else:
        raise ValueError("Invalid input type. Must be a list of strings.")

    # get cost per token
    cost_per_token = get_cost_per_token(model_name)
    
    # calculate cost
    cost = tokens * cost_per_token
    
    return {
        "token_count": tokens,
        "cost": cost,
        "currency": "USD",
    }
