

def get_models():
    return {
        "gpt-4": { "cost_per_1k_tokens": 0.12 },
        "gpt-3.5-turbo": { "cost_per_1k_tokens": 0.002 },
        "text-embedding-ada-002": { "cost_per_1k_tokens": 0.002 },
        "text-davinci-002": { "cost_per_1k_tokens": 0.002 },
        "text-davinci-003": { "cost_per_1k_tokens": 0.02 },
        "text-curie-001": { "cost_per_1k_tokens": 0.002 },
        "text-babbage-001": { "cost_per_1k_tokens": 0.005 },
        "text-ada-001": { "cost_per_1k_tokens": 0.0004 },
        "text-embedding-ada-002": { "cost_per_1k_tokens": 0.0004 },
    }