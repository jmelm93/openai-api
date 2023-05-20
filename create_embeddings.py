import pandas as pd
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pickle
import openai

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ") # replace newlines, which can negatively affect performance.
    return openai.Embedding.create(input=text, model=model)["data"][0]["embedding"]

# define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(
    string,
    embedding_cache_path,
    model="text-embedding-ada-002"
):
    """Return embedding of given string, using a cache to avoid recomputing."""
    
    # establish a cache of embeddings to avoid recomputing
    # cache is a dict of tuples (text, model) -> embedding, saved as a pickle file

    # load the cache if it exists, and save a copy to disk
    try:
        embedding_cache = pd.read_pickle(embedding_cache_path)
    except FileNotFoundError:
        embedding_cache = {}
    with open(embedding_cache_path, "wb") as embedding_cache_file:
        pickle.dump(embedding_cache, embedding_cache_file)
    
    # if the embedding is not in the cache, request it from the API
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20]}")
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


def create_embeddings(list_of_strings, embedding_cache_path, model="text-embedding-ada-002"):
    print('list_of_strings', list_of_strings)
    
    # This line actaully generates the embeddings
    embeddings = [embedding_from_string(str_item, embedding_cache_path, model) for str_item in list_of_strings]
    
    # print 1 embedding for testing
    print(embeddings[0])
    