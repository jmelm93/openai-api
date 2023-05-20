import json
import openai
from dotenv import dotenv_values
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np

config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")  # replace newlines, which can negatively affect performance.
    return openai.Embedding.create(input=text, model=model)["data"][0]["embedding"]


def load_embedding_cache(embedding_cache_path):
    try:
        with open(embedding_cache_path, "r") as embedding_cache_file:
            embedding_cache = json.load(embedding_cache_file)
    except (FileNotFoundError, json.JSONDecodeError):
        embedding_cache = {}
    return embedding_cache


def save_embedding_cache(embedding_cache_path, embedding_cache):
    with open(embedding_cache_path, "w") as embedding_cache_file:
        json.dump(embedding_cache, embedding_cache_file)


def embedding_from_object(obj, embedding_cache_path, model="text-embedding-ada-002"):
    embedding_cache = load_embedding_cache(embedding_cache_path)
    
    # get all 'string' key values from the embedding cache
    
    string = obj["string"]
    context = obj.get("context", None)  # Get additional context if available
    
    # if (string, model) not in embedding_cache:
    #     embedding = get_embedding(string, model)
    #     print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20] if len(string) > 20 else string}")
    #     embedding_cache[(string, model)] = {"embedding": embedding, "context": context}
    #     save_embedding_cache(embedding_cache_path, embedding_cache)
    
    embedding = get_embedding(string, model)
    print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20] if len(string) > 20 else string}")
    embedding_cache[(string, model)] = {"embedding": embedding, "context": context}
    
    return embedding_cache[(string, model)]


def create_embeddings(list_of_objects, embedding_cache_path, model="text-embedding-ada-002"):
    embeddings = []
    for obj in list_of_objects:
        try:
            embedding = embedding_from_object(obj, embedding_cache_path, model)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing object: {obj['context']}. Error: {e}")
    
    if embeddings:
        return embeddings
    else:
        raise ValueError("No embeddings were created.")
