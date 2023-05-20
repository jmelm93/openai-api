import pickle
import openai
from dotenv import dotenv_values
from nomic import atlas
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
        with open(embedding_cache_path, "rb") as embedding_cache_file:
            embedding_cache = pickle.load(embedding_cache_file)
    except (FileNotFoundError, pickle.UnpicklingError):
        embedding_cache = {}
    return embedding_cache


def save_embedding_cache(embedding_cache_path, embedding_cache):
    with open(embedding_cache_path, "wb") as embedding_cache_file:
        pickle.dump(embedding_cache, embedding_cache_file)


def embedding_from_string(string, embedding_cache_path, model="text-embedding-ada-002"):
    embedding_cache = load_embedding_cache(embedding_cache_path)
    
    if (string, model) not in embedding_cache:
        embedding = get_embedding(string, model)
        print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20] if len(string) > 20 else string}")
        embedding_cache[(string, model)] = embedding
        save_embedding_cache(embedding_cache_path, embedding_cache)
    
    return embedding_cache[(string, model)]


def create_embeddings(list_of_strings, embedding_cache_path, model="text-embedding-ada-002"):
    embeddings = []
    for str_item in list_of_strings:
        try:
            embedding = embedding_from_string(str_item, embedding_cache_path, model)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing string: {str_item}. Error: {e}")
    
    if embeddings:
        return embeddings
    else:
        raise ValueError("No embeddings were created.")


def visualize_embeddings(embeddings, criteria):
    project = atlas.map_embeddings(
        embeddings=np.array(embeddings),
        data=criteria
    )
    return project

if __name__ == "__main__":
    # Test embedding creation
    create_embeddings(["I like cats", "I like dogs"], "embedding_cache/embeddings.pkl")