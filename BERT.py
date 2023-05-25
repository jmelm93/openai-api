from sentence_transformers import SentenceTransformer # https://pypi.org/project/sentence-transformers/
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dotenv import dotenv_values
import psycopg2
import numpy as np
import pickle

config = dotenv_values(".env")

# Config
MODEL = 'all-mpnet-base-v2' # https://www.sbert.net/docs/pretrained_models.html
QUERY_URL = '/example-1'
CSV_FILE_NAME = config["CONTENT_CSV_PATH"]
STORAGE_TYPE = 'pickle' # 'pickle' or 'postgres'

# PostgreSQL credentials
HOST = "localhost"
DATABASE = "postgres"
USER = "postgres"
PASSWORD = "postgres"

# list of objects with keys "url", "text", "exclude", "weight"
data = pd.read_csv(f"./data/{CSV_FILE_NAME}.csv")
data['index'] = data.index
list_of_objects = data.to_dict("records")

# Create a connection to the PostgreSQL database
conn = psycopg2.connect(
    host=HOST,
    database=DATABASE,
    user=USER,
    password=PASSWORD
)

c = conn.cursor()

# Create the table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS data (
        url TEXT PRIMARY KEY,
        text TEXT,
        text_embedding REAL[],
        exclude BOOLEAN,
        weight REAL
    )
''')

# Create the BERT model
model = SentenceTransformer(MODEL)


def create_embedding_and_store_if_needed(obj, storage_type):
    if storage_type == "pickle":
        # Check if the embedding already exists for the text string
        try:
            with open('data/data.pickle', 'rb') as f:
                data = pickle.load(f)
            result = next((item for item in data if item['text'] == obj['text']), None)
            if result is not None:
                # Use existing embedding
                obj['text_embedding'] = np.array(result['text_embedding'])
                print(f'Item #{obj["index"] + 1}: Using existing embedding for {obj["url"]}')
                return obj
            else:
                # Create new embedding and store to database
                obj['text_embedding'] = model.encode(obj['text']).tolist()
                data.append(obj)
                with open('data/data.pickle', 'wb') as f:
                    pickle.dump(data, f)
                print(f'Item #{obj["index"] + 1}: Created new embedding for {obj["url"]}')
                return obj
        except FileNotFoundError:
            # Create new embedding and store to database
            obj['text_embedding'] = model.encode(obj['text']).tolist()
            data = [obj]
            with open('data/data.pickle', 'wb') as f:
                pickle.dump(data, f)
            print(f'Item #{obj["index"] + 1}: Created new file at data/data.pickle for {obj["url"]}')
            return obj
        
    elif storage_type == "postgres":
        with conn.cursor() as c:
            # Check if the embedding already exists for the text string
            c.execute('SELECT text_embedding FROM data WHERE text = %s', (obj['text'],))
            result = c.fetchone()
            if result is not None:
                # Use existing embedding
                obj['text_embedding'] = np.array(result[0])
                return obj
            else:
                # Create new embedding and store to database
                obj['text_embedding'] = model.encode(obj['text']).tolist()
                c.execute('''
                    INSERT INTO data (url, text, text_embedding, exclude, weight) 
                    VALUES (%s, %s, %s, %s, %s)
                ''', (obj['url'], obj['text'], obj['text_embedding'], obj['exclude'], obj['weight']))
                conn.commit()
                return obj
    else:
        raise Exception("storage_type must be 'pickle' or 'postgres'")


def get_related_pages(query_url, n, list_of_objects, storage_type):
    # create embedding for items in list_of_objects that don't have one in the database
    list_of_objects = [create_embedding_and_store_if_needed(obj, storage_type) for obj in list_of_objects]
    
    # find the query object within the list_of_objects
    query_obj = next((obj for obj in list_of_objects if obj['url'] == query_url), None)
    if query_obj is None:
        return None

    # Calculate cosine similarity between query object and all other objects
    similarity_scores = []
    for obj in list_of_objects:
        if obj['url'] != query_url and not obj['exclude']:
            similarity = cosine_similarity([query_obj['text_embedding']], [obj['text_embedding']])[0][0]
            weighted_similarity = similarity * obj['weight']
            similarity_scores.append((obj['url'], query_obj['url'], obj['text'], weighted_similarity))

    # Sort the objects by the weighted similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[3], reverse=True)

    # Return the top 'n' URLs
    top_n = similarity_scores[:n]
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame(top_n, columns=["match_url", "query_url", "text", "similarity"])
    
    return df


if __name__ == '__main__':
    related_pages = get_related_pages(QUERY_URL, 2, list_of_objects, STORAGE_TYPE)
    print(related_pages)