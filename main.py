import openai 
from helpers.count_tokens_and_cost import count_tokens_and_cost
from dotenv import dotenv_values
import pandas as pd 
# from handle_embeddings_v2 import create_embeddings
from handle_embeddings import create_embeddings, visualize_embeddings, get_recommendations

config = dotenv_values(".env")

# HARD CODED VALUES
MAX_COST = 0.5
MODEL = "text-embedding-ada-002"
CSV_FILE_NAME = config["CONTENT_CSV_PATH"]
EMBEDDING_PATH = f"embedding_cache/{CSV_FILE_NAME}.pkl"
# EMBEDDING_PATH = f"embedding_cache/CSV_FILE_NAME{CSV_FILE_NAME}.json"
CONTENT_COL_NAME = "content"
CONTEXT_COL_NAME = "criteria"

# get message list
csv = pd.read_csv(f"./data/{CSV_FILE_NAME}.csv")

# MESSAGES = csv.to_dict("records")
MESSAGES = csv[CONTENT_COL_NAME].values.tolist()
CRITERIA = csv[[CONTEXT_COL_NAME]].to_dict("records")

# configure openai
openai.api_key = config["OPENAI_API_KEY"]


def run_openai(
    message_list, 
    model="gpt-3.5-turbo", 
    max_cost=None, 
    embedding_cache_path=None,
    criteria_for_embeddings=None
):

    # get tokens and cost 
    tokens_and_cost = count_tokens_and_cost(
        model_name=model,
        inputs=message_list
    )
    
    print('tokens_and_cost', tokens_and_cost)
    
    if max_cost and tokens_and_cost["cost"] > max_cost:
        raise ValueError(f"Cost of {tokens_and_cost['cost']} exceeds max cost of {max_cost}")

    if model == "gpt-3.5-turbo":
        # res = openai.ChatCompletion.create( messages=messages, model=model )
        # print(res)
        pass 
    
    elif model == "text-embedding-ada-002":
        # embeddings = create_embeddings(message_list, embedding_cache_path)
        # visual_context = visualize_embeddings(embeddings, criteria_for_embedding_visualization)
        # recommendations = get_recommendations(
        #     list_of_strings=message_list,
        #     embedding_cache_path=embedding_cache_path,
        #     index_of_source_string=3,
        #     criteria_for_embeddings=criteria_for_embeddings
        # )
        
        # get length of list_of_strings and loop through each updating the "index_of_source_string" 
        # to get recommendations for each string
        list_of_recommendation_dfs = [
            get_recommendations(
                list_of_strings=message_list,
                embedding_cache_path=embedding_cache_path,
                index_of_source_string=i,
                criteria_for_embeddings=criteria_for_embeddings
            )
            # for i in range(len(message_list))
            # run for first 50 messages
            for i in range(50)
        ]
        
        recommendations = pd.concat(list_of_recommendation_dfs)
        
        recommendations.to_csv(f"./data/{CSV_FILE_NAME}_recommendations.csv", index=False)

if __name__ == "__main__":
    run_openai(
        message_list=MESSAGES, 
        model=MODEL, 
        max_cost=MAX_COST,
        embedding_cache_path=EMBEDDING_PATH,
        criteria_for_embeddings=CRITERIA
    )