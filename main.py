import openai 
from helpers.count_tokens_and_cost import count_tokens_and_cost
from dotenv import dotenv_values
import pandas as pd 
from create_embeddings import create_embeddings

config = dotenv_values(".env")

# configure openai
openai.api_key = config["OPENAI_API_KEY"]

MAX_COST = 0.5
MODEL = "text-embedding-ada-002"
EMBEDDING_PATH = "embeddings.pkl"

content_df = pd.read_csv('rrs-blog-content.csv')

# MESSAGES = list of values from content_df['content']
# MESSAGES = content_df['Content'].values.tolist()


# messages = [
#     {"role": "user", "content": f"write a hello world python function for me"}
# ]

# # list of content from messages
# message_content = [message["content"] for message in messages] 

MESSAGES = [
    "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".",
    "What is human life expectancy in the United States?",
    "Who was president of the United States in 1955?"
]


def run_openai(message_list, model="gpt-3.5-turbo", max_cost=None, embedding_cache_path=None):

    # get tokens and cost 
    tokens_and_cost = count_tokens_and_cost(
        model_name=model,
        inputs=message_list
    )
    
    print('tokens_and_cost', tokens_and_cost)
    
    if max_cost and tokens_and_cost["cost"] > max_cost:
        raise ValueError(f"Cost of {tokens_and_cost['cost']} exceeds max cost of {max_cost}")

    if model == "gpt-3.5-turbo":
        # res = openai.ChatCompletion.create(
        #     messages=messages,
        #     model=model
        # )
        
        # print(res)
        # print(res["choices"][0]["message"]["content"])
        pass 
    
    elif model == "text-embedding-ada-002":
        create_embeddings(message_list, embedding_cache_path)

# tokens_and_cost = count_tokens_and_cost(

if __name__ == "__main__":
    run_openai(MESSAGES, MODEL, MAX_COST)