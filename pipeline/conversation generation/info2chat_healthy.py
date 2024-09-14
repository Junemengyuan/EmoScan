from openai import OpenAI
from tqdm import tqdm
import os
import json
import pandas as pd

# get the OpenAI api

openai_key = "your api_key"
client = OpenAI(api_key=openai_key)

def response_return(system_role_,prompt_):
    completion = client.chat.completions.create(
    model="gpt-4",#
    messages=[
        {"role": "system", "content": system_role_},
        {"role": "user", "content": prompt_}
    ]
    )
    return completion.choices[0].message.content

# get the data

os.chdir('your path')
case_test_file_healthy = pd.read_excel("healthy_cases_v2.xlsx")

# transform the psychiatric profile to row conversations

def info2chat(case_description):
    system_role_ = "Your task is to generate a around 8-round psychiatrist-client conversation according to the case descriptions below. This conversation is used to screening emotional disorders for client based on dsm-5."
    prompt_ = f"""
    case descriptions: {case_description}
    
    The topic flow of the conversation should be:
    1.    get identification, determine the chief complaint, history of chief complaint
    2.    ask about the past psychiatric history, and history of drug or alcohol abuse
    3.    obtain medical history and family history
    4.    obtain personal and social history and trauma/abuse history
    
    You should follow the general rules below:
    1) The psychiatrist knows nothing about the client before the conversation and any fact of the client should be provided by the client
    2) Do not generate any information that is not in the case description
    3) Range of rounds of interaction: LESS THAN 8 ROUNDS!!!!
    4) If the subject is a child or young adolescent, the only client spearker should be the subject's parent in the conversation (only one client can talk).

    The output format should be:
    Client: xxx
    (newline)
    Psychiatrist: xxx
    (newline)
    The conversation starts by the client, ends with the psychiatrist's thanks for the client's cooperation, and let the client wait for the result (only result, without treatment). Do not add any other things.
    """
    try:
        conv = response_return(system_role_, prompt_)
    except:
        conv = "error"
        print("error!!")
    return conv

# read the data

os.chdir('your path')
case_test_file_healthy = pd.read_excel("healthy_cases_v2.xlsx")

# generate raw conversations

case_ls = []
for i in tqdm(range(0,len(case_test_file_anxdep['extracted_info']),1)):
    case_description = case_test_file_healthy['extracted_info'].iloc[i]
    conv = info2chat(case_description)
    case_ls.append(conv)

# save the data
save_healthy = pd.read_excel("healthy_cases_v2_save.xlsx")
save_healthy["raw_conversation"] = case_ls
save_healthy.to_excel("healthy_cases_v2_save.xlsx",index=False)
