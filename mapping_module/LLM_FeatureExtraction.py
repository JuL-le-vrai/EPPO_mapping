from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
import torch
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import xmltodict
import os
import argparse
import json
from types import SimpleNamespace

cfg = SimpleNamespace(**{})
cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def list_of_strings(arg):
    return arg.split(',')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def produce_sentence(name, prompt_type):

    if prompt_type == "expert_prompt":
        sentence = f'As an expert phytopathologist, with comprehensive knowledge on pests and diseases of plants, and particularly crops, tell me how to differentiate {name} that affect my plant from other similar damages.'
    elif prompt_type == "expert_prompt2":
        sentence = f'As a expert phytopathologist, calculate an embedding of the following plant disease name, avoiding to make a different embedding if two names are very very similar and vary by only 2 or 3 characters : {name}'
    elif prompt_type == "expert_prompt3":
        sentence = f'''I am working with a list of plant pathogen and damage names to identify possible synonyms based on their underlying phytopathological features.
                    For each name, I need to extract a precise, domain-specific representation that reflects the biological and symptomatic relationships between different pathogens and types of plant damage. 
                    Many names may appear distinct but represent similar concepts in terms of plant disease and damage mechanisms.
                    I will provide you either French common names, English common names or scientific names.

                    Please provide a detailed embedding for the following plant pathogen or damage name that captures:

                    The biological cause (bacterial, viral, fungal, parasitic, etc.)
                    The specific symptoms or damages caused to the plant (wilting, leaf spots, cankers, etc.)
                    The plant species or group commonly affected
                    Any common synonym relationships that might exist within the field of phytopathology
                    Any common French/English traduction that might exist within the field of phytopathology
                    Contextual relevance to common agricultural practices or biological processes
                    Possible misspellings of the names

                    The output embedding should prioritize the underlying scientific relationships rather than surface-level or language-level differences in names. 
                    Here is the first name for extraction: {name}'''
    else:
        sentence = f'This plant is affected by {name}'

    return sentence

def extract_features(model_name, sentence, hf_model = None, hf_tokenizer = None, prompt_type = '', gpt_api_key = None):

    if model_name == 'GPT':
        
        client = OpenAI(api_key=gpt_api_key)
        response = client.embeddings.create(
            input=sentence,
            model="text-embedding-3-large"
        )
        embdg = response.data[0].embedding

    else:
        batch = hf_tokenizer(sentence, return_tensors="pt", padding='max_length', max_length=padding_len, truncation=True, add_special_tokens=True)
        batch.to(cfg.device)
        with torch.no_grad():
            response = hf_model(**batch).last_hidden_state.detach().cpu().mean(axis=1)
            embdg = response
    return embdg




if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-name', type = str, default='')
    parser.add_argument('--GPT-api-key-file', type = str, default='../GPT_API_key.txt')
    parser.add_argument('--names-json', type=list_of_strings, default = ['IPM_names.json', 'PN_names.json', 'Bousset_names.json', 'Avelino_names.json', 'PV_names.json'])
    parser.add_argument('--EPPO-json', type=str, default = 'EPPO_codes.json')
    parser.add_argument('--model', default = "GPT", choices = ['Xianjun/PLLaMa-7b-base', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised', 'GPT'])

    args = parser.parse_args()

    with open(args.GPT_api_key_file, 'r') as f:
        gpt_api_key = f.read().strip()

    # listing the datasets to map, considering one json file per dataset
    ds_list = [i.replace('.json', '') for i in args.names_json]

    # Loading the dictionnary of possible names for the datasets to map.
    dic_possible_names = {}
    for json_file, ds in zip(args.names_json, ds_list):
        with open(json_file,'r') as f:
            dic_possible_names[ds] = json.load(f)


    # Loading the EPPO codes and infos from the json obtained with xml_to_json.py
    with open(args.EPPO_json,'r') as f:
        dict_codes = json.load(f)
 
    # languages considered for the mapping
    considered_lang = ['la', 'en', 'fr']

    dic_possible_names['EPPO'] = {}

    # standardizing the possible names for EPPO codes
    for code in dict_codes:
        dic_possible_names['EPPO'][code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','').replace('\'','').replace("'", " ") for name in dict_codes[code]['names'] if name['lang'] in considered_lang and name['full_name'] != None and name['active'] == 'true']

    # Feature extraction method depends on the model considered.
    # For GPT, we directly use the OPenAI API.
    # For other models, we use the HuggingFace library to extract features in batches.
    if args.model == 'GPT':

        if args.prompt_name != '':
            name_exp = f'_{args.prompt_name}'
        else:
            name_exp = ''

        Path(f'GPT_features{name_exp}').mkdir(exist_ok=True)


        for ds in dic_possible_names:
            Path(f'GPT_features{name_exp}/{ds}').mkdir(exist_ok=True)
            for subid in tqdm(dic_possible_names[ds]):
                Path(f'GPT_features{name_exp}/{ds}/{subid}').mkdir(exist_ok=True)

                for name in dic_possible_names[ds][subid]:
                    name = name.lower().replace('_', ' ').replace('.','')
                    if name != '' and not os.path.isfile(f'GPT_features{name_exp}/{ds}/{subid}/{name}.pt'):

                        sentence = produce_sentence(name, prompt_type = args.prompt_name)
                        embdg = extract_features('GPT', sentence, prompt_type = args.prompt_name, gpt_api_key = gpt_api_key)
                        name = name.replace('/','')
                        # Computed embeddings are saved to avoid recomputation at each run.                
                        torch.save(embdg,f"GPT_features{name_exp}/{ds}/{subid}/{name}.pt")
    else:

        raw_inputs = []

        for ds in ds_list:

            for k in dic_possible_names[ds]:
                raw_inputs += dic_possible_names[ds][k]


        checkpoint = args.model
        model_name = args.model.split('/')[-1]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, legacy = False, token = "../../Desktop/Test_Directory/dammagenet/Hf_token")
        model = AutoModel.from_pretrained(checkpoint, token = "../../Desktop/Test_Directory/dammagenet/Hf_token")
        model.to(cfg.device)


        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        padding_len = 50

        Path(f'{model_name}_features').mkdir(exist_ok=True)

        total_raw_batches_used = []

        for ds in dic_possible_names.keys():

            names_stacked = []
            features_stacked = []
            codes_stacked =[]
            for subid in tqdm(dic_possible_names[ds]):
                if len(dic_possible_names[ds][subid]) > 0:
                    Path(f'{model_name}_features/{ds}/{subid}').mkdir(exist_ok=True, parents=True)
                    raw_batch = dic_possible_names[ds][subid]
                    for name in raw_batch:
                        name = name.lower().replace('_', ' ').replace('.','')
                        if name != '' and not os.path.isfile(f'{model_name}_features/{ds}/{subid}/{name}.pt'):
                            sentence = produce_sentence(name, prompt_type = args.prompt_name)
                            embdg = extract_features(model_name, sentence, hf_model = model, hf_tokenizer = tokenizer, prompt_type = args.prompt_name)
                            name = name.replace('/','')
                            # Computed embeddings are saved to avoid recomputation at each run.                
                            torch.save(embdg,f'{model_name}_features/{ds}/{subid}/{name}.pt')



