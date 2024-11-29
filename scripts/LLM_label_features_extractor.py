from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModel
import pandas as pd
import json
import re
from tqdm import tqdm
import torch
from types import SimpleNamespace
import xmltodict
from pathlib import Path
import argparse
import itertools
import os

def list_of_strings(arg):
    return arg.split(',')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds-list', type = list_of_strings)
    parser.add_argument('--model', default = "Xianjun/PLLaMa-7b-base", choices = ['Xianjun/PLLaMa-7b-base', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised'])
    args = parser.parse_args()

    cfg = SimpleNamespace(**{})
    # cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.device = 'cpu'

    dic_possible_names = {}

    # with open('../data/Extract_EPPO/Harmonization_outputs/DS_names/Bousset_names.json','r') as f:
    #     dic_possible_names['Bousset'] = json.load(f)
    # with open('../data/Extract_EPPO/Harmonization_outputs/DS_names/PV_names.json','r') as f:
    #     dic_possible_names['PV'] = json.load(f)
    # with open('../data/Extract_EPPO/Harmonization_outputs/DS_names/PN_names.json','r') as f:
    #     dic_possible_names['PN'] = json.load(f)
    # with open('../data/Extract_EPPO/Harmonization_outputs/DS_names/IPM_names.json','r') as f:
    #     dic_possible_names['IPM'] = json.load(f)


    for ds in args.ds_list:
        with open(f'../data/Extract_EPPO/Harmonization_outputs/DS_names/{ds}_names.json','r') as f:
            dic_possible_names[ds] = json.load(f)




    raw_inputs = []

    for ds in args.ds_list:

        for k in dic_possible_names[ds]:
            raw_inputs += dic_possible_names[ds][k]
        # for k in dic_possible_names['Bousset']:
        #     raw_inputs += dic_possible_names['Bousset'][k]
        # for k in dic_possible_names['PN']:
        #     raw_inputs += dic_possible_names['PN'][k]
        # for k in dic_possible_names['IPM']:
        #     raw_inputs += dic_possible_names['IPM'][k]



    checkpoint = args.model
    model_name = args.model.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, legacy = False, token = "../../Desktop/Test_Directory/dammagenet/Hf_token")
    model = AutoModel.from_pretrained(checkpoint, token = "../../Desktop/Test_Directory/dammagenet/Hf_token")
    model.to(cfg.device)


    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))


    # longest_string = max(raw_inputs, key=len)
    # padding_len = len(longest_string)

    padding_len = 50

    with open('../data/Extract_EPPO/fullcodes.xml','r') as f:
        xml = xmltodict.parse(f.read())

    dict_codes = {}

    for indiv in xml['codes']['code']:
        if indiv['@isactive'] == 'true' and indiv['@type'] not in ['PFL', 'SPT']: # Not considering neither inactive codes nor plant and plant taxonomic group codes
            indiv_id = indiv['@id']
            indiv_type = indiv['@type']
            indiv_code = indiv['eppocode']
            indiv_parents = []
            if indiv['parents'] is not None:
                for p in indiv['parents']:
                    if not type(indiv['parents'][p]) == list:
                        parents_list = [indiv['parents'][p]]
                    else:
                        parents_list = indiv['parents'][p]
                    for parent in parents_list:
                        indiv_parents.append(parent['#text'])

            indiv_names = []
            for name in indiv['names']:
                if not type(indiv['names'][name]) == list:
                    names_list = [indiv['names'][name]]
                else:
                    names_list = indiv['names'][name]
                for n in names_list:
    #                if n['lang'] in considered_languages:
                    indiv_names.append({'full_name' : n['fullname'], 'lang' : n['lang'], 'active' : n['@isactive']})

            dict_codes[indiv_code] = {'type' : indiv_type, 'parents' : indiv_parents, 'names' : indiv_names, 'mapsto' : {'IPM' : set(), 'PN' : set(), 'Bousset' : set(), 'Avelino' : set(), 'PV' : set()}}


    considered_lang = ['la', 'en', 'fr']

    dic_possible_names['EPPO'] = {}

    for code in dict_codes:
        dic_possible_names['EPPO'][code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','') for name in dict_codes[code]['names'] if name['lang'] in considered_lang and name['full_name'] != None and name['active'] == 'true']


    Path(f'../data/Extract_EPPO/{model_name}_features').mkdir(exist_ok=True)

    total_raw_batches_used = []

    # names_stacked = []
    # features_stacked = []
    # codes_stacked =[]    
    # for k in tqdm(dict_codes):
    #     # Path(f'../data/Extract_EPPO/{model_name}_features/EPPO').mkdir(exist_ok=True)
    #     raw_batch = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','') for name in dict_codes[k]['names'] if name['lang'] in considered_lang and name['full_name'] != None and name['active'] == 'true']
    #     raw_batch = [f'This plant is affected by {name}' for name in raw_batch]
    #     raw_codes = [k for i in raw_batch]
    #     total_raw_batches_used.append({'EPPO':raw_batch})
    #     batch = tokenizer(raw_batch, return_tensors="pt", padding='max_length', max_length=padding_len, truncation=True, add_special_tokens=True)
    #     # print(f"{raw_batch} tokenized as {batch}")
    #     batch.to(cfg.device)
    #     Path(f'../data/Extract_EPPO/{model_name}_features/EPPO/{k}').mkdir(exist_ok = True)
    #     with torch.no_grad():
    #         # print(model(**batch).last_hidden_state.detach().cpu().shape)
    #         liste = model(**batch).last_hidden_state.detach().cpu().mean(axis=1)
    #         # print(liste.shape)
    #         names_stacked.append(raw_batch)
    #         features_stacked.append(liste)
    #         codes_stacked.append(raw_codes)
    #         # for i in range(len(raw_batch)):
    #         #     torch.save(liste[i], f"../data/Extract_EPPO/{model_name}_features/EPPO/{k}/{raw_batch[i].replace('/', '-').replace(':','')}")
    # features_stacked =  torch.cat(features_stacked, 0)
    # names_stacked = list(itertools.chain.from_iterable(names_stacked))
    # codes_stacked = list(itertools.chain.from_iterable(codes_stacked))
    # print(features_stacked.shape)
    # dict_features = {'names' : names_stacked, 'tensor' : features_stacked, 'codes' : codes_stacked}
    # torch.save(dict_features, f"../data/Extract_EPPO/{model_name}_features/EPPO.pt")


    for ds in dic_possible_names.keys():

        names_stacked = []
        features_stacked = []
        codes_stacked =[]
        for k in tqdm(dic_possible_names[ds]):
            if len(dic_possible_names[ds][k]) > 0:
                Path(f'../data/Extract_EPPO/{model_name}_features/{ds}/{k}').mkdir(exist_ok=True, parents=True)
                raw_batch = dic_possible_names[ds][k]
                for name in raw_batch:
                    name = name.lower().replace('_', ' ').replace('.','')
                    if name != '' and not os.path.isfile(f'../data/Extract_EPPO/{model_name}_features/{ds}/{k}/{name}.pt'):
                        sentence = f'This plant is affected by {name}'
                        batch = tokenizer(sentence, return_tensors="pt", padding='max_length', max_length=padding_len, truncation=True, add_special_tokens=True)
                        batch.to(cfg.device)
                        with torch.no_grad():
                            response = model(**batch).last_hidden_state.detach().cpu().mean(axis=1)
                            name = name.replace('/','')
                            torch.save(response,f'../data/Extract_EPPO/{model_name}_features/{ds}/{k}/{name}.pt')

        #         raw_batch = [f'This plant is affected by {name}' for name in raw_batch]
        #         raw_codes = [k for i in raw_batch]
        #         raw_batch = [n.replace('_', ' ') for n in raw_batch]
        #         total_raw_batches_used.append({ds:raw_batch})
        #         batch = tokenizer(raw_batch, return_tensors="pt", padding='max_length', max_length=padding_len, truncation=True, add_special_tokens=True)
        #         batch.to(cfg.device)
        #         # Path(f'../data/Extract_EPPO/{model_name}_features/Bousset/{k}').mkdir(exist_ok = True)
        #         with torch.no_grad():
        #             liste = model(**batch).last_hidden_state.detach().cpu().mean(axis=1)
        #             # print(liste.shape)
        #             names_stacked.append(raw_batch)
        #             features_stacked.append(liste)
        #             codes_stacked.append(raw_codes)
        #             #print(liste.shape)
        #             # for i in range(len(raw_batch)):
        #             #     name = raw_batch[i].replace('/', '-')
        #             #     torch.save(liste[i], f"../data/Extract_EPPO/{model_name}_features/PV/{k}/{name}")
        
        # features_stacked =  torch.cat(features_stacked, 0)
        # names_stacked = list(itertools.chain.from_iterable(names_stacked))
        # codes_stacked = list(itertools.chain.from_iterable(codes_stacked))
        # print(features_stacked.shape)
        # dict_features = {'names' : names_stacked, 'tensor' : features_stacked, 'codes' : codes_stacked}
        # torch.save(dict_features, f"../data/Extract_EPPO/{model_name}_features/{ds}.pt")

        # with open('total_raw_batches_used.json', 'w') as f:
        #     json.dump(total_raw_batches_used, f, indent = 2)


    # for k in tqdm(dic_possible_names_PV):
    #     Path(f'../data/Extract_EPPO/{model_name}_features/PV').mkdir(exist_ok=True)
    #     raw_batch = dic_possible_names_PV[k]
    #     batch = tokenizer(raw_batch, return_tensors="pt", padding='max_length', max_length=padding_len)
    #     batch.to(cfg.device)
    #     Path(f'../data/Extract_EPPO/{model_name}_features/PV/{k}').mkdir(exist_ok = True)
    #     with torch.no_grad():
    #         liste = model(**batch).last_hidden_state.detach().cpu().numpy().mean(axis=1)
    #         #print(liste.shape)
    #         for i in range(len(raw_batch)):
    #             name = raw_batch[i].replace('/', '-')
    #             torch.save(liste[i], f"../data/Extract_EPPO/{model_name}_features/PV/{k}/{name}")
                
    # for k in tqdm(dic_possible_names_bousset):
    #     Path(f'../data/Extract_EPPO/{model_name}_features/Bousset').mkdir(exist_ok=True)
    #     raw_batch = dic_possible_names_bousset[k]
    #     batch = tokenizer(raw_batch, return_tensors="pt", padding='max_length', max_length=padding_len)
    #     batch.to(cfg.device)
    #     Path(f'../data/Extract_EPPO/{model_name}_features/Bousset/{k}').mkdir(exist_ok = True)
    #     with torch.no_grad():
    #         #print(batch.shape)
    #         liste = model(**batch).last_hidden_state.detach().cpu().numpy().mean(axis=1)
    #         #print(liste.shape)
    #         for i in range(len(raw_batch)):
    #             name = raw_batch[i].replace('/', '-')
    #             torch.save(liste[i], f"../data/Extract_EPPO/{model_name}_features/Bousset/{k}/{name}")

    # for k in tqdm(dic_possible_names_PN):
    #     Path(f'../data/Extract_EPPO/{model_name}_features/PN').mkdir(exist_ok=True)
    #     raw_batch = dic_possible_names_PN[k]
    #     batch = tokenizer(raw_batch, return_tensors="pt", padding='max_length', max_length=padding_len)
    #     batch.to(cfg.device)
    #     Path(f'../data/Extract_EPPO/{model_name}_features/PN/{k}').mkdir(exist_ok = True)
    #     with torch.no_grad():
    #         #print(batch.shape)
    #         liste = model(**batch).last_hidden_state.detach().cpu().numpy().mean(axis=1)
    #         #print(liste.shape)
    #         for i in range(len(raw_batch)):
    #             name = raw_batch[i].replace('/', '-')
    #             torch.save(liste[i], f"../data/Extract_EPPO/{model_name}_features/PN/{k}/{name}")

    # for k in tqdm(dic_possible_names_IPM):
    #     Path(f'../data/Extract_EPPO/{model_name}_features/IPM').mkdir(exist_ok=True)
    #     raw_batch = dic_possible_names_IPM[k]
    #     if len(raw_batch) > 0:
    #         batch = tokenizer(raw_batch, return_tensors="pt", padding='max_length', max_length=padding_len)
    #         batch.to(cfg.device)
    #         Path(f'../data/Extract_EPPO/{model_name}_features/IPM/{k}').mkdir(exist_ok = True)
    #         with torch.no_grad():
    #             #print(batch.shape)
    #             liste = model(**batch).last_hidden_state.detach().cpu().numpy().mean(axis=1)
    #             #print(liste.shape)
    #             for i in range(len(raw_batch)):
    #                 name = raw_batch[i].replace('/', '-')
    #                 if name != '':
    #                     torch.save(liste[i], f"../data/Extract_EPPO/{model_name}_features/IPM/{k}/{name}")