import json
import os
import xmltodict
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
import argparse
import torch
from types import SimpleNamespace
import rapidfuzz

def list_of_strings(arg):
    return arg.split(',')



if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tolerance', type = float, default=100)
    parser.add_argument('--method', default='EditionDistance', choices=['GPT-embed',
                                                                        'Pllama-embed',
                                                                        'Llama3-embed',
                                                                        'Levenshtein'])
    parser.add_argument('--names-json', type=list_of_strings, default='IPM_names.json')
    parser.add_argument('--drctn', type=str, default = 'DStoEPPO', choices=['EPPOtoDS', 'DStoEPPO'])
    parser.add_argument('--EPPO-json', type=str, default = 'EPPO_codes.json')
    args = parser.parse_args()

    # Tolerance threshold for similarity - from 0 to 1
    tolerance = args.tolerance

    ds_list = [i.replace('.json', '') for i in args.names_json]

    # Loading the EPPO codes and infos from the json obtained with xml_to_json.py
    with open(args.EPPO_json,'r') as f:
        dict_codes = json.load(f)

    dic_possible_names = { ds : {} for ds in ds_list}

    # Loading the dictionnary of possible names for the datasets to map. Obtained with Extract_DS_names.py
    for json_file, ds in zip(args.names_json, ds_list):
        with open(json_file,'r') as f:
            dic_possible_names[ds] = json.load(f)


    if args.method == 'Levenshtein':

        mapping = { ds : {} for ds in ds_list}

        for code in tqdm(dict_codes):
            for name1 in dict_codes[code]['names']:
                if name1['active'] == 'true':
                    # standardizing the names with the same standard
                    standard_name1 = name1['full_name'].lower()
                    standard_name1 = ''.join(x for x in standard_name1.title() if x.isalnum()) 

                    for ds in ds_list:
                        for subid in dic_possible_names[ds]:
                            if subid not in mapping[ds]:
                                mapping[ds][subid] = []
                            for name2 in dic_possible_names[ds][subid]:
                                # standardizing the names with the same standard
                                standard_name2 = name2.lower()
                                standard_name2 = ''.join(x for x in standard_name2.title() if x.isalnum())
                                # Computing the edition distance
                                ratio = rapidfuzz.distance.Levenshtein.normalized_similarity(standard_name1, standard_name2, score_cutoff = tolerance)
                                if ratio >= tolerance:
                                    dict_codes[code]['mapsto'][ds].add((subid, ratio))
                                    mapping[ds][subid].append((code, ratio))


    if args.method == 'GPT-embed':

        cfg = SimpleNamespace(**{})
        cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cosim = torch.nn.CosineSimilarity(dim=1)

        mapping = { ds : {} for ds in ds_list}

        dic_names_EPPO = {}
        for code in dict_codes:
            dic_names_EPPO[code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','').replace('\'','').replace("'", " ") for name in dict_codes[code]['names'] if name['lang'] in considered_languages_for_mapping and name['full_name'] != None and name['active'] == 'true']

        # Loading the precomputed tensors for the different names of each EPPO code
        list_tensors = []
        list_EPPO_codes = []
        for EPPO_code in dic_names_EPPO:
            for EPPO_name in dic_names_EPPO[EPPO_code]:
                EPPO_name = EPPO_name.replace('/','')
                list_tensors.append(torch.load(f'GPT_features/EPPO/{EPPO_code}/{EPPO_name}.pt'))
                list_EPPO_codes.append(EPPO_code)
        tensor_EPPO = torch.Tensor(list_tensors)

        # For each dataset, for each subid and each name, loading the precomputed tensors 
        dic_tensors_DS = {}
        for ds in ds_list:
            dic_tensors_DS[ds] = {}
            for subid in dic_possible_names[ds]:
                dic_tensors_DS[ds][subid] = {}
                for name in dic_possible_names[ds][subid]:
                    name = name.replace('/','').replace('_',' ')      
                    if name != '':
                        dic_tensors_DS[ds][subid][name] = torch.Tensor(torch.load(f'GPT_features/{ds}/{subid}/{name}.pt'))

        for ds in dic_tensors_DS:
            for subid in tqdm(dic_tensors_DS[ds]):
                if subid not in mapping[ds]:
                    mapping[ds][subid] = []
                for name in dic_tensors_DS[ds][subid]:
                    # Defining the similarity function used to compare tensors : Cosine similarity
                    sim = cosim(dic_tensors_DS[ds][subid][name], tensor_EPPO)
                    # Finding the index of the EPPO code that maximizes then similarity (cosine similarity)
                    i_max = torch.argmax(sim).item()
                    if sim[i_max].item() >= tolerance:
                        EPPO_code = list_EPPO_codes[i_max]
                        dict_codes[EPPO_code]['mapsto'][ds].add((subid, sim[i_max].item()))
                        mapping[ds][subid].append((EPPO_code, sim[i_max].item()))


    if args.method == 'Pllama-embed':

        cfg = SimpleNamespace(**{})
        cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cosim = torch.nn.CosineSimilarity(dim=1)

        mapping = { ds : {} for ds in ds_list}


        dic_names_EPPO = {}
        for code in dict_codes:
            dic_names_EPPO[code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','').replace('\'','').replace("'", " ") for name in dict_codes[code]['names'] if name['lang'] in considered_languages_for_mapping and name['full_name'] != None and name['active'] == 'true']

        # Loading the precomputed tensors for the different names of each EPPO code
        list_tensors = []
        list_EPPO_codes = []
        for EPPO_code in dic_names_EPPO:
            for EPPO_name in dic_names_EPPO[EPPO_code]:
                EPPO_name = EPPO_name.replace('/','')          
                list_tensors.append(torch.load(f'PLLaMa-7b-base_features/EPPO/{EPPO_code}/{EPPO_name}.pt')[0].tolist())
                list_EPPO_codes.append(EPPO_code)
        # print(list_tensors[0])
        tensor_EPPO = torch.Tensor(list_tensors)

        # For each dataset, for each subid and each name, loading the precomputed tensors 
        dic_tensors_DS = {}
        for ds in ds_list:
            dic_tensors_DS[ds] = {}
            for subid in dic_possible_names[ds]:
                dic_tensors_DS[ds][subid] = {}
                for name in dic_possible_names[ds][subid]:
                    name = name.replace('/','').replace('_',' ')
                    if name != '':
                        dic_tensors_DS[ds][subid][name] = torch.Tensor(torch.load(f'PLLaMa-7b-base_features/{ds}/{subid}/{name}.pt'))

        for ds in dic_tensors_DS:
            for subid in tqdm(dic_tensors_DS[ds]):
                if subid not in mapping[ds]:
                    mapping[ds][subid] = []
                for name in dic_tensors_DS[ds][subid]:
                    # Defining the similarity function used to compare tensors : Cosine similarity
                    sim = cosim(dic_tensors_DS[ds][subid][name], tensor_EPPO)
                    # Finding the index of the EPPO code that maximizes then similarity (cosine similarity)
                    i_max = torch.argmax(sim).item()
                    if sim[i_max].item() >= tolerance:
                        EPPO_code = list_EPPO_codes[i_max]
                        dict_codes[EPPO_code]['mapsto'][ds].add((subid, sim[i_max].item()))
                        mapping[ds][subid].append((EPPO_code, sim[i_max].item()))


    if args.method == 'Llama3-embed':

        cfg = SimpleNamespace(**{})
        cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cosim = torch.nn.CosineSimilarity(dim=1)

        mapping = { ds : {} for ds in ds_list}

        dic_names_EPPO = {}
        for code in dict_codes:
            dic_names_EPPO[code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','').replace('\'','').replace("'", " ") for name in dict_codes[code]['names'] if name['lang'] in considered_languages_for_mapping and name['full_name'] != None and name['active'] == 'true']

        # Loading the precomputed tensors for the different names of each EPPO code
        list_tensors = []
        list_EPPO_codes = []
        for EPPO_code in dic_names_EPPO:
            for EPPO_name in dic_names_EPPO[EPPO_code]:
                EPPO_name = EPPO_name.replace('/','')          
                list_tensors.append(torch.load(f'Meta-Llama-3.1-8B_features/EPPO/{EPPO_code}/{EPPO_name}.pt')[0].tolist())
                list_EPPO_codes.append(EPPO_code)
        # print(list_tensors[0])
        tensor_EPPO = torch.Tensor(list_tensors)

        # For each dataset, for each subid and each name, loading the precomputed tensors 
        dic_tensors_DS = {}
        for ds in ds_list:
            dic_tensors_DS[ds] = {}
            for subid in dic_possible_names[ds]:
                dic_tensors_DS[ds][subid] = {}
                for name in dic_possible_names[ds][subid]:
                    name = name.replace('/','').replace('_',' ')
                    if name != '':
                        dic_tensors_DS[ds][subid][name] = torch.Tensor(torch.load(f'Meta-Llama-3.1-8B_features/{ds}/{subid}/{name}.pt'))

        for ds in dic_tensors_DS:
            for subid in tqdm(dic_tensors_DS[ds]):
                if subid not in mapping[ds]:
                    mapping[ds][subid] = []
                for name in dic_tensors_DS[ds][subid]:
                    # Defining the similarity function used to compare tensors : Cosine similarity
                    sim = cosim(dic_tensors_DS[ds][subid][name], tensor_EPPO)
                    # Finding the index of the EPPO code that maximizes then similarity (cosine similarity)
                    i_max = torch.argmax(sim).item()
                    if sim[i_max].item() >= tolerance:
                        EPPO_code = list_EPPO_codes[i_max]
                        dict_codes[EPPO_code]['mapsto'][ds].add((subid, sim[i_max].item()))
                        mapping[ds][subid].append((EPPO_code, sim[i_max].item()))




    # Replacing sets by lists to make the dict json parsable
    for code in dict_codes:
        for db in dict_codes[code]['mapsto']:
            dict_codes[code]['mapsto'][db] = list(dict_codes[code]['mapsto'][db])

    if 'outputs' not in os.listdir():
        os.mkdir('outputs')

    with open(f'outputs/CodesDict_{args.method}_{tolerance}_V2.json', 'w') as f:
        json.dump(dict_codes, f, indent = 2)

    if args.drctn == 'EPPOtoDS':

        # Method 1 : Finding in each dataset the subid that matches best a given EPPO code. 

        final_mapping = {}
        for ds in ds_list:
            final_mapping[ds] = {subid : [] for subid in dic_possible_names[ds]}
            for code in dict_codes:
                if dict_codes[code]['mapsto'][ds] != []:
                    if len(dict_codes[code]['mapsto'][ds]) > 0:
                        
                        best_mapping = dict_codes[code]['mapsto'][ds][np.argmax([i1[1] for i1 in dict_codes[code]['mapsto'][ds]])]                                

                        if len(final_mapping[ds][best_mapping[0]]) > 0:
                            if final_mapping[ds][best_mapping[0]][1] < best_mapping[1]:
                                final_mapping[ds][best_mapping[0]] = (code, best_mapping[1])
                        else:
                            final_mapping[ds][best_mapping[0]] = (code, best_mapping[1])
    
    elif args.drctn == 'DStoEPPO':

        # Method 2 : Finding for a given subid in a dataset the EPPO code that matches best

        final_mapping = {}
        for ds in mapping:
            final_mapping[ds] = {subid : [] for subid in dic_possible_names[ds]}
            for subid in mapping[ds]:
                if len(mapping[ds][subid]) > 0:
                    sim = [i[1] for i in mapping[ds][subid]]
                    i_max = np.argmax(sim)
                    final_mapping[ds][subid] = mapping[ds][subid][i_max]
                else:
                    final_mapping[ds][subid] = []


    with open(f'outputs/Mapping_{args.method}_{tolerance}_{args.drctn}_V2.json', 'w') as f:
        json.dump(final_mapping, f, indent=2)



