import json
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
                                                                        'GPT-embed-ExpertPrompt',
                                                                        'EditionDistanceIndel',
                                                                        'GPT-embed-ExpertPrompt3',
                                                                        'GPT-embed-relative-cosim',
                                                                        'HandmapedOnly',
                                                                        'GPT-HybridHandmaped',
                                                                        'Pllama-embed',
                                                                        'Llama3-embed',
                                                                        'Levenshtein'])
    parser.add_argument('--ds-list', type=list_of_strings, default='PN,PV,Bousset,IPM')
    parser.add_argument('--drctn', type=str, default = 'DStoEPPO', choices=['EPPOtoDS', 'DStoEPPO'])
    args = parser.parse_args()

    # Tolerance threshold for similarity - from 0 to 1
    tolerance = args.tolerance

    # Loading the xml file of EPPO codes (downloadable on the EPPO website)
    with open('../data/Extract_EPPO/fullcodes.xml','r') as f:
        xml = xmltodict.parse(f.read())

    dict_codes = {}

    # Languages considered for the mapping
    considered_languages_for_mapping = ['la', 'en', 'fr']

    # Creating a clean dictionary with EPPO codes and infos found about it in the xml
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
                    if n['lang'] in considered_languages_for_mapping:
                        indiv_names.append({'full_name' : n['fullname'], 'lang' : n['lang'], 'active' : n['@isactive']})

            dict_codes[indiv_code] = {'type' : indiv_type, 'parents' : indiv_parents, 'names' : indiv_names, 'mapsto' : { ds : set() for ds in args.ds_list}}

    dic_possible_names = { ds : {} for ds in args.ds_list}


    # Loading the dictionnary of possible names for the datasets to map. Obtained with Extract_DS_names.py
    for ds in args.ds_list:
        with open(f'../data/Extract_EPPO/Harmonization_outputs/DS_names/{ds}_names.json','r') as f:
            dic_possible_names[ds] = json.load(f)


    if args.method == 'EditionDistanceIndel':

        mapping = { ds : {} for ds in args.ds_list}

        for code in tqdm(dict_codes):
            for name1 in dict_codes[code]['names']:
                if name1['active'] == 'true':
                    # standardizing the names with the same standard
                    standard_name1 = name1['full_name'].lower()
                    standard_name1 = ''.join(x for x in standard_name1.title() if x.isalnum()) 

                    for ds in args.ds_list:
                        for subid in dic_possible_names[ds]:
                            if subid not in mapping[ds]:
                                mapping[ds][subid] = []
                            for name2 in dic_possible_names[ds][subid]:
                                # standardizing the names with the same standard
                                standard_name2 = name2.lower()
                                standard_name2 = ''.join(x for x in standard_name2.title() if x.isalnum())
                                # Computing the edition distance
                                ratio = rapidfuzz.fuzz.ratio(standard_name1, standard_name2, score_cutoff=100*tolerance)/100
                                if ratio >= tolerance:
                                    dict_codes[code]['mapsto'][ds].add((subid, ratio))
                                    mapping[ds][subid].append((code, ratio))

    if args.method == 'Levenshtein':

        mapping = { ds : {} for ds in args.ds_list}

        for code in tqdm(dict_codes):
            for name1 in dict_codes[code]['names']:
                if name1['active'] == 'true':
                    # standardizing the names with the same standard
                    standard_name1 = name1['full_name'].lower()
                    standard_name1 = ''.join(x for x in standard_name1.title() if x.isalnum()) 

                    for ds in args.ds_list:
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



                                    

    if args.method == 'HandmapedOnly':

        mapping = { ds : {} for ds in args.ds_list}
        for ds in args.ds_list:
            df = pd.read_csv(f'../data/Extract_EPPO/EPPO_API_results_{ds}.csv', sep = ';')
            for _, row in df.iterrows():
                if row['EPPO_final'] != '****NOT FOUND*****':
                    if row['iName'] not in mapping[ds]:
                        mapping[ds][row['iName']] = []
                    mapping[ds][row['iName']].append((row['EPPO_final'], 101))
                    dict_codes[row['EPPO_final']]['mapsto'][ds].add((str(row['iName']), 101))



    if args.method == 'GPT-HybridHandmaped':

        mapping = { ds : {} for ds in args.ds_list}
        for ds in args.ds_list:
            df = pd.read_csv(f'../data/Extract_EPPO/EPPO_API_results_{ds}.csv', sep = ';')
            for _, row in df.iterrows():
                if row['EPPO_final'] != '****NOT FOUND*****':
                    if row['iName'] not in mapping[ds]:
                        mapping[ds][row['iName']] = []
                    mapping[ds][row['iName']].append((row['EPPO_final'], 101))
                    dict_codes[row['EPPO_final']]['mapsto'][ds].add((str(row['iName']), 101))


        cfg = SimpleNamespace(**{})
        cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cosim = torch.nn.CosineSimilarity(dim=1)

        considered_lang = ['la', 'en', 'fr']
        dic_names_EPPO = {}
        for code in dict_codes:
            dic_names_EPPO[code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','') for name in dict_codes[code]['names'] if name['lang'] in considered_lang and name['full_name'] != None and name['active'] == 'true']

        # Loading the precomputed tensors for the different names of each EPPO code
        list_tensors = []
        list_EPPO_codes = []
        for EPPO_code in dic_names_EPPO:
            for EPPO_name in dic_names_EPPO[EPPO_code]:
                EPPO_name = EPPO_name.replace('/','')          
                list_tensors.append(torch.load(f'../data/Extract_EPPO/GPT_features/EPPO/{EPPO_code}/{EPPO_name}.pt'))
                list_EPPO_codes.append(EPPO_code)
        tensor_EPPO = torch.Tensor(list_tensors)

        # For each dataset, for each subid and each name, loading the precomputed tensors 
        dic_tensors_DS = {}
        for ds in args.ds_list:
            dic_tensors_DS[ds] = {}
            for subid in dic_possible_names[ds]:
                dic_tensors_DS[ds][subid] = {}
                for name in dic_possible_names[ds][subid]:
                    name = name.replace('/','').replace('_',' ')      
                    if name != '':
                        dic_tensors_DS[ds][subid][name] = torch.Tensor(torch.load(f'../data/Extract_EPPO/GPT_features/{ds}/{subid}/{name}.pt'))

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



    if args.method == 'GPT-embed':

        cfg = SimpleNamespace(**{})
        cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cosim = torch.nn.CosineSimilarity(dim=1)

        mapping = { ds : {} for ds in args.ds_list}

        considered_lang = ['la', 'en', 'fr']
        dic_names_EPPO = {}
        for code in dict_codes:
            dic_names_EPPO[code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','') for name in dict_codes[code]['names'] if name['lang'] in considered_lang and name['full_name'] != None and name['active'] == 'true']

        # Loading the precomputed tensors for the different names of each EPPO code
        list_tensors = []
        list_EPPO_codes = []
        for EPPO_code in dic_names_EPPO:
            for EPPO_name in dic_names_EPPO[EPPO_code]:
                EPPO_name = EPPO_name.replace('/','')          
                list_tensors.append(torch.load(f'../data/Extract_EPPO/GPT_features/EPPO/{EPPO_code}/{EPPO_name}.pt'))
                list_EPPO_codes.append(EPPO_code)
        tensor_EPPO = torch.Tensor(list_tensors)

        # For each dataset, for each subid and each name, loading the precomputed tensors 
        dic_tensors_DS = {}
        for ds in args.ds_list:
            dic_tensors_DS[ds] = {}
            for subid in dic_possible_names[ds]:
                dic_tensors_DS[ds][subid] = {}
                for name in dic_possible_names[ds][subid]:
                    name = name.replace('/','').replace('_',' ')      
                    if name != '':
                        dic_tensors_DS[ds][subid][name] = torch.Tensor(torch.load(f'../data/Extract_EPPO/GPT_features/{ds}/{subid}/{name}.pt'))

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

        mapping = { ds : {} for ds in args.ds_list}

        considered_lang = ['la', 'en', 'fr']
        dic_names_EPPO = {}
        for code in dict_codes:
            dic_names_EPPO[code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','').replace('\'','') for name in dict_codes[code]['names'] if name['lang'] in considered_lang and name['full_name'] != None and name['active'] == 'true']

        # Loading the precomputed tensors for the different names of each EPPO code
        list_tensors = []
        list_EPPO_codes = []
        for EPPO_code in dic_names_EPPO:
            for EPPO_name in dic_names_EPPO[EPPO_code]:
                EPPO_name = EPPO_name.replace('/','')          
                list_tensors.append(torch.load(f'../data/Extract_EPPO/PLLaMa-7b-base_features/EPPO/{EPPO_code}/{EPPO_name}.pt')[0].tolist())
                list_EPPO_codes.append(EPPO_code)
        # print(list_tensors[0])
        tensor_EPPO = torch.Tensor(list_tensors)

        # For each dataset, for each subid and each name, loading the precomputed tensors 
        dic_tensors_DS = {}
        for ds in args.ds_list:
            dic_tensors_DS[ds] = {}
            for subid in dic_possible_names[ds]:
                dic_tensors_DS[ds][subid] = {}
                for name in dic_possible_names[ds][subid]:
                    name = name.replace('/','').replace('_',' ')
                    if name != '':
                        dic_tensors_DS[ds][subid][name] = torch.Tensor(torch.load(f'../data/Extract_EPPO/PLLaMa-7b-base_features/{ds}/{subid}/{name}.pt'))

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

        mapping = { ds : {} for ds in args.ds_list}

        considered_lang = ['la', 'en', 'fr']
        dic_names_EPPO = {}
        for code in dict_codes:
            dic_names_EPPO[code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','').replace('\'','') for name in dict_codes[code]['names'] if name['lang'] in considered_lang and name['full_name'] != None and name['active'] == 'true']

        # Loading the precomputed tensors for the different names of each EPPO code
        list_tensors = []
        list_EPPO_codes = []
        for EPPO_code in dic_names_EPPO:
            for EPPO_name in dic_names_EPPO[EPPO_code]:
                EPPO_name = EPPO_name.replace('/','')          
                list_tensors.append(torch.load(f'../data/Extract_EPPO/Meta-Llama-3.1-8B_features/EPPO/{EPPO_code}/{EPPO_name}.pt')[0].tolist())
                list_EPPO_codes.append(EPPO_code)
        # print(list_tensors[0])
        tensor_EPPO = torch.Tensor(list_tensors)

        # For each dataset, for each subid and each name, loading the precomputed tensors 
        dic_tensors_DS = {}
        for ds in args.ds_list:
            dic_tensors_DS[ds] = {}
            for subid in dic_possible_names[ds]:
                dic_tensors_DS[ds][subid] = {}
                for name in dic_possible_names[ds][subid]:
                    name = name.replace('/','').replace('_',' ')
                    if name != '':
                        dic_tensors_DS[ds][subid][name] = torch.Tensor(torch.load(f'../data/Extract_EPPO/Meta-Llama-3.1-8B_features/{ds}/{subid}/{name}.pt'))

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




    if args.method == 'GPT-embed-ExpertPrompt3':

        cfg = SimpleNamespace(**{})
        cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cosim = torch.nn.CosineSimilarity(dim=1)

        mapping = { ds : {} for ds in args.ds_list}

        considered_lang = ['la', 'en', 'fr']
        dic_names_EPPO = {}
        for code in dict_codes:
            dic_names_EPPO[code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','').replace('\'','') for name in dict_codes[code]['names'] if name['lang'] in considered_lang and name['full_name'] != None and name['active'] == 'true']

        # Loading the precomputed tensors for the different names of each EPPO code
        list_tensors = []
        list_EPPO_codes = []
        for EPPO_code in dic_names_EPPO:
            for EPPO_name in dic_names_EPPO[EPPO_code]:
                EPPO_name = EPPO_name.replace('/','')          
                list_tensors.append(torch.load(f'../data/Extract_EPPO/GPT_features_expert_prompt3/EPPO/{EPPO_code}/{EPPO_name}.pt'))
                list_EPPO_codes.append(EPPO_code)
        print(list_tensors[0])
        tensor_EPPO = torch.Tensor(list_tensors)

        # For each dataset, for each subid and each name, loading the precomputed tensors 
        dic_tensors_DS = {}
        for ds in args.ds_list:
            dic_tensors_DS[ds] = {}
            for subid in dic_possible_names[ds]:
                dic_tensors_DS[ds][subid] = {}
                for name in dic_possible_names[ds][subid]:
                    name = name.replace('/','').replace('_',' ')      
                    if name != '':
                        dic_tensors_DS[ds][subid][name] = torch.Tensor(torch.load(f'../data/Extract_EPPO/GPT_features_expert_prompt3/{ds}/{subid}/{name}.pt'))

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



    if args.method == 'GPT-embed-relative-cosim':

        cfg = SimpleNamespace(**{})
        cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cosim = torch.nn.CosineSimilarity(dim=1)

        mapping = { ds : {} for ds in args.ds_list}

        considered_lang = ['la', 'en', 'fr']
        dic_names_EPPO = {}
        for code in dict_codes:
            dic_names_EPPO[code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','') for name in dict_codes[code]['names'] if name['lang'] in considered_lang and name['full_name'] != None and name['active'] == 'true']

        # Loading the precomputed tensors for the different names of each EPPO code
        list_tensors = []
        list_EPPO_codes = []
        for EPPO_code in dic_names_EPPO:
            for EPPO_name in dic_names_EPPO[EPPO_code]:
                EPPO_name = EPPO_name.replace('/','')          
                list_tensors.append(torch.Tensor(torch.load(f'../data/Extract_EPPO/GPT_features/EPPO/{EPPO_code}/{EPPO_name}.pt')))
                list_EPPO_codes.append(EPPO_code)
        tensor_EPPO = torch.stack(list_tensors)

        # For each dataset, for each subid and each name, loading the precomputed tensors 
        dic_tensors_DS = {}
        for ds in args.ds_list:
            dic_tensors_DS[ds] = {}
            for subid in dic_possible_names[ds]:
                dic_tensors_DS[ds][subid] = {}
                for name in dic_possible_names[ds][subid]:
                    name = name.replace('/','').replace('_',' ')      
                    if name != '':
                        dic_tensors_DS[ds][subid][name] = torch.Tensor(torch.load(f'../data/Extract_EPPO/GPT_features/{ds}/{subid}/{name}.pt'))

        for ds in dic_tensors_DS:
            for subid in tqdm(dic_tensors_DS[ds]):
                if subid not in mapping[ds]:
                    mapping[ds][subid] = []
                for name in dic_tensors_DS[ds][subid]:
                    # Defining the similarity function used to compare tensors : Relative Cosine similarity
                    n = 10
                    sim = cosim(dic_tensors_DS[ds][subid][name], tensor_EPPO) # Compute cosim
                    top_n_similarities, top_n_indices = torch.topk(sim, n, dim=0) # Top n
                    sum_top_n = torch.sum(top_n_similarities, dim=0, keepdim=True)
                    sim = sim / sum_top_n
                    # Finding the index of the EPPO code that maximizes then similarity (cosine similarity)
                    i_max = torch.argmax(sim).item()
                    if sim[i_max].item() >= tolerance/n:
                        EPPO_code = list_EPPO_codes[i_max]
                        dict_codes[EPPO_code]['mapsto'][ds].add((subid, sim[i_max].item()))
                        mapping[ds][subid].append((EPPO_code, sim[i_max].item()))

    # Replacing sets by lists to make the dict json parsable
    for code in dict_codes:
        for db in dict_codes[code]['mapsto']:
            dict_codes[code]['mapsto'][db] = list(dict_codes[code]['mapsto'][db])

    with open(f'../data/Extract_EPPO/Harmonization_outputs/CodesDict_{args.method}_{tolerance}_V2.json', 'w') as f:
        json.dump(dict_codes, f, indent = 2)

    if args.drctn == 'EPPOtoDS':

        # Method 1 : Finding in each dataset the subid that matches best a given EPPO code. 

        final_mapping = {}
        for ds in args.ds_list:
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


    if args.method =='HandmapedOnly':

        final_mapping = {}
        for ds in mapping:
            final_mapping[ds] = {subid : [] for subid in dic_possible_names[ds]}
            for subid in mapping[ds]:
                if len(mapping[ds][subid]) > 0:
                    final_mapping[ds][subid] = mapping[ds][subid][0]
                else:
                    final_mapping[ds][subid] = []


    with open(f'../data/Extract_EPPO/Harmonization_outputs/Mapping_{args.method}_{tolerance}_{args.drctn}_V2.json', 'w') as f:
        json.dump(final_mapping, f, indent=2)



