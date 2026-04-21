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


def list_of_strings(arg):
    return arg.split(',')

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type = str, default='')
    parser.add_argument('--ds-list', type = list_of_strings)
    args = parser.parse_args()


    dic_possible_names = {}

    for ds in args.ds_list:
        with open(f'../data/Extract_EPPO/Harmonization_outputs/DS_names/{ds}_names.json','r') as f:
            dic_possible_names[ds] = json.load(f)


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
        dic_possible_names['EPPO'][code] = [name['full_name'].lower().strip(' _').replace('_',' ').replace('.','').replace('\'','') for name in dict_codes[code]['names'] if name['lang'] in considered_lang and name['full_name'] != None and name['active'] == 'true']



    client = OpenAI(api_key='sk-proj-rc8MovhS7XUdqNS7FIG4T3BlbkFJYGEs27t6fLcPqn8cTy9x')

    if args.name != '':
        name_exp = f'_{args.name}'
    else:
        name_exp = ''

    Path(f'../data/Extract_EPPO/GPT_features{name_exp}').mkdir(exist_ok=True)

    if args.name == 'hierarchic':
        with open('../data/Extract_EPPO/Hierarchy.json','r') as f:
            hierarchy = json.load(f)


    for ds in dic_possible_names:
        Path(f'../data/Extract_EPPO/GPT_features{name_exp}/{ds}').mkdir(exist_ok=True)
        for subid in tqdm(dic_possible_names[ds]):
            Path(f'../data/Extract_EPPO/GPT_features{name_exp}/{ds}/{subid}').mkdir(exist_ok=True)

            for name in dic_possible_names[ds][subid]:
                name = name.lower().replace('_', ' ').replace('.','')
                if name != '' and not os.path.isfile(f'../data/Extract_EPPO/GPT_features{name_exp}/{ds}/{subid}/{name}.pt'):
                    if args.name == "expert_prompt":
                        sentence = f'As an expert phytopathologist, with comprehensive knowledge on pests and diseases of plants, and particularly crops, tell me how to differentiate {name} that affect my plant from other similar damages.'
                    elif args.name == "expert_prompt2":
                        sentence = f'As a expert phytopathologist, calculate an embedding of the following plant disease name, avoiding to make a different embedding if two names are very very similar and vary by only 2 or 3 characters : {name}'
                    elif args.name == "expert_prompt3":
                        sentence = f'''I am working with a list of plant pathogen and damage names to identify possible synonyms based on their underlying phytopathological features.
                                    For each name, I need to extract a precise, domain-specific representation that reflects the biological and symptomatic relationships between different pathogens and types of plant damage. 
                                    Many names may appear distinct but represent similar concepts in terms of plant disease and damage mechanisms.
                                    I will provide you either French common names, English common names or scientific name

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
                    elif args.name == 'hierarchic':
                        complete_description = f'{name} '
                        appendix = 'X'
                        if subid in hierarchy[ds].keys():
                            appendix1 = 'which '
                            appendix2 = ''.join(f'{key} is {hierarchy[ds][subid][key]},' for key in hierarchy[ds][subid].keys() if key in ['kingdom', 'genus', 'order', 'family'])
                            if appendix2 != '':
                                appendix = appendix1 + appendix2
                        complete_description = complete_description + appendix
                        complete_description = complete_description[:-1]
                        sentence = f'As a expert phytopathologist, with comprehensive knowledge on pests and diseases of plants, and particularly crops, tell me how to differientate {complete_description} that affect my plant from other similar dammages.'
                    else:
                        sentence = f'This plant is affected by {name}'
                    response = client.embeddings.create(
                        input=sentence,
                        model="text-embedding-3-large"
                    )
                    embdg = response.data[0].embedding
                    name = name.replace('/','')          
                    torch.save(embdg,f'../data/Extract_EPPO/GPT_features{name_exp}/{ds}/{subid}/{name}.pt')


