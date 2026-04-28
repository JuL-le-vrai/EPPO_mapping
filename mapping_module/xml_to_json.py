import json
import os
import xmltodict
import argparse
from types import SimpleNamespace


def list_of_strings(arg):
    return arg.split(',')



if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--EPPO-xml', type=str, default = 'fullcodes.xml')
    args = parser.parse_args()

    # Loading the xml file of EPPO codes (downloadable on the EPPO website)
    with open(args.EPPO_xml,'r') as f:
        xml = xmltodict.parse(f.read())

    ds_list = [i.replace('.json', '') for i in args.names_json]

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

            dict_codes[indiv_code] = {'type' : indiv_type, 'parents' : indiv_parents, 'names' : indiv_names, 'mapsto' : { ds : set() for ds in ds_list}}


    with open('EPPO_codes.json', 'w') as f:
        json.dump(dict_codes, f)
        
