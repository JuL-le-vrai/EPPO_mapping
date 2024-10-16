import json
import pandas as pd
import re
from pathlib import Path


considered_languages_for_mapping = ['la', 'en', 'fr']


#### Extracting IPM possible names

Path("../data/Extract_EPPO/Harmonization_outputs/DS_names/").mkdir(exist_ok=True, parents=True)

BD_IPM = pd.read_csv('../data/IPM_data/BD_bugwood.csv', sep = ';')
subjects_IPM = BD_IPM['subid'].unique()
dic_possible_names_IPM = {}
for subid in subjects_IPM:
    df_sub = BD_IPM[BD_IPM['subid'] == subid]
    subid = str(subid)
    dic_possible_names_IPM[subid] = set()
    dic_possible_names_IPM[subid] = dic_possible_names_IPM[subid].union([x for x in list(df_sub['sub_name'].unique()) if not str(x) == 'nan'])
    dic_possible_names_IPM[subid] = dic_possible_names_IPM[subid].union([x for x in list(df_sub['scientificName'].unique()) if not str(x) == 'nan']) 
    dic_possible_names_IPM[subid] = dic_possible_names_IPM[subid].union([x for x in list(df_sub['commonName'].unique()) if not str(x) == 'nan'])
    possible_synos = list(df_sub['syn'].unique())
    synos = []
    for s in possible_synos:
        if not str(s) == 'nan':
            for j in re.findall(r"\(.*?\)", s):
                s.replace(j, j.replace(',','|'))
            [synos.append(syno) for syno in s.split(',')]

    dic_possible_names_IPM[subid] = dic_possible_names_IPM[subid].union(synos)
    dic_possible_names_IPM[subid] = [i.lower().strip(' _').replace(' ','_').replace('.','') for i in dic_possible_names_IPM[subid]]


with open(f'../data/Extract_EPPO/Harmonization_outputs/DS_names/IPM_names.json', 'w') as f:
    json.dump(dic_possible_names_IPM, f, indent = 2)

#### Extracting ephytia possible names

BD_PN = pd.read_csv('../data/PN_data/PN_import_ePhy_BD.csv', sep = ';')
subjects_names_PN = list(BD_PN['patho'].unique())
subjects_names_PN.sort()
dic_possible_names_PN = {}

for k in subjects_names_PN:
    dic_possible_names_PN[k] = set()
    dic_possible_names_PN[k].add(k)
    dic_possible_names_PN[k] = [i.lower().strip(' _').replace(' ','_').replace('.','').replace('_spp', '_sp') for i in dic_possible_names_PN[k]]

with open(f'../data/Extract_EPPO/Harmonization_outputs/DS_names/PN_names.json', 'w') as f:
    json.dump(dic_possible_names_PN, f, indent = 2)


#### Extracting Bousset possible names

BD_bousset = pd.read_csv('../data/Bousset_data/BD_Bousset.csv', sep = ';')
subjects_bousset = list(BD_bousset['organisme'].unique())
dic_possible_names_bousset = {}

for k in subjects_bousset:
    df_sub = BD_bousset[BD_bousset['organisme'] == k]
    dic_possible_names_bousset[k] = set()
    # dic_possible_names_bousset[k] = dic_possible_names_bousset[k].union([x for x in list(df_sub['cause'].unique()) if not str(x) == 'nan'])
    dic_possible_names_bousset[k].add(k)
    dic_possible_names_bousset[k] = [i.lower().strip(' _').replace(' ','_').replace('.','') for i in dic_possible_names_bousset[k]]

with open(f'../data/Extract_EPPO/Harmonization_outputs/DS_names/Bousset_names.json', 'w') as f:
    json.dump(dic_possible_names_bousset, f, indent = 2)

#### Extracting Bousset2 possible names

BD_bousset2 = pd.read_csv('../data/Bousset2_data/BD_Bousset2.csv', sep = ';')
subjects_bousset2 = list(BD_bousset2['subid'].unique())
dic_possible_names_bousset2 = {}

for k in subjects_bousset2:
    df_sub = BD_bousset2[BD_bousset2['subid'] == k]
    dic_possible_names_bousset2[k] = set()
    # dic_possible_names_bousset2[k] = dic_possible_names_bousset[k].union([x for x in list(df_sub['cause'].unique()) if not str(x) == 'nan'])
    dic_possible_names_bousset2[k].add(k)
    dic_possible_names_bousset2[k] = [i.lower().strip(' _').replace(' ','_').replace('.','') for i in dic_possible_names_bousset2[k]]

with open(f'../data/Extract_EPPO/Harmonization_outputs/DS_names/Bousset2_names.json', 'w') as f:
    json.dump(dic_possible_names_bousset2, f, indent = 2)


### Extracting PV possible names
    
BD_PV = pd.read_csv('../data/PV_data/BD_PV.csv', sep = ';')
BD_PV['subid'] = BD_PV['patho']
subjects_PV = list(BD_PV['subid'].unique())
dic_possible_names_PV ={}
for patho in subjects_PV:
    liste_names = patho.split(' ')
    real_liste_names = set()
    for name in liste_names:
        matches = re.findall(r"\(.*?\)", name)
        if len(matches) > 0:
            for match in matches:
                if match in name.split('___')[1]:
                    real_liste_names.add(match.replace('(', '').replace(')', ''))
                name = name.replace(match, '')
        real_liste_names.add(name)

    dic_possible_names_PV[patho] = [i.lower().strip(' _').replace('___',' ').replace(' ','_').replace('.','').replace('__','_') for i in real_liste_names]

with open(f'../data/Extract_EPPO/Harmonization_outputs/DS_names/PV_names.json', 'w') as f:
    json.dump(dic_possible_names_PV, f, indent = 2)



#### Extracting CDDM possible names

BD_CDDM = pd.read_csv('../data/CDDM_data/BD_CDDM.csv', sep = ';')
BD_CDDM['subid'] = BD_CDDM['crop'] + ' ' + BD_CDDM['code']
subjects_CDDM = list(BD_CDDM['subid'].unique())
dic_possible_names_CDDM = {}

for k in subjects_CDDM:
    df_sub = BD_CDDM[BD_CDDM['subid'] == k]
    dic_possible_names_CDDM[k] = set()
    # dic_possible_names_bousset2[k] = dic_possible_names_bousset[k].union([x for x in list(df_sub['cause'].unique()) if not str(x) == 'nan'])
    dic_possible_names_CDDM[k].add(k)
    dic_possible_names_CDDM[k] = [i.lower().strip(' _').replace(' ','_').replace('.','') for i in dic_possible_names_CDDM[k]]

with open(f'../data/Extract_EPPO/Harmonization_outputs/DS_names/CDDM_names.json', 'w') as f:
    json.dump(dic_possible_names_CDDM, f, indent = 2)
