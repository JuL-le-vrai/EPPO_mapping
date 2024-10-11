import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import numpy as np
from pathlib import Path

def list_of_strings(arg):
    return arg.split(',')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--list-mappings', type=list_of_strings, default='PN')
    parser.add_argument('--SimRounding', type = int, default = False)
    parser.add_argument('--IPM-only', action='store_true', default = False)
    parser.add_argument('--mapping-choice', action= 'store_true', default = False)
    args = parser.parse_args()

    GT_PN = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_PN.csv', sep = ';')
    GT_PV = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_PV.csv', sep = ';')
    GT_Bousset = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_Bousset.csv', sep = ';')
    GT_IPM = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_IPM.csv', sep = ';')


    GT_tot = {'Bousset' : {}, 'PN' : {}, 'PV' : {}, 'IPM' : {}}

    for _, l in GT_PN.iterrows():
        GT_tot['PN'][l['Name_PN'].replace(' ', '_')] = l['EPPO_final']

    for _, l in GT_Bousset.iterrows():
        GT_tot['Bousset'][l['organisme']] = l['EPPO_final']

    for _, l in GT_PV.iterrows():
        GT_tot['PV'][l['iName']] = l['EPPO_final']

    for _, l in GT_IPM.iterrows():
        GT_tot['IPM'][str(l['iName'])] = l['EPPO_final']

    tot_tol = []
    tot_name = []
    tot_TPR = []
    tot_FPR = []
    tot_Recall = []
    tot_F1 = []
    tot_Precision = []
    tot_IPM_mapped = []
    tot_img_IPM_map = []

    IPM_df = pd.read_csv('../data/IPM_data/BD_IPM.csv', sep = ';', encoding = 'utf-8')

    for mapping_path in tqdm(args.list_mappings):


        with open(mapping_path, 'r') as f:
            Inferred_mapping = json.load(f)

        grain = 0.005
        tol = 0.75

        list_tol = []
        list_FPR = []
        list_TPR = []
        list_Recall = []
        list_Precision = [] 
        list_F1 = []
        list_IPM_mapped = []
        list_images_IPM_mapped = []

        if args.SimRounding != False:

            for ds in Inferred_mapping:
                for k in Inferred_mapping[ds]:
                    if len(Inferred_mapping[ds][k]) > 0:
                        Inferred_mapping[ds][k][1] = round(Inferred_mapping[ds][k][1], args.SimRounding)


        nearly_one = 0.99999

        while tol <= nearly_one:

            V_positif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
            F_positif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
            F_negatif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
            V_negatif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
            total_mapped = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
            total_with_GT_mapping = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}


            F_positif_list = []

            for ds in Inferred_mapping:
                for k in Inferred_mapping[ds]:
                    if len(Inferred_mapping[ds][k]) > 0:
                        # print(Inferred_mapping[ds][k])
                        if Inferred_mapping[ds][k][1] < tol:
                            Inferred_mapping[ds][k] = []

            inv_mapping = {}
            for ds in Inferred_mapping:
                for subid in Inferred_mapping[ds]:
                    if len(Inferred_mapping[ds][subid]) > 0:
                        if Inferred_mapping[ds][subid][0] not in inv_mapping:
                            inv_mapping[Inferred_mapping[ds][subid][0]] = set()
                        inv_mapping[Inferred_mapping[ds][subid][0]].add(ds)

            Overlaped_labels = {}   

            for eppo_code in inv_mapping:
                if len(inv_mapping[eppo_code]) > 1:
                    ovl_ds = tuple(inv_mapping[eppo_code])
                    if ovl_ds not in Overlaped_labels:
                        Overlaped_labels[ovl_ds] = set()
                    Overlaped_labels[ovl_ds].add(eppo_code)

            Overlaped_labels = {k : (len(Overlaped_labels[k]), tuple(Overlaped_labels[k])) for k in Overlaped_labels}

            # print(Overlaped_labels)

            tot_IPM_overlaped_labels = 0

            for k in Overlaped_labels:
                if 'IPM' in k:
                    tot_IPM_overlaped_labels += Overlaped_labels[k][0]

            if args.IPM_only == True:

                ds = 'IPM'

                for key in GT_tot[ds]:
                    if len(Inferred_mapping[ds][key]) != 0:
                        total_mapped[ds] += 1
                        if GT_tot[ds][key] == Inferred_mapping[ds][key][0]:
                            V_positif[ds] += 1
                        else:
                            F_positif[ds] += 1
                            F_positif_list.append((GT_tot[ds][key], f'{key} -> {Inferred_mapping[ds][key][0]}'))
                    
                    if GT_tot[ds][key] != '****NOT FOUND*****':
                        total_with_GT_mapping[ds] += 1
                        if len(Inferred_mapping[ds][key]) != 0:
                            F_negatif[ds] += 1
                        else:
                            V_negatif[ds] += 1

            else: 

                for ds in GT_tot:
                    for key in GT_tot[ds]:
                        if len(Inferred_mapping[ds][key]) != 0:
                            total_mapped[ds] += 1
                            if GT_tot[ds][key] == Inferred_mapping[ds][key][0]:
                                V_positif[ds] += 1
                            else:
                                F_positif[ds] += 1
                                F_positif_list.append((GT_tot[ds][key], f'{key} -> {Inferred_mapping[ds][key][0]}'))
                        
                        if GT_tot[ds][key] != '****NOT FOUND*****':
                            total_with_GT_mapping[ds] += 1
                            if len(Inferred_mapping[ds][key]) != 0:
                                F_negatif[ds] += 1
                            else:
                                V_negatif[ds] += 1


            list_tol.append(tol)
            # list_TPR.append()
            # list_FPR.append()
            if sum(total_with_GT_mapping[ds] for ds in total_with_GT_mapping.keys()) > 0:
                recall = sum(V_positif[ds] for ds in V_positif.keys())/(sum(total_with_GT_mapping[ds] for ds in total_with_GT_mapping.keys()))
            else:
                recall = 0
            list_Recall.append(recall)

            if sum(total_mapped[ds] for ds in total_mapped.keys()) > 0:
                precision = sum(V_positif[ds] for ds in V_positif.keys())/(sum(total_mapped[ds] for ds in total_mapped.keys()))
            else:
                precision = 0
            list_Precision.append(precision)

            if recall > 0:
                F1 = 2*(recall*precision)/(recall + precision)
            else:
                F1 = 0
            list_F1.append(F1)


            if args.IPM_only == False:
                list_IPM_mapped.append(100*len([1 for i in Inferred_mapping['IPM'].keys() if len(Inferred_mapping['IPM'][i]) > 0])/len(Inferred_mapping['IPM']))
                list_images_IPM_mapped.append(100*sum([len(IPM_df[IPM_df['subid'] == int(i)]) for i in Inferred_mapping['IPM'].keys() if len(Inferred_mapping['IPM'][i]) > 0])/len(IPM_df))


            tol += grain
            tol = round(tol,3)

            if tol == 1:
                tol = nearly_one
            # sys.stdout.write(f'Tolerance : {tol}')
            # sys.stdout.flush()

        tot_tol.append(list_tol)
        tot_name.append(mapping_path.split('/')[-1].replace('.json',''))
        # tot_TPR.append(list_TPR)
        # tot_FPR.append(list_FPR)
        tot_Recall.append(list_Recall)
        tot_Precision.append(list_Precision)
        tot_F1.append(list_F1)
        tot_IPM_mapped.append(list_IPM_mapped)
        tot_img_IPM_map.append(list_images_IPM_mapped)


    if args.IPM_only == True:

       # Create a figure
        plt.figure(figsize=(10, 8))

        # Add axis labels and title
        for i, (precision, recall) in enumerate(zip(tot_Precision, tot_Recall)):
            plt.plot(recall, precision, marker='o', linestyle='-', label=f'Precision-Recall {tot_name[i].replace(f"_{tot_name[i].split('_')[-1]}", "").replace("EPPO_mapping_", "")}')

        # Add axis labels and title
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='best')
        plt.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure to a file
        plt.savefig('./output/mapping_metrics/precision_recall_curves_IPM_GT.png', dpi=300)

        # Show the plot
        plt.show()        

    else:

        # Create a figure
        #plt.figure(figsize=(10, 8))
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)


        # Plot Precision-Recall curves
        for i, (ipm_mapped, img_mapped, recall) in enumerate(zip(tot_IPM_mapped, tot_img_IPM_map, tot_Recall)):
            # axes[0].plot(recall, ipm_mapped, marker='o', linestyle='-', label=f'IPM mapped proportion {tot_name[i].replace(f"_{tot_name[i].split('_')[-1]}", "").replace("EPPO_mapping_", "")}')
            axes[0].plot(recall, img_mapped, linestyle='--', label=f'IPM images mapped proportion {tot_name[i].replace(f"_{tot_name[i].split('_')[-1]}", "").replace("EPPO_mapping_", "")}')


        # Add axis labels and title

        axes[0].set_ylabel('IPM mapped')
        axes[0].set_title('Proporiton of subjects mapped')
        axes[0].legend(loc='best')
        axes[0].grid(True)

        for i, (precision, recall) in enumerate(zip(tot_Precision, tot_Recall)):
            axes[1].plot(recall, precision, marker='o', linestyle='-', label=f'Precision-Recall {tot_name[i].replace(f"_{tot_name[i].split('_')[-1]}", "").replace("EPPO_mapping_", "")}')

        # Add axis labels and title
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curves')
        axes[1].legend(loc='best')
        axes[1].grid(True)



        # Adjust layout to prevent overlap
        plt.tight_layout()

        Path('./output/mapping_metrics/').mkdir(exist_ok=True, parents=True)

        # Save the figure to a file
        plt.savefig('./output/mapping_metrics/precision_recall_curves.png', dpi=300)

        # Show the plot
        plt.show()



