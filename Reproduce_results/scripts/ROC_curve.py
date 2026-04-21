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
    parser.add_argument('--test-GT', type=list_of_strings, default=['PN','PV','Bousset','IPM'])
    args = parser.parse_args()


    GT = {}
    for ds in args.test_GT:
        GT[ds] = pd.read_csv(f'../data/Extract_EPPO/EPPO_API_results_{ds}.csv', sep = ';')

    # GT_PN = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_PN.csv', sep = ';')
    # GT_PV = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_PV.csv', sep = ';')
    # GT_Bousset = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_Bousset.csv', sep = ';')
    # GT_IPM = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_IPM.csv', sep = ';')
    # GT_Bousset2 = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_Bousset2.csv', sep = ';')




    # GT_tot = {'Bousset' : {}, 'PN' : {}, 'PV' : {}, 'IPM' : {}, 'Bousset2' : {}}
    GT_tot = {ds:{} for ds in args.test_GT}

    for ds in GT:

        for _, l in GT[ds].iterrows():
            GT_tot[ds][l['iName']] = l['EPPO_final']

    # for _, l in GT_PV.iterrows():
    #     GT_tot['PV'][l['iName']] = l['EPPO_final']

    # for _, l in GT_IPM.iterrows():
    #     GT_tot['IPM'][str(l['iName'])] = l['EPPO_final']

    # for _, l in GT_Bousset2.iterrows():
    #     GT_tot['Bousset2'][str(l['iName'])] = l['EPPO_final']

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


        if "relative" in mapping_path:
            tol = 0
            grain = 0.001
        # if "Pllama" in mapping_path:
        #     tol = 0.999
        #     grain = 0.0001
        else:
            tol = 0.75
            grain = 0.005

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

        list_sims = []
        for ds in Inferred_mapping:
            for k in Inferred_mapping[ds]:
                if len(Inferred_mapping[ds][k]) > 0:
                    list_sims.append(Inferred_mapping[ds][k][1])


        if "relative" in mapping_path:
            nearly_one = max(list_sims)
        # if "Pllama" in mapping_path:
        #     nearly_one = 0.999999
        else:
            nearly_one = 0.99999
        

        while tol <= nearly_one:

            print(tol)

            V_positif = {}
            F_positif = {}
            F_negatif = {}
            V_negatif = {}
            total_mapped = {}
            total_with_GT_mapping = {}

            for ds in GT_tot:


                V_positif[ds] = 0
                F_positif[ds] = 0
                F_negatif[ds] = 0
                V_negatif[ds] = 0
                total_mapped[ds] = 0
                total_with_GT_mapping[ds] = 0


            # V_positif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
            # F_positif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
            # F_negatif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
            # V_negatif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
            # total_mapped = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
            # total_with_GT_mapping = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}


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
            if "relative" in mapping_path:
                tol = round(tol,5)
            # if "Pllama" in mapping_path:
            #     tol = round(tol,7)
            else:
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
            axes[1].plot(recall, precision, marker='o', linestyle='-', label=tot_name[i].replace(f"_{tot_name[i].split('_')[-1]}", "").replace("EPPO_mapping_", "").replace('_0.75_EPPOtoDS', ''))

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

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=False)

        # Plot Precision curves
        for i, (precision, real_recall) in enumerate(zip(tot_Precision, tot_img_IPM_map)):
            label = tot_name[i].replace(f"_{tot_name[i].split('_')[-1]}", "").replace("EPPO_mapping_", "").replace('_0.75_EPPOtoDS', '')
            axes[0].plot(real_recall, precision, marker='x', label=f'Precision {label}')

        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision against Images mapped from IPM')
        axes[0].legend(loc='best')
        axes[0].grid(True)

        # Plot F1 curves and highlight the top 3 maximum points
        for i, (F1, real_recall, tol) in enumerate(zip(tot_F1, tot_img_IPM_map, tot_tol)):
            label = tot_name[i].replace(f"_{tot_name[i].split('_')[-1]}", "").replace("EPPO_mapping_", "").replace('_0.75_EPPOtoDS', '')
            
            # Plot the F1 curve
            axes[1].plot(tol, F1, linestyle='-', label=f'F1 score {label}', linewidth=3.0)
            
            # # Find the indices of the top 3 F1 scores
            # top_3_indices = np.argsort(F1)[-3:][::-1]  # Sort to get top 3 indices, in descending order
            # top_3_indices = [np.argmax(F1)]
            
            # # Plot the top 3 F1 points in red
            # for j,idx in enumerate(top_3_indices):
            #     max_tol = tol[idx]
            #     max_F1 = F1[idx]
            #     max_recall = real_recall[idx]
            #     axes[1].plot(max_tol, max_F1, 'ro')  # Red 'o' marker for max F1 score
            #     # Offset for annotation to prevent overlap
            #     xytext_offset = (0,0 - 10*j)  # Adjust the vertical offset
            #     axes[1].annotate(f'Tol : {max_tol}, F1 : {max_F1:4f}', xy=(max_tol, max_F1), xytext=xytext_offset,
            #                     textcoords='offset points', ha='center', color='black')
            #     # Add vertical lines at max F1 points across both subplots
            #     axes[0].axvline(x=max_tol, color='red', linestyle='--')
            #     axes[1].axvline(x=max_tol, color='red', linestyle='--')

            idx_best_F1 = np.argmax(F1)


            with open('./output/mapping_metrics/Mapping_choice_summary.txt', 'w') as f:
                f.write(f'\nFor mapping : {args.list_mappings[i]}\n')

                f.write(f'Mapping with best F1 : Tolerance = {tol[idx_best_F1]}, Precision = {precision[idx_best_F1]}, Recall = {recall[idx_best_F1]}')
                with open(mapping_path, 'r') as f2:
                    Inferred_mapping = json.load(f2)
                for ds in Inferred_mapping:
                    for k in Inferred_mapping[ds]:
                        if len(Inferred_mapping[ds][k]) > 0:
                            # print(Inferred_mapping[ds][k])
                            if Inferred_mapping[ds][k][1] < tol[idx_best_F1]:
                                Inferred_mapping[ds][k] = []
                with open('./output/mapping_metrics/'+ mapping_path.split('/')[-1].replace('.json', '') + '_WithBestF1.json', 'w') as f3:
                    json.dump(Inferred_mapping, f3, indent=2)



        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('F1')
        axes[1].set_title('F1 against threshold value')
        axes[1].legend(loc='lower left')
        axes[1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the figure to a file
        plt.savefig('./output/mapping_metrics/Mapping_choice_curves.png', dpi=300)

        # Show the plot
        plt.show()


