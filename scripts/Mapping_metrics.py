import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tolerance', default=0, type=float)
    parser.add_argument('--handmap', action = 'store_true', default = False)
    parser.add_argument('--FullRange', action = 'store_true', default = False)
    parser.add_argument('--MappingPath')
    parser.add_argument('--rescaleTolerance', action = 'store_true', default = False)
    args = parser.parse_args()


    # Reading the ground truth for mapping
    GT_PN = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_PN.csv', sep = ';')
    GT_PV = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_PV.csv', sep = ';')
    GT_Bousset = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_Bousset.csv', sep = ';')
    GT_IPM = pd.read_csv('../data/Extract_EPPO/EPPO_API_results_IPM.csv', sep = ';')

    GT_tot = {'Bousset' : {}, 'PN' : {}, 'PV' : {}, 'IPM' : {}}

    for _, l in GT_PN.iterrows():
        GT_tot['PN'][l['iName'].replace(' ', '_')] = l['EPPO_final']

    for _, l in GT_Bousset.iterrows():
        GT_tot['Bousset'][l['iName']] = l['EPPO_final']

    for _, l in GT_PV.iterrows():
        GT_tot['PV'][l['iName']] = l['EPPO_final']

    for _, l in GT_IPM.iterrows():
        GT_tot['IPM'][str(l['iName'])] = l['EPPO_final']


    with open(args.MappingPath, 'r') as f:
        Inferred_mapping = json.load(f)


    V_positif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
    F_positif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
    F_negatif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
    V_negatif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
    total_mapped = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}
    total_with_GT_mapping = {'Bousset' : 0, 'PN' : 0, 'PV' : 0, 'IPM' : 0}

    F_positif_list = []


    # Keeping only the mapping for tolerance above threshold
    for ds in Inferred_mapping:
        for k in Inferred_mapping[ds]:
            if len(Inferred_mapping[ds][k]) > 0:
                # print(Inferred_mapping[ds][k])
                if args.handmap and ds in ['PN', 'Bousset', 'PV']:
                    pass
                else:
                    if Inferred_mapping[ds][k][1] < args.tolerance:
                        Inferred_mapping[ds][k] = []
                    elif args.rescaleTolerance == True:
                        Inferred_mapping[ds][k][1] = Inferred_mapping[ds][k][1]/100

    # Building an inv_mapping dict wich associates to each EPPO code the set of tuples in the form (Dataset, Subid)
    inv_mapping = {}
    for ds in Inferred_mapping:
        for subid in Inferred_mapping[ds]:
            if len(Inferred_mapping[ds][subid]) > 0:
                if Inferred_mapping[ds][subid][0] not in inv_mapping:
                    inv_mapping[Inferred_mapping[ds][subid][0]] = set()
                inv_mapping[Inferred_mapping[ds][subid][0]].add((ds,subid))


    # # Building an Overlaped_labels dict
    # Overlaped_labels = {}   
    # for eppo_code in inv_mapping:
    #     if len(inv_mapping[eppo_code]) > 1:
    #         ovl_ds = tuple(i[0] for i in inv_mapping[eppo_code])
    #         if ovl_ds not in Overlaped_labels:
    #             Overlaped_labels[ovl_ds] = set()
    #         Overlaped_labels[ovl_ds].add(eppo_code)

    # Overlaped_labels = {k : (len(Overlaped_labels[k]), tuple(Overlaped_labels[k])) for k in Overlaped_labels}

    # print(Overlaped_labels)

    eppo_with_overlap = {}
    for eppo_code in inv_mapping:
        if len(inv_mapping[eppo_code])>1:
            eppo_with_overlap[eppo_code] = inv_mapping[eppo_code]
    
    print(f'With this mapping {len(eppo_with_overlap)}/{len(inv_mapping)} of the created labels contains data from at least 2 sources')
    
    print(eppo_with_overlap)

    potentially_wrong_overlaps = {}
    wrong_overlaps = {}
    for eppo_code in inv_mapping:
        truth_list = []
        true_truth_list = []
        for ds, subid in inv_mapping[eppo_code]:
            if subid in GT_tot[ds]:
                truth_list.append(GT_tot[ds][subid])
                true_truth_list.append(GT_tot[ds][subid])
            else:
                truth_list.append('Out_of_GT')
        if len(set(truth_list)) > 1:
            potentially_wrong_overlaps[eppo_code] = {'Composed_of':inv_mapping[eppo_code], 'GT' : truth_list}
        if len(set(true_truth_list)) > 1:
            wrong_overlaps[eppo_code] = {'Composed_of':inv_mapping[eppo_code], 'GT' : truth_list}

            
    print(f'Of which {len(potentially_wrong_overlaps) - len(wrong_overlaps)}/{len(eppo_with_overlap)} are potentially wrong, but we have no access to the GT')
    print(f'And of which {len(wrong_overlaps)}/{len(eppo_with_overlap)} are wrong for sure')

    # tot_IPM_overlaped_labels = 0

    # for k in Overlaped_labels:
    #     if 'IPM' in k:
    #         tot_IPM_overlaped_labels += Overlaped_labels[k][0]



    # for ds in GT_tot:
    #     for key in GT_tot[ds]:
    #         if GT_tot[ds][key] != '****NOT FOUND*****' and len(Inferred_mapping[ds][key]) == 0:
    #             F_negatif[ds] += 1
    #         elif len(Inferred_mapping[ds][key]) != 0:
    #             if (GT_tot[ds][key] == '****NOT FOUND*****' or (GT_tot[ds][key] != Inferred_mapping[ds][key][0])):
    #                 F_positif[ds] += 1
    #                 F_positif_list.append((GT_tot[ds][key], f'{key} -> {Inferred_mapping[ds][key][0]}'))
    #             elif GT_tot[ds][key] == Inferred_mapping[ds][key][0]:
    #                 V_positif[ds] += 1

    set_of_real_labels = set()
    set_of_mapped_labels = set()

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
                set_of_real_labels.add(key)
                total_with_GT_mapping[ds] += 1
                if len(Inferred_mapping[ds][key]) != 0:
                    F_negatif[ds] += 1
                    if GT_tot[ds][key] == Inferred_mapping[ds][key][0]:
                        set_of_mapped_labels.add(key)

                else:
                    V_negatif[ds] += 1
                

    print(f'False positive : {F_positif}')
    print(f'Liste : {F_positif_list}')

    print(f'Total Mapped : {total_mapped}')
    print(f'True positive : {V_positif}')


    Recall = {}
    Precision = {}
    F1 = {}

    for ds in V_positif:
        # Recall[ds] = V_positif[ds]/(V_positif[ds] + F_negatif[ds])
        # if V_positif[ds] + F_positif[ds] > 0:
        #     Precision[ds] = V_positif[ds]/(V_positif[ds] + F_positif[ds])
        # else:
        #     Precision[ds] = 0


        Precision[ds] = 0
        if V_positif[ds] > 0:
            Precision[ds] = V_positif[ds]/total_mapped[ds]
        Recall[ds] = V_positif[ds]/total_with_GT_mapping[ds]
        if Recall[ds] + Precision[ds] != 0:
            F1[ds] = 2*(Recall[ds]*Precision[ds])/(Recall[ds] + Precision[ds])
        else:
            F1[ds] = 0

    # Recall['total'] = sum(V_positif[ds] for ds in V_positif.keys())/(sum(V_positif[ds] for ds in V_positif.keys()) + sum(F_negatif[ds] for ds in F_negatif.keys()))
    # Precision['total'] = sum(V_positif[ds] for ds in V_positif.keys())/(sum(V_positif[ds] for ds in V_positif.keys()) + sum(F_positif[ds] for ds in F_positif.keys()))
    Precision['total'] = sum(V_positif[ds] for ds in V_positif.keys())/(sum(total_mapped[ds] for ds in total_mapped.keys()))
    Recall['total'] = sum(V_positif[ds] for ds in V_positif.keys())/(sum(total_with_GT_mapping[ds] for ds in total_with_GT_mapping.keys()))

    F1['total'] = 2*(Recall['total']*Precision['total'])/(Recall['total'] + Precision['total'])

    for ds in Recall:
        print(f'{ds} : Recall = {Recall[ds]} | Precision = {Precision[ds]} | F1 = {F1[ds]}')

    IPM_mapped = [1 for i in Inferred_mapping['IPM'].keys() if len(Inferred_mapping['IPM'][i]) > 0]

    IPM_df = pd.read_csv('../data/IPM_data/BD_IPM.csv', sep = ';', encoding = 'utf-8')

    Images_IPM_mapped = [len(IPM_df[IPM_df['subid'] == int(i)]) for i in Inferred_mapping['IPM'].keys() if len(Inferred_mapping['IPM'][i]) > 0]

    print(f"IPM subjects mapped : {len(IPM_mapped)} | {round(100*len(IPM_mapped)/len(Inferred_mapping['IPM']),2)}%")
    print(f"IPM images mapped : {sum(Images_IPM_mapped)} | {round(100*sum(Images_IPM_mapped)/len(IPM_df),2)}%")
    # print(f"Labels with an overlap containing IPM : {tot_IPM_overlaped_labels}")

    print(f'{len(set_of_mapped_labels)}/{len(set_of_real_labels)} labels mapped')

    # if args.FullRange == True:
        
    #     grain = 0.001
    #     tol = args.tolerance

    #     list_tol = []
    #     list_F1 = []
    #     list_Precision = []
    #     list_Recall = []


    #     while tol <= 1:

    #         tol = round(tol, 3)

    #         V_positif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0}
    #         F_positif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0}
    #         F_negatif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0}
    #         V_negatif = {'Bousset' : 0, 'PN' : 0, 'PV' : 0}
    #         total_mapped = {'Bousset' : 0, 'PN' : 0, 'PV' : 0}
    #         total_with_GT_mapping = {'Bousset' : 0, 'PN' : 0, 'PV' : 0}
            

    #         F_positif_list = []

    #         for ds in Inferred_mapping:
    #             for k in Inferred_mapping[ds]:
    #                 if len(Inferred_mapping[ds][k]) > 0:
    #                     # print(Inferred_mapping[ds][k])
    #                     if args.handmap and ds in ['PN', 'Bousset', 'PV']:
    #                         pass
    #                     elif Inferred_mapping[ds][k][1] < tol:
    #                         Inferred_mapping[ds][k] = []

    #         inv_mapping = {}
    #         for ds in Inferred_mapping:
    #             for subid in Inferred_mapping[ds]:
    #                 if len(Inferred_mapping[ds][subid]) > 0:
    #                     if Inferred_mapping[ds][subid][0] not in inv_mapping:
    #                         inv_mapping[Inferred_mapping[ds][subid][0]] = set()
    #                     inv_mapping[Inferred_mapping[ds][subid][0]].add(ds)

    #         Overlaped_labels = {}   

    #         for eppo_code in inv_mapping:
    #             if len(inv_mapping[eppo_code]) > 1:
    #                 ovl_ds = tuple(inv_mapping[eppo_code])
    #                 if ovl_ds not in Overlaped_labels:
    #                     Overlaped_labels[ovl_ds] = set()
    #                 Overlaped_labels[ovl_ds].add(eppo_code)

    #         Overlaped_labels = {k : (len(Overlaped_labels[k]), tuple(Overlaped_labels[k])) for k in Overlaped_labels}

    #         # print(Overlaped_labels)

    #         tot_IPM_overlaped_labels = 0

    #         for k in Overlaped_labels:
    #             if 'IPM' in k:
    #                 tot_IPM_overlaped_labels += Overlaped_labels[k][0]


    #         # for ds in GT_tot:
    #         #     for key in GT_tot[ds]:
    #         #         if GT_tot[ds][key] != '****NOT FOUND*****' and len(Inferred_mapping[ds][key]) == 0:
    #         #             F_negatif[ds] += 1
    #         #         elif len(Inferred_mapping[ds][key]) != 0:
    #         #             if (GT_tot[ds][key] == '****NOT FOUND*****' or (GT_tot[ds][key] != Inferred_mapping[ds][key][0])):
    #         #                 F_positif[ds] += 1
    #         #                 F_positif_list.append((GT_tot[ds][key], f'{key} -> {Inferred_mapping[ds][key][0]}'))
    #         #             elif GT_tot[ds][key] == Inferred_mapping[ds][key][0]:
    #         #                 V_positif[ds] += 1

    #         for ds in GT_tot:
    #             for key in GT_tot[ds]:
    #                 if len(Inferred_mapping[ds][key]) != 0:
    #                     total_mapped[ds] += 1
    #                     if GT_tot[ds][key] == Inferred_mapping[ds][key][0]:
    #                         V_positif[ds] += 1
    #                     else:
    #                         F_positif[ds] += 1
    #                         F_positif_list.append((GT_tot[ds][key], f'{key} -> {Inferred_mapping[ds][key][0]}'))
                    
    #                 if GT_tot[ds][key] != '****NOT FOUND*****':
    #                     total_with_GT_mapping[ds] += 1
    #                     if len(Inferred_mapping[ds][key]) != 0:
    #                         F_negatif[ds] += 1
    #                     else:
    #                         V_negatif[ds] += 1

    #         # print(f'False positive : {F_positif}')
    #         # print(f'Liste : {F_positif_list}')

    #         Recall = {}
    #         Precision = {}
    #         F1 = {}

    #         for ds in V_positif:
    #         #     Recall[ds] = V_positif[ds]/(V_positif[ds] + F_negatif[ds])
    #         #     if V_positif[ds] + F_positif[ds] > 0:
    #         #         Precision[ds] = V_positif[ds]/(V_positif[ds] + F_positif[ds])
    #         #     else:
    #         #         Precision[ds] = 0
    #         #     if Recall[ds] + Precision[ds] != 0:
    #         #         F1[ds] = 2*(Recall[ds]*Precision[ds])/(Recall[ds] + Precision[ds])
    #         #     else:
    #         #         F1[ds] = 0

    #             Precision[ds] = 0
    #             if V_positif[ds] > 0:
    #                 Precision[ds] = V_positif[ds]/total_mapped[ds]
    #             Recall[ds] = V_positif[ds]/total_with_GT_mapping[ds]
    #             if Recall[ds] + Precision[ds] != 0:
    #                 F1[ds] = 2*(Recall[ds]*Precision[ds])/(Recall[ds] + Precision[ds])
    #             else:
    #                 F1[ds] = 0

    #         # Recall['total'] = sum(V_positif[ds] for ds in V_positif.keys())/(sum(V_positif[ds] for ds in V_positif.keys()) + sum(F_negatif[ds] for ds in F_negatif.keys()))
    #         # Precision['total'] = sum(V_positif[ds] for ds in V_positif.keys())/(sum(V_positif[ds] for ds in V_positif.keys()) + sum(F_positif[ds] for ds in F_positif.keys()))
    #         Precision['total'] = sum(V_positif[ds] for ds in V_positif.keys())/(sum(total_mapped[ds] for ds in total_mapped.keys()))
    #         Recall['total'] = sum(V_positif[ds] for ds in V_positif.keys())/(sum(total_with_GT_mapping[ds] for ds in total_with_GT_mapping.keys()))
    #         F1['total'] = 2*(Recall['total']*Precision['total'])/(Recall['total'] + Precision['total'])


    #         # for ds in F1:
    #         #     print(f'{ds} : Recall = {Recall[ds]} | Precision = {Precision[ds]} | F1 = {F1[ds]}')

    #         IPM_mapped = [1 for i in Inferred_mapping['IPM'].keys() if len(Inferred_mapping['IPM'][i]) > 0]


    #         list_tol.append(tol)
    #         list_F1.append(F1['total'])
    #         list_Precision.append(Precision['total'])
    #         list_Recall.append(Recall['total'])

    #         tol += grain
        
    #     df = pd.DataFrame({'tolerance' : list_tol,
    #                        'F1' : list_F1,
    #                        'precision' : list_Precision,
    #                        'recall' : list_Recall})
        
    #     df.to_csv(f"./output/mapping_metrics/Metrics_{args.MappingPath.split('/')[-1].replace('.json', '')}.csv")

        
    #     # Find the index of the maximum F1 score
    #     max_f1_index = df['F1'].idxmax()
    #     max_f1 = df['F1'][max_f1_index]
    #     best_tolerance = df['tolerance'][max_f1_index]

    #     # Create a figure and two subplots
    #     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)

    #     # Plot F1 score
    #     axes[0].plot(df['tolerance'], df['F1'], marker='o', linestyle='-', color='b', label='F1 Score')
    #     axes[0].axhline(y=max_f1, color='gray', linestyle='--', linewidth=1)
    #     axes[0].axvline(x=best_tolerance, color='gray', linestyle='--', linewidth=1)
    #     axes[0].annotate(f'Best F1: {max_f1:.2f}', xy=(0.05, max_f1), xytext=(best_tolerance + 0.05, max_f1),
    #                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    #     axes[0].annotate(f'Tolerance: {best_tolerance:.2f}', xy=(best_tolerance, 0.6), xytext=(best_tolerance, max_f1 + 0.05),
    #                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)        
    #     axes[0].set_ylabel('F1 Score')
    #     axes[0].legend(loc='best')
    #     axes[0].grid(True)
    #     axes[0].set_title('Tolerance vs. F1 Score, Precision, and Recall')

    #     # Plot precision and recall on the same subplot
    #     axes[1].plot(df['tolerance'], df['precision'], marker='o', linestyle='-', color='g', label='Precision')
    #     axes[1].plot(df['tolerance'], df['recall'], marker='o', linestyle='-', color='r', label='Recall')
    #     axes[1].axvline(x=best_tolerance, color='gray', linestyle='--', linewidth=1)
    #     axes[1].set_xlabel('Tolerance')
    #     axes[1].set_ylabel('Value')
    #     axes[1].legend(loc='best')
    #     axes[1].grid(True)


    #     # Adjust layout to prevent overlap
    #     plt.tight_layout()

    #     # Save the figure to a file
    #     plt.savefig(f"output/mapping_metrics/tolerance_vs_metrics{args.MappingPath.split('/')[-1].replace('.json', '')}.png", dpi=300)

    #     # Show the plot
    #     plt.show()






