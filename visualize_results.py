import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import regex as re
import seaborn as sns
from termcolor import colored

colors = ['#332288', '#88CCEE', '#117733', "#e28743", '#DDCC77', '#44AA99', '#CC6677', '#882255', '#AA4499']


def get_noc(iou_arr, iou_thr, max_clicks):
    vals = iou_arr >= iou_thr
    return np.argmax(vals) + 1 if np.any(vals) else max_clicks


def improve_label(og_label, abbreviations):
    new_label = og_label.replace('_', ' ')
    new_label = new_label.capitalize()
    if abbreviations:
        new_label = new_label.replace('Arteria mesenterica superior', 'SMA')
        new_label = new_label.replace('Common bile duct', 'CBD')
        new_label = new_label.replace('Gastroduodenalis', 'GA')
        new_label = new_label.replace('Pancreatic duct', 'PD')
    else:
        new_label = new_label.replace('Arteria mesenterica superior', 'Superior mesenteric artery')
    new_label = new_label.replace('Tumour', 'Tumor')

    return new_label


def load_data_to_plot(data_dict):
    iou_dict = {}

    for model in os.listdir(experiments_path):
        if model_type not in model:
            continue
        model_path = experiments_path + model + "/"
        for model_try in os.listdir(model_path):
            model_try_path = model_path + model_try + "/"
            evaluation_path = model_try_path + "evaluation_logs/test_set/others/"
            if os.path.exists(evaluation_path):
                for epoch in os.listdir(evaluation_path):
                    plots_path = evaluation_path + epoch + "/" + epoch + "/"
                    if os.path.exists(plots_path):
                        for file in os.listdir(plots_path):
                            for k in data_dict:
                                if ".pickle" in file:
                                    if data_dict[k]['try'] in plots_path \
                                            and 'epoch-' + str(data_dict[k]['epoch']) in plots_path \
                                            and k in plots_path and '1' + str(data_dict[k]['epoch']) not in plots_path \
                                            and '2' + str(data_dict[k]['epoch']) not in plots_path:
                                        print(file)
                                        '''
                                        print(colored(f"Key: {k}", "green"))
                                        print(f"Loading {plots_path + file}")
                                        with open(plots_path + file, "rb") as f:
                                            label = k.replace("_", " ")
                                            label = label.capitalize()
                                            label = label.replace("Arteria mesenterica superior",
                                                                  "Superior mesenteric artery")
                                            iou_dict[label] = np.array(pickle.load(f)['all_ious'])
                                        '''

    return iou_dict


def boundary_progression_plot(structures_dict, lw=0.5, font_size=12):
    boundary_hu_per_structure = {}
    selected_colors = colors[0:len(structures_dict)]

    for key in structures_dict:
        boundary_hu_per_structure[key] = []

    with open('structure_boundary_hu.json', 'r') as f:
        data = json.load(f)

    for key in data:
        print(key)
        for structure in data[key]:
            boundary_hu_per_structure[structure].append(data[key][structure][0])

    print(np.shape(np.array(boundary_hu_per_structure['aorta'])))

    f, ax = plt.subplots()
    ax.vlines(9, -250, 250, linewidth=lw, linestyles='dashed', color='k')
    for structure, color in zip(boundary_hu_per_structure, selected_colors):
        a = np.array(boundary_hu_per_structure[structure])
        a[a == 40000] = np.nan
        ax.plot(np.nanmean(a, axis=0) - np.nanmean(a, axis=0)[9], linewidth=lw,
                label=improve_label(structure, abbreviations=True), color=color)

    plt.ylim([-200, 110])
    plt.xlim([0, 19])
    plt.xticks([0, 4, 9, 14, 19], ['', '-5', '0', '5', '10'])
    plt.xlabel('Distance from boundary [pixels]', fontsize=font_size)
    plt.ylabel('Relative intensity w.r.t. the boundary [HU]', fontsize=font_size)
    plt.legend(prop={'size': font_size})
    plt.grid(visible=True, which='both', axis='both')
    plt.show()


def in_out_plot(structures_dict, save=False, font_size=12):
    strcts = [key for key in structures_dict]
    print("Loading pickle...")
    df = pd.read_pickle('in_out_df.pkl')
    print("Loaded pickle.")

    boxplot = sns.boxplot(data=df, x='structure', y='values', hue='in', palette=["#e28743", "#eab676"], showfliers=False)
    boxplot.set_xlabel('Structure', fontsize=font_size)
    boxplot.set_ylabel('Intensity [HU]', fontsize=font_size)
    boxplot.set_xticklabels([improve_label(structure, True) for structure in strcts])

    handles, _ = boxplot.get_legend_handles_labels()
    boxplot.legend(handles, ['Inside', 'Outside'])
    if save:
        plt.savefig(f"epoch_evaluations/in_out_boxplot.pdf", dpi=300)
    else:
        plt.show()


def single_boxplot(data_dict, label, n_clicks, save=False, font_size=12):
    ious_array = data_dict[label][:, :n_clicks]

    f, ax = plt.subplots()
    ax.boxplot(ious_array, showfliers=False, medianprops=dict(color="#e28743"))
    plt.xlabel("Amount of clicks", fontsize=font_size)
    plt.ylabel("mIoU", fontsize=font_size)
    plt.ylim([0, 1])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_{label}_single_boxplot.pdf", dpi=300)
    else:
        plt.show()


def single_std_plot(data_dict, label, n_clicks, lw=0.5, save=False, font_size=12):
    ious_array = data_dict[label][:, :n_clicks]
    mean = np.mean(ious_array, axis=0)
    std = np.std(ious_array, axis=0)

    f, ax = plt.subplots()
    ax.plot(range(1, n_clicks + 1), mean, color="#e28743", linewidth=lw)
    ax.plot(range(1, n_clicks + 1), mean - std, color="#eab676", linewidth=lw)
    ax.plot(range(1, n_clicks + 1), mean + std, color="#eab676", linewidth=lw)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.fill_between(range(1, n_clicks + 1), mean - std, mean + std, color="#f3cfb4", alpha=0.5)
    plt.xlabel("Amount of clicks", fontsize=font_size)
    plt.ylabel("mIoU", fontsize=font_size)
    plt.ylim([0, 1])
    plt.xlim([1, n_clicks])
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_{label}_individual_std.pdf", dpi=300)
    else:
        plt.show()


def single_noc_histogram(data_dict, label, n_clicks, noc_thr, lw=0.5, save=False, font_size=12):
    ious_array = data_dict[label][:, :n_clicks]
    nocs = []

    for i in range(ious_array.shape[0]):
        noc = get_noc(ious_array[i], iou_thr=0.8, max_clicks=n_clicks)
        nocs.append(noc)

    print(f"NoC@{int(noc_thr * 100)}: {np.mean(nocs)}")

    # noc_list, over_max_list = compute_noc_metric(ious_array, [noc_thr], max_clicks=n_clicks)
    # print(noc_list)

    f, ax = plt.subplots()
    ax.hist(nocs, bins=n_clicks, histtype='step', color="#e28743", lw=lw)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Amount of clicks", fontsize=font_size)
    plt.xlim([0, n_clicks + 1])
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_{label}_individual_histogram.pdf", dpi=300)
    else:
        plt.show()


def combined_miou_plot(data_dict, n_clicks, lw=0.5, save=False, font_size=12):
    f, ax = plt.subplots()
    # colors = ['#8c510a','#d8b365','#f6e8c3','#c7eae5','#5ab4ac','#01665e']
    selected_colors = colors[0:len(data_dict)]
    for k, color in zip(data_dict, selected_colors):
        ax.plot(range(1, n_clicks + 1), np.mean(data_dict[k][:, :n_clicks], axis=0), linewidth=lw, label=k, color=color)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Amount of clicks", fontsize=font_size)
    plt.ylabel("mIoU", fontsize=font_size)
    plt.ylim([0, 1])
    plt.xlim([1, n_clicks])
    plt.legend(prop={'size': font_size})
    plt.grid(visible=True, which='both', axis='both')
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_mIoU_combined.pdf", dpi=300)
    else:
        plt.show()


def combined_noc_histogram(data_dict, n_clicks, noc_thr, lw=0.5, save=False, font_size=12):
    f, ax = plt.subplots()
    selected_colors = colors[0:len(data_dict)]

    for k, color in zip(data_dict, selected_colors):
        nocs = []
        ious_array = data_dict[k][:, :n_clicks]

        for i in range(ious_array.shape[0]):
            noc = get_noc(ious_array[i], noc_thr, n_clicks)
            nocs.append(noc)

        ax.hist(nocs, bins=n_clicks, histtype='step', label=k, color=color, density=True, lw=lw)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel(f"Amount of clicks for NoC of {noc_thr}", fontsize=font_size)
    plt.ylim([0, 1])
    plt.xlim([0, n_clicks + 1])
    plt.ylabel("Density", fontsize=font_size)
    plt.legend()
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_nocs_combined.pdf", dpi=300)
    else:
        plt.show()


def plot_avg_mask_influence(data_dict, structures_dict, noc_thr, lw=0.5, save=False, font_size=12):
    avg_mask_list = []
    avg_nocs = []
    keys = []

    for k in data_dict:
        keys.append(k)
        # return to the original labels
        label = k.lower()
        label = label.replace("superior mesenteric artery", "arteria mesenterica superior")
        label = label.replace(" ", "_")

        nocs = []
        avg_mask_list.append(structures_dict[label]['avg_mask'])
        for i in range(data_dict[k].shape[0]):
            noc = get_noc(data_dict[k][i], noc_thr, data_dict[k].shape[1])
            nocs.append(noc)
        avg_nocs.append(np.mean(np.array(nocs)))

    f, ax = plt.subplots()
    ax.plot(avg_mask_list, avg_nocs, 'x', color="#e28743", lw=lw)
    for i in range(len(avg_mask_list)):
        if "Tumour" in keys[i]:
            plt.text(avg_mask_list[i] + 70, avg_nocs[i] - 1, keys[i], fontsize=font_size)
        else:
            plt.text(avg_mask_list[i] + 70, avg_nocs[i], keys[i], fontsize=font_size)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.ylim([0, math.ceil(np.max(avg_nocs)) + 1])
    plt.xlabel("Average mask size", fontsize=font_size)
    plt.ylabel(f"NoC@{int(100 * noc_thr)}", fontsize=font_size)
    plt.xlim([0, 3500])
    plt.grid(visible=True, which='both', axis='both')
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_noc_vs_mask_size.pdf", dpi=300)
    else:
        plt.show()


def combined_delta_relative(data_dict, n_clicks, lw=0.5, save=False, font_size=12):
    selected_colors = colors[0:len(data_dict)]

    f, ax = plt.subplots()
    for k, color in zip(data_dict, selected_colors):
        ious_array = data_dict[k][:, :n_clicks]
        improvement = np.mean(ious_array - np.mean(ious_array[:, 0]), axis=0)
        ax.plot(range(1, n_clicks + 1), improvement, label=k, color=color, lw=lw)
        print(f"{improvement[9]:.2f} improvement after 10 clicks for {k}")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Number of clicks", fontsize=font_size)
    plt.ylabel(f"Increase in mIoU", fontsize=font_size)
    plt.xlim([1, n_clicks])
    plt.legend(prop={'size': font_size})
    plt.grid(visible=True, which='both', axis='both')
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_delta_relative.pdf", dpi=300)
    else:
        plt.show()


def combined_delta_absolute(data_dict, n_clicks, lw=0.5, save=False, font_size=12):
    selected_colors = colors[0:len(data_dict)]

    f, ax = plt.subplots()
    plt.hlines(0, 1, n_clicks, colors=['k'], lw=lw)
    for k, color in zip(data_dict, selected_colors):
        ious_array = data_dict[k][:, :n_clicks]
        ax.plot(range(1, n_clicks), np.diff(np.mean(ious_array, axis=0)), linewidth=lw, label=k, color=color)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Number of clicks", fontsize=font_size)
    plt.ylabel(f"Delta", fontsize=font_size)
    plt.xlim([1, n_clicks - 1])
    plt.legend()
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_delta_absolute.pdf", dpi=300)
    else:
        plt.show()


def create_latex_table(structures_dict):
    latex_table_string = ''

    for key in structures_dict:
        print(key)
        label = improve_label(key, abbreviations=True)
        latex_table_string = latex_table_string + label + " & "

        for model in os.listdir(experiments_path):
            if key not in model or "hrnet" in model:
                continue
            model_path = experiments_path + model + "/"
            for model_try in os.listdir(model_path):
                if structures_dict[key]['try'] not in model_try:
                    continue
                model_try_path = model_path + model_try + "/"
                evaluation_path = model_try_path + "evaluation_logs/test_set/others/"
                if os.path.exists(evaluation_path):
                    for epoch in os.listdir(evaluation_path):
                        epoch_path = evaluation_path + epoch + "/" + epoch + "/"
                        if os.path.exists(epoch_path):
                            for file in os.listdir(epoch_path):
                                if ".txt" in file:
                                    with open(epoch_path + file, "r") as f:
                                        text = f.read()
                                    values = re.findall(r"([0-9]{0,2}\.[0-9]{2})(?!\|)(?![0-9])", text)

                                    # add to latex table
                                    n_values = len(values)
                                    for i, value in enumerate(values):
                                        if i + 1 == n_values:
                                            latex_table_string = latex_table_string + str(value) + r" \\" + "\n"
                                        else:
                                            latex_table_string = latex_table_string + str(value) + " & "

                                    print(latex_table_string)


if __name__ == "__main__":
    fs = 13
    linew = 1

    model_type = 'cdnet'

    if model_type == 'segformer':
        experiments_path = "./experiments/focalclick/"
        structures = {'aorta': {'try': '000', 'epoch': 49, 'avg_mask': 3000},
                      'arteria_mesenterica_superior': {'try': '000', 'epoch': 39, 'avg_mask': 252},
                      'common_bile_duct': {'try': '000', 'epoch': 169, 'avg_mask': 501},
                      'gastroduodenalis': {'try': '000', 'epoch': 69, 'avg_mask': 61},
                      'pancreas': {'try': '000', 'epoch': 49, 'avg_mask': 979},
                      'pancreatic_duct': {'try': '000', 'epoch': 79, 'avg_mask': 162},
                      'tumour': {'try': '000', 'epoch': 109, 'avg_mask': 75}}
    elif model_type == 'cdnet':
        experiments_path = "./experiments/cdnet/"
        structures = {'aorta': {'try': '000', 'epoch': 19, 'avg_mask': 3000},
                      'arteria_mesenterica_superior': {'try': '000', 'epoch': 39, 'avg_mask': 252},
                      'common_bile_duct': {'try': '000', 'epoch': 29, 'avg_mask': 501},
                      'gastroduodenalis': {'try': '000', 'epoch': 59, 'avg_mask': 61},
                      'pancreas': {'try': '000', 'epoch': 19, 'avg_mask': 979},
                      'pancreatic_duct': {'try': '000', 'epoch': 9, 'avg_mask': 162},
                      'tumour': {'try': '000', 'epoch': 19, 'avg_mask': 75}}
    else:
        raise ValueError(f"Expected model_type cdnet or segformer, not {model_type}")

    # boundary_progression_plot(structures, lw=linew)
    # in_out_plot(structures, save=True, font_size=12)  # only do this one if you're really sure, since it takes ages

    # create_latex_table(structures)
    
    data = load_data_to_plot(structures)

    # plot_avg_mask_influence(data, structures, noc_thr=0.8, save=True, font_size=fs)
    # single_noc_histogram(data, 'Pancreas', n_clicks=50, noc_thr=0.8, lw=0.5, save=True, font_size=fs)
    # combined_miou_plot(data, n_clicks=20, lw=linew, font_size=fs, save=True)
    # combined_noc_histogram(data, n_clicks=50, noc_thr=0.8)
    # combined_noc_histogram(data, n_clicks=30, noc_thr=0.8)
    # combined_delta_absolute(data, n_clicks=50)
    # combined_delta_absolute(data, n_clicks=20, save=False)
