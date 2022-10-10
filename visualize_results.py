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
from isegm.inference.utils import compute_noc_metric


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


def load_data_to_plot(data_dict, exp_path, model_type, ious=True):
    iou_dict = {}
    print(colored(f"{model_type}", 'yellow'))

    for model in os.listdir(exp_path):
        if model_type not in model:
            continue
        model_path = os.path.join(exp_path, model)
        for model_try in os.listdir(model_path):
            model_try_path = os.path.join(model_path, model_try)
            evaluation_path = os.path.join(model_try_path, "evaluation_logs/test_set/others/")
            if os.path.exists(evaluation_path):
                for epoch in os.listdir(evaluation_path):
                    if model_type == 'hrnet':
                        plots_path = os.path.join(evaluation_path, epoch, "plots")
                    else:
                        plots_path = os.path.join(evaluation_path, epoch, epoch)
                    if os.path.exists(plots_path):
                        for file in os.listdir(plots_path):
                            for k in data_dict:
                                if ".pickle" in file:
                                    if data_dict[k]['try'] in plots_path \
                                            and 'epoch-' + str(data_dict[k]['epoch']) in plots_path \
                                            and k in plots_path and '1' + str(data_dict[k]['epoch']) not in plots_path \
                                            and '2' + str(data_dict[k]['epoch']) not in plots_path:
                                        print('\t' + colored(f"Key: {k}", "green"))
                                        print(f"\tLoading {os.path.join(plots_path, file)}")
                                        with open(os.path.join(plots_path, file), "rb") as f:
                                            if ious:
                                                iou_dict[k] = np.array(pickle.load(f)['all_ious'])
                                            else:
                                                iou_dict[k] = np.array(pickle.load(f)['all_dices'])

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

    boxplot = sns.boxplot(data=df, x='structure', y='values', hue='in', palette=["#e28743", "#eab676"],
                          showfliers=False)
    boxplot.set_xlabel('Structure', fontsize=font_size)
    boxplot.set_ylabel('Intensity [HU]', fontsize=font_size)
    boxplot.set_xticklabels([improve_label(structure, True) for structure in strcts])

    handles, _ = boxplot.get_legend_handles_labels()
    boxplot.legend(handles, ['Inside', 'Outside'])
    if save:
        plt.savefig(f"epoch_evaluations/in_out_boxplot.pdf", dpi=300)
    else:
        plt.show()


def single_boxplot(data_dict, model_type, label, n_clicks, save=False, font_size=12):
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


def single_std_plot(data_dict, model_type, label, n_clicks, lw=0.5, save=False, font_size=12):
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


def single_noc_histogram(data_dict, model_type, label, n_clicks, noc_thr, lw=0.5, save=False, font_size=12):
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
    plt.xlabel(f"NoC@{int(noc_thr * 100)}", fontsize=font_size)
    plt.ylabel("Frequency")
    plt.xlim([0, n_clicks + 1])
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_{label}_individual_histogram.pdf", dpi=300)
    else:
        plt.show()


def all_models_noc_histogram(label, n_clicks, noc_thr, save):
    selected_colors = colors[:len(data_all_models)]

    f, ax = plt.subplots()
    for data_dict, color in zip(data_all_models, selected_colors):
        ious_array = data_dict[label][:, :n_clicks]
        nocs = []

        # for each model, calculate the nocs of each sample
        for i in range(ious_array.shape[0]):
            noc = get_noc(ious_array[i], iou_thr=noc_thr, max_clicks=n_clicks)
            nocs.append(noc)

        ax.hist(nocs, bins=n_clicks, histtype='step', color=color, lw=linewidth, label=improve_label(label, True))
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

    plt.xlabel(f"NoC@{int(noc_thr * 100)}", fontsize=fs)
    plt.ylabel("Frequency", fontsize=fs)
    plt.xlim([0, n_clicks + 1])
    plt.legend(['FocalClick', 'CDNet', 'HRNet'], loc='upper center')
    if save:
        plt.savefig(f"epoch_evaluations/histograms/all_models_histogram_{label}.pdf", dpi=300)
    else:
        plt.show()


def combined_miou_plot(data_dict, model_type, n_clicks, lw=0.5, save=False, font_size=12):
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


def combined_noc_histogram(data_dict, model_type, n_clicks, noc_thr, lw=0.5, save=False, font_size=12):
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


def all_models_mask_influence(noc_thr, n_clicks, save=False):
    selected_colors = colors[:len(data_all_models)]
    selected_styles = linestyles[:len(data_all_models)]

    f, ax = plt.subplots()
    ax2 = ax.twiny()
    xtick_locs = []
    xtick_labels = []
    coeffs = []  # will be three values

    for data_dict, color, l_style, m_style, model_type in zip(data_all_models, selected_colors,
                                                              selected_styles, ['x', 'o', '+'],
                                                              ['FocalClick', 'CDNet', 'RITM']):
        mask_sizes = []
        nocs = []

        ax.plot(-10, -10, marker=m_style, label=model_type, linestyle=l_style, color=color)

        for key in data_dict:
            noc, _ = compute_noc_metric(data_dict[key], [noc_thr], n_clicks)
            nocs.append(noc[0])
            mask_size = structures_hrnet[key]['avg_mask']
            mask_sizes.append(mask_size)

            ax.plot(mask_size, noc, m_style, color=color, lw=linewidth)

        # add linear fits
        coeff = np.squeeze(np.polyfit(mask_sizes, nocs, 1))
        coeffs.append(coeff[0])
        poly_fn = np.poly1d(coeff)
        print(mask_sizes)
        popt, _ = scipy.optimize.curve_fit(lambda t, a, b: a / t + b, mask_sizes, nocs, p0=(22, 28))
        mask_sizes.sort()
        ax.plot(mask_sizes, poly_fn(mask_sizes), color=color, lw=linewidth, linestyle=l_style)

        # add inverse fits
        print(f"Mask sizes: {mask_sizes}\nnocs: {nocs}")

        print(f"Exponential fit: {popt}")
        ax.plot(mask_sizes, popt[0] / mask_sizes + popt[1])

    for key in structures_hrnet:
        xtick_locs.append(structures_hrnet[key]['avg_mask'])
        if key == 'tumour':
            xtick_labels.append('       ' + improve_label(key, True))
        else:
            xtick_labels.append(improve_label(key, True))

    ax.set_xlabel("Average mask size", fontsize=fs)
    ax.set_ylabel(f"NoC@{int(100 * noc_thr)}", fontsize=fs)
    ax2.set_xticks(xtick_locs)
    ax2.set_xticklabels(xtick_labels, rotation=90, fontsize=fs)
    plt.ylim([0, n_clicks])
    ax.set_xlim([0, 3020])
    ax2.set_xlim([0, 3020])
    ax.grid(visible=True, which='both', axis='both')
    # ax2.grid(visible=True, which='both', axis='both')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    # create line to tumor label
    avg_tumour_mask = structures_hrnet['tumour']['avg_mask']
    ax.plot([avg_tumour_mask, avg_tumour_mask], [n_clicks, n_clicks * 1.15], color='k', clip_on=False, lw=0.5)
    plt.tight_layout()
    ax.legend(loc=(2250 / 3010, 35 / 50))

    if save:
        plt.savefig(f"epoch_evaluations/all_models_noc_vs_mask_size_{metric}.png", dpi=300, transparent=True)
    else:
        plt.show()
    return coeffs


def plot_avg_mask_influence(data_dict, structures_dict, model_type, noc_thr, lw=0.5, save=False, font_size=12):
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


def contrast_influence(save=False):
    with open('./data/contrast_difference.json', 'r') as f:
        contrast_dict = json.load(f)

    print(contrast_dict)

    poly_fn_seg = np.poly1d(np.squeeze(np.polyfit(contrast_dict['q2s'], contrast_dict['nocs_segformer'], 1)))
    poly_fn_cd = np.poly1d(np.squeeze(np.polyfit(contrast_dict['q2s'], contrast_dict['nocs_cdnet'], 1)))
    poly_fn_hr = np.poly1d(np.squeeze(np.polyfit(contrast_dict['q2s'], contrast_dict['nocs_hrnet'], 1)))

    f, ax = plt.subplots()
    ax.plot(contrast_dict['q2s'], contrast_dict['nocs_segformer'], 'x', color=colors[0], lw=linewidth)
    ax.plot(contrast_dict['q2s'], contrast_dict['nocs_cdnet'], 'x', color=colors[1], lw=linewidth)
    ax.plot(contrast_dict['q2s'], contrast_dict['nocs_hrnet'], 'x', color=colors[2], lw=linewidth)
    ax.plot(contrast_dict['q2s'], poly_fn_seg(contrast_dict['q2s']), color=colors[0], label='FocalClick', lw=linewidth)
    ax.plot(contrast_dict['q2s'], poly_fn_cd(contrast_dict['q2s']), color=colors[1], label='CDNet', lw=linewidth)
    ax.plot(contrast_dict['q2s'], poly_fn_hr(contrast_dict['q2s']), color=colors[2], label='RITM', lw=linewidth)
    ax.set_xlabel('Absolute difference HU in/out', fontsize=fs)
    ax.set_ylabel(f"NoC@80", fontsize=fs)
    ax.grid(visible=True, which='both', axis='both')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.show()


def combined_delta_relative(data_dict, model_type, n_clicks, lw=0.5, save=False, font_size=12):
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


def combined_delta_absolute(data_dict, model_type, n_clicks, lw=0.5, save=False, font_size=12):
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


def create_latex_table(structures_dict, exp_path):
    latex_table_string = ''

    for key in structures_dict:
        print(key)
        label = improve_label(key, abbreviations=True)
        latex_table_string = latex_table_string + label + " & "

        for model in os.listdir(exp_path):
            if key not in model or "hrnet" in model:
                continue
            model_path = exp_path + model + "/"
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
    colors = ["#e28743", '#117733', '#332288', '#88CCEE', '#DDCC77', '#44AA99', '#CC6677', '#882255', '#AA4499']
    linewidth = 1
    fs = 12

    experiments_path_segformer = "./experiments/focalclick/"
    experiments_path_cdnet = "./experiments/cdnet/"
    experiments_path_hrnet = '../ritm_interactive_segmentation/experiments/iter_mask/'

    structures_segformer = {'aorta': {'try': '000', 'epoch': 49, 'avg_mask': 3000},
                            'arteria_mesenterica_superior': {'try': '000', 'epoch': 39, 'avg_mask': 252},
                            'common_bile_duct': {'try': '000', 'epoch': 169, 'avg_mask': 501},
                            'gastroduodenalis': {'try': '000', 'epoch': 69, 'avg_mask': 61},
                            'pancreas': {'try': '000', 'epoch': 49, 'avg_mask': 979},
                            'pancreatic_duct': {'try': '000', 'epoch': 79, 'avg_mask': 162},
                            'tumour': {'try': '000', 'epoch': 109, 'avg_mask': 75}}
    structures_cdnet = {'aorta': {'try': '000', 'epoch': 19, 'avg_mask': 3000},
                        'arteria_mesenterica_superior': {'try': '000', 'epoch': 39, 'avg_mask': 252},
                        'common_bile_duct': {'try': '000', 'epoch': 29, 'avg_mask': 501},
                        'gastroduodenalis': {'try': '000', 'epoch': 59, 'avg_mask': 61},
                        'pancreas': {'try': '000', 'epoch': 19, 'avg_mask': 979},
                        'pancreatic_duct': {'try': '000', 'epoch': 9, 'avg_mask': 162},
                        'tumour': {'try': '000', 'epoch': 19, 'avg_mask': 75}}
    structures_hrnet = {'aorta': {'try': '001', 'epoch': 169, 'avg_mask': 3001},
                        'arteria_mesenterica_superior': {'try': '001', 'epoch': 119, 'avg_mask': 252},
                        'common_bile_duct': {'try': '001', 'epoch': 110, 'avg_mask': 501},
                        'gastroduodenalis': {'try': '001', 'epoch': 29, 'avg_mask': 61},
                        'pancreas': {'try': '002', 'epoch': 149, 'avg_mask': 979},
                        'pancreatic_duct': {'try': '000', 'epoch': 179, 'avg_mask': 162},
                        'tumour': {'try': '000', 'epoch': 159, 'avg_mask': 75}}

    # loads test set results
    data_segformer = load_data_to_plot(structures_segformer, experiments_path_segformer, 'segformer', ious=True)
    data_cdnet = load_data_to_plot(structures_cdnet, experiments_path_cdnet, 'cdnet', ious=True)
    data_hrnet = load_data_to_plot(structures_hrnet, experiments_path_hrnet, 'hrnet', ious=True)

    data_all_models = [data_segformer, data_cdnet, data_hrnet]

    # plots with all three models
    for key in structures_hrnet:
        all_models_noc_histogram(key, n_clicks=50, noc_thr=0.8, save=True)
