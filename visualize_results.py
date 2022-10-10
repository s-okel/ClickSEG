import json
import math
import matplotlib.pyplot as plt
import scipy.optimize
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import os
import pandas as pd
import pickle
import regex as re
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.metrics import mean_absolute_error
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


def load_data_to_plot(data_dict, experiment_path, model_type):
    iou_dict = {}
    print(colored(f"{model_type}", 'yellow'))

    for model in os.listdir(experiment_path):
        if model_type not in model:
            continue
        model_path = os.path.join(experiment_path, model)
        for model_try in os.listdir(model_path):
            model_try_path = os.path.join(model_path, model_try)
            evaluation_path = os.path.join(model_try_path, "evaluation_logs/test_set/others/")
            if os.path.exists(evaluation_path):
                for epoch in os.listdir(evaluation_path):
                    if model_type == 'hrnet':
                        plots_path = os.path.join(evaluation_path, epoch, epoch, "plots")
                    else:
                        plots_path = os.path.join(evaluation_path, epoch, epoch)
                    if os.path.exists(plots_path):
                        for file in os.listdir(plots_path):
                            for key in data_dict:
                                if ".pickle" in file:
                                    if data_dict[key]['try'] in plots_path \
                                            and 'epoch-' + str(data_dict[key]['epoch']) in plots_path \
                                            and key in plots_path \
                                            and '1' + str(data_dict[key]['epoch']) not in plots_path \
                                            and '2' + str(data_dict[key]['epoch']) not in plots_path:
                                        print('\t' + colored(f"Key: {key}", "green"))
                                        print(f"\tLoading {os.path.join(plots_path, file)}")
                                        with open(os.path.join(plots_path, file), "rb") as f:
                                            if metric == 'iou':
                                                iou_dict[key] = np.array(pickle.load(f)['all_ious'])
                                            elif metric == 'dice':
                                                iou_dict[key] = np.array(pickle.load(f)['all_dices'])

    return iou_dict


def boundary_progression_plot(structures_dict, lw=0.5, font_size=12):
    boundary_hu_per_structure = {}
    selected_colors = colors[0:len(structures_dict)]

    for key in structures_dict:
        boundary_hu_per_structure[key] = []

    with open('structure_boundary_hu.json', 'r') as f:
        boundary_data = json.load(f)

    for key in boundary_data:
        print(key)
        for structure in boundary_data[key]:
            boundary_hu_per_structure[structure].append(boundary_data[key][structure][0])

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
    plt.legend(prop={'size': fs})
    plt.grid(visible=True, which='both', axis='both')
    plt.show()


def in_out_stats(noc_thr, n_clicks, save=False):
    contrast_dict = {'means_in': [], 'means_out': [], 'q1s_in': [], 'q2s_in': [], 'q3s_in': [], 'q1s_out': [],
                     'q2s_out': [], 'q3s_out': [], 'nocs_hrnet': [], 'nocs_segformer': [], 'nocs_cdnet': [],
                     'stds_in': [], 'stds_out': []}
    temp_dict = {}
    for key in structures_hrnet:
        temp_dict[key] = {'in': [], 'out': []}

    print("Loading JSON...")
    with open('./data/in_vs_out_values.json', 'r') as f:
        d = json.load(f)
    print("Loaded JSON.")

    for patient_case in d:  # patient case e.g. ct_1005
        print(patient_case)
        for structure in d[patient_case]:  # e.g. aorta
            temp_dict[structure]['in'].extend(d[patient_case][structure]['in_values'][0])
            temp_dict[structure]['out'].extend(d[patient_case][structure]['out_values'][0])

    for key in temp_dict:
        nocs_hrnet, _ = compute_noc_metric(data_hrnet[key], [noc_thr], n_clicks)
        nocs_segformer, _ = compute_noc_metric(data_segformer[key], [noc_thr], n_clicks)
        nocs_cdnet, _ = compute_noc_metric(data_cdnet[key], [noc_thr], n_clicks)

        contrast_dict['means_in'].append(np.mean(np.array(temp_dict[key]['in'])))
        contrast_dict['means_out'].append(np.mean(np.array(temp_dict[key]['out'])))
        contrast_dict['stds_in'].append(np.std(np.array(temp_dict[key]['in'])))
        contrast_dict['stds_out'].append(np.std(np.array(temp_dict[key]['out'])))
        contrast_dict['q1s_in'].append(np.percentile(np.array(temp_dict[key]['in']), 25))
        contrast_dict['q2s_in'].append(np.percentile(np.array(temp_dict[key]['in']), 50))
        contrast_dict['q3s_in'].append(np.percentile(np.array(temp_dict[key]['in']), 75))
        contrast_dict['q1s_out'].append(np.percentile(np.array(temp_dict[key]['out']), 25))
        contrast_dict['q2s_out'].append(np.percentile(np.array(temp_dict[key]['out']), 50))
        contrast_dict['q3s_out'].append(np.percentile(np.array(temp_dict[key]['out']), 75))
        contrast_dict['nocs_hrnet'].append(nocs_hrnet[0])
        contrast_dict['nocs_segformer'].append(nocs_segformer[0])
        contrast_dict['nocs_cdnet'].append(nocs_cdnet[0])

    print(contrast_dict)

    with open('./data/contrast_difference.json', 'w') as f:
        json.dump(contrast_dict, f)


def in_out_plot(save=False):
    print("Loading pickle...")
    df = pd.read_pickle('./data/in_out_df.pkl')
    print("Loaded pickle.")
    df = df.sort_values('structure')  # ensure that data is in alphabetical order - I have it everywhere in my paper
    structure_names = df.structure.unique()

    boxplot = sns.boxplot(data=df, x='structure', y='values', hue='in', palette=["#e28743", "#eab676"],
                          showfliers=False, width=0.4)
    boxplot.set_xlabel('Structure', fontsize=fs)
    boxplot.set_ylabel('Intensity [HU]', fontsize=fs)
    boxplot.set_xticklabels([improve_label(structure, True) for structure in structure_names])
    boxplot.spines.right.set_visible(False)
    boxplot.spines.top.set_visible(False)
    boxplot.xaxis.set_tick_params(length=0)
    boxplot.set_ylim([-300, 450])

    boxplot.grid(visible=True, which='both', axis='y')
    boxplot.set_axisbelow(True)

    handles, _ = boxplot.get_legend_handles_labels()
    boxplot.legend(handles, ['Inside', 'Outside'])
    if save:
        plt.savefig(f"epoch_evaluations/in_out_boxplot.png", dpi=300, transparent=True)
    else:
        plt.show()


def single_boxplot(data_dict, model_type, label, n_clicks, save=False):
    ious_array = data_dict[label][:, :n_clicks]

    f, ax = plt.subplots()
    ax.boxplot(ious_array, showfliers=False, medianprops=dict(color="#e28743"))
    plt.xlabel("NoC", fontsize=fs)
    plt.ylabel("mIoU", fontsize=fs)
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
    plt.xlabel("NoC", fontsize=font_size)
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
        plt.savefig(f"epoch_evaluations/histograms/{model_type}_{label}_individual_histogram.pdf", dpi=300)
    else:
        plt.show()


def all_models_noc_histogram(label, n_clicks, noc_thr, save):
    selected_colors = colors[:len(data_all_models)]
    selected_styles = linestyles[:len(data_all_models)]

    f, ax = plt.subplots()
    for data_dict, color, l_style in zip(data_all_models, selected_colors, selected_styles):
        ious_array = data_dict[label][:, :n_clicks]
        nocs = []

        # for each model, calculate the nocs of each sample
        for i in range(ious_array.shape[0]):
            noc = get_noc(ious_array[i], iou_thr=noc_thr, max_clicks=n_clicks)
            nocs.append(noc)

        ax.hist(nocs, bins=n_clicks, histtype='step', color=color, lw=linewidth, label=improve_label(label, True),
                linestyle=l_style)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        plt.grid(visible=True, which='both', axis='y')

    plt.xlabel(f"NoC@{int(noc_thr * 100)}", fontsize=fs)
    plt.ylabel("Frequency", fontsize=fs)
    plt.xlim([0, n_clicks + 1])
    plt.legend(['FocalClick', 'CDNet', 'RITM'], loc='upper center')
    if save:
        plt.savefig(f"epoch_evaluations/histograms/all_models_histogram_{label}_{metric}.pdf", dpi=300)
    else:
        plt.show()


def combined_miou_plot(data_dict, model_type, n_clicks, save=False):
    f, ax = plt.subplots(figsize=(6, 4))
    selected_colors = colors[0:len(data_dict)]
    selected_styles = linestyles[:len(data_dict)]
    for key, color, l_style in zip(data_dict, selected_colors, selected_styles):
        metrics = np.mean(data_dict[key][:, :n_clicks], axis=0)
        ax.plot(range(1, n_clicks + 1), metrics, linewidth=linewidth,
                label=improve_label(key, abbreviations=True), color=color, linestyle=l_style)
        # print(f"Structure {key}, mDSC: {metrics}")
        print(f"Gain after {n_clicks} clicks for {key}: {metrics[-1] - metrics[0]:.2f}, from {metrics[0]:.2f} to {metrics[-1]:.2f}")

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # ax.legend(prop={'size': fs}, bbox_to_anchor=(1, 0.5), loc='center left')
    plt.xlabel("NoC", fontsize=fs)
    if metric == 'iou':
        plt.ylabel("mIoU", fontsize=fs)
    elif metric == 'dice':
        plt.ylabel("mDSC", fontsize=fs)
    plt.ylim([0, 1])
    plt.xlim([1, n_clicks])
    plt.tight_layout()

    plt.grid(visible=True, which='both', axis='both')
    if save:
        plt.savefig(f"epoch_evaluations/miou/{model_type}_mIoU_combined_{metric}.png", dpi=300, transparent=True)
    else:
        plt.show()


def combined_noc_histogram(data_dict, model_type, n_clicks, noc_thr, lw=0.5, save=False, font_size=12):
    f, ax = plt.subplots()
    selected_colors = colors[0:len(data_dict)]

    for key, color in zip(data_dict, selected_colors):
        nocs = []
        ious_array = data_dict[key][:, :n_clicks]

        for i in range(ious_array.shape[0]):
            noc = get_noc(ious_array[i], noc_thr, n_clicks)
            nocs.append(noc)

        ax.hist(nocs, bins=n_clicks, histtype='step', label=key, color=color, density=True, lw=lw)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel(f"NoC for NoC of {noc_thr}", fontsize=font_size)
    plt.ylim([0, 1])
    plt.xlim([0, n_clicks + 1])
    plt.ylabel("Density", fontsize=font_size)
    plt.legend(prop={'size': fs})
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

    for key in data_dict:
        keys.append(key)
        # return to the original labels
        label = improve_label(key, abbreviations=True)

        nocs = []
        avg_mask_list.append(structures_dict[label]['avg_mask'])
        for i in range(data_dict[key].shape[0]):
            noc = get_noc(data_dict[key][i], noc_thr, data_dict[key].shape[1])
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


def all_models_delta_relative(label, n_clicks, noc_thr, save):
    selected_colors = colors[:len(data_all_models)]

    f, ax = plt.subplots()
    for data_dict, color in zip(data_all_models, selected_colors):
        ious_array = data_dict[label][:, :n_clicks]
        improvement = np.mean(ious_array - np.mean(ious_array[:, 0]), axis=0)
        ax.plot(range(1, n_clicks + 1), improvement, color=color, lw=linewidth)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
    plt.xlabel("Number of clicks", fontsize=fs)
    plt.ylabel(f"Increase in mIoU", fontsize=fs)
    plt.xlim([1, n_clicks])
    plt.legend(['FocalClick', 'CDNet', 'RITM'], prop={'size': fs})
    plt.grid(visible=True, which='both', axis='both')
    plt.show()


def combined_delta_relative(data_dict, model_type, n_clicks, lw=0.5, save=False, font_size=12):
    selected_colors = colors[0:len(data_dict)]

    f, ax = plt.subplots()
    for key, color in zip(data_dict, selected_colors):
        ious_array = data_dict[key][:, :n_clicks]
        improvement = np.mean(ious_array - np.mean(ious_array[:, 0]), axis=0)
        ax.plot(range(1, n_clicks + 1), improvement, label=key, color=color, lw=lw)
        print(f"{improvement[9]:.2f} improvement after 10 clicks for {key}")
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Number of clicks", fontsize=font_size)
    plt.ylabel(f"Increase in mIoU", fontsize=font_size)
    plt.xlim([1, n_clicks])
    plt.legend(prop={'size': fs})
    plt.grid(visible=True, which='both', axis='both')
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_delta_relative.pdf", dpi=300)
    else:
        plt.show()


def all_models_miou(label, n_clicks, save):
    selected_colors = colors[:len(data_all_models)]
    selected_styles = linestyles[:len(data_all_models)]

    f, ax = plt.subplots()
    for data_dict, color, l_style in zip(data_all_models, selected_colors, selected_styles):
        ious_array = data_dict[label][:, :n_clicks]
        improvement = np.mean(ious_array, axis=0)
        ax.plot(range(1, n_clicks + 1), improvement, color=color, lw=linewidth, linestyle=l_style)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
    plt.xlabel("NoC", fontsize=fs)
    if metric == 'iou':
        plt.ylabel("mIoU", fontsize=fs)
    elif metric == 'dice':
        plt.ylabel("mDSC", fontsize=fs)
    plt.ylim([0, 1])
    plt.xlim([1, n_clicks])
    if sota[label] is not None:
        ax.hlines(sota[label], 0, n_clicks, color=colors[len(data_all_models)], lw=linewidth)
    plt.legend(['FocalClick', 'CDNet', 'RITM', 'SOTA'], prop={'size': fs})
    plt.grid(visible=True, which='both', axis='both')
    if save:
        plt.savefig(f"epoch_evaluations/miou/all_models_mIoU_{label}_{metric}.png", dpi=300, transparent=True)
    else:
        plt.show()


def combined_delta_absolute(data_dict, model_type, n_clicks, lw=0.5, save=False, font_size=12):
    selected_colors = colors[0:len(data_dict)]

    f, ax = plt.subplots()
    plt.hlines(0, 1, n_clicks, colors=['k'], lw=lw)
    for key, color in zip(data_dict, selected_colors):
        ious_array = data_dict[key][:, :n_clicks]
        ax.plot(range(1, n_clicks), np.diff(np.mean(ious_array, axis=0)), linewidth=lw, label=key, color=color)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.xlabel("Number of clicks", fontsize=font_size)
    plt.ylabel(f"Delta", fontsize=font_size)
    plt.xlim([1, n_clicks - 1])
    plt.legend(prop={'size': fs})
    if save:
        plt.savefig(f"epoch_evaluations/{model_type}_delta_absolute.pdf", dpi=300)
    else:
        plt.show()


def cum_performance_plot(noc_thr, n_clicks, save=False):
    selected_colors = colors[:len(structures_cdnet)]
    all_nocs = []
    for data_dict in data_all_models:
        nocs_per_model = []
        for label in data_dict:
            ious_array = data_dict[label][:n_clicks]

            nocs, _ = compute_noc_metric(ious_array, [noc_thr], n_clicks)
            nocs_per_model.extend(nocs)
        all_nocs.append(nocs_per_model)

    nocs = np.array(all_nocs)
    print(nocs)

    f, ax = plt.subplots()

    bottom_values = [0 for _ in range(3)]
    for i, structure in enumerate(structures_cdnet):
        ax.bar(['FocalClick', 'CDNet', 'RITM'], nocs[:, i], label=improve_label(structure, True), bottom=bottom_values,
               color=selected_colors[i])
        bottom_values = [sum(x) for x in zip(bottom_values, nocs[:, i])]
    ax.set_axisbelow(True)
    ax.grid(visible=True, which='both', axis='y')
    ax.set_ylabel(f"NoC@{int(100 * noc_thr)}", fontsize=fs)
    plt.legend(prop={'size': fs})

    if save:
        plt.savefig(f'./epoch_evaluations/cum_noc_plot.pdf', dpi=300)
    else:
        plt.show()


def create_latex_table(structures_dict, experiment_path, metric):
    assert metric == 'iou' or metric == 'dice', f"expected metric iou or dice, not {metric}"
    latex_table_string = ''

    for key in structures_dict:
        label = improve_label(key, abbreviations=True)
        latex_table_string = latex_table_string + label + " & "

        for model in os.listdir(experiment_path):
            if key not in model:
                continue
            model_path = os.path.join(experiment_path, model)
            for model_try in os.listdir(model_path):
                if structures_dict[key]['try'] not in model_try:
                    continue
                model_try_path = os.path.join(model_path, model_try)
                evaluation_path = os.path.join(model_try_path, "evaluation_logs/test_set/others")
                if os.path.exists(evaluation_path):
                    for epoch in os.listdir(evaluation_path):
                        epoch_path = os.path.join(evaluation_path, epoch, epoch)
                        if os.path.exists(epoch_path):
                            for file in os.listdir(epoch_path):
                                if ".txt" in file:
                                    with open(os.path.join(epoch_path, file), "r") as f:
                                        text = f.read()
                                    values = re.findall(r"([0-9]{0,2}\.[0-9]{2})(?!\|)(?![0-9])", text)
                                    nocs = [round(float(val) * 10) / 10 for val in values[:6]]
                                    mious = [round(float(val)) / 100 for val in values[6:]]

                                    if metric == 'iou':
                                        values = nocs[:3] + mious[:6]
                                    else:
                                        values = nocs[3:] + mious[6:]

                                    # add to latex table
                                    n_values = len(values)
                                    for i, value in enumerate(values):
                                        if i + 1 == n_values:
                                            latex_table_string = latex_table_string + str(value) + r" \\" + "\n"
                                        else:
                                            latex_table_string = latex_table_string + str(value) + " & "

    print(latex_table_string)


def val_loss_vs_metrics(model_type, label, n_clicks, noc_thr, save=False):
    if model_type == 'segformer':
        try_nr = structures_segformer[label]['try']
        experiments_path = experiments_path_segformer
    elif model_type == 'cdnet':
        try_nr = structures_cdnet[label]['try']
        experiments_path = experiments_path_cdnet
    elif model_type == 'hrnet':
        try_nr = structures_hrnet[label]['try']
        experiments_path = experiments_path_hrnet
    else:
        raise ValueError(f'Expected model type segformer/cdnet/hrnet, but received {model_type}')

    # load for one label, one try, one model all IoUs for each checkpoint
    losses = []
    nocs = []
    epochs = []

    for model in os.listdir(experiments_path):
        if label not in model:
            continue
        model_path = os.path.join(experiments_path, model)
        for model_try in os.listdir(model_path):
            if try_nr not in model_try:
                continue
            model_try_path = os.path.join(model_path, model_try)
            if model_type == 'hrnet':
                evaluation_path = os.path.join(model_try_path, 'evaluation_logs', 'others')
            else:
                evaluation_path = os.path.join(model_try_path, "evaluation_logs")
            if os.path.exists(evaluation_path):
                for epoch in os.listdir(evaluation_path):
                    if model_type == 'hrnet':
                        plots_path = os.path.join(evaluation_path, epoch, 'plots')
                    else:
                        plots_path = os.path.join(evaluation_path, epoch)
                    if os.path.exists(plots_path):
                        for file in os.listdir(plots_path):
                            if ".pickle" in file:
                                epoch = re.findall(r'[0-9]{1,3}(?=-)', plots_path)[0]
                                epochs.append(float(epoch))
                                loss = re.findall(r'[0-9]{1,2}\.[0-9]{1,2}', plots_path)[0]
                                losses.append(float(loss))
                                with open(os.path.join(plots_path, file), "rb") as f:
                                    ious_array = np.array(pickle.load(f)['all_ious'])[:, :n_clicks]
                                noc_per_image = []
                                for i in range(ious_array.shape[0]):
                                    noc = get_noc(ious_array[i], iou_thr=noc_thr, max_clicks=n_clicks)
                                    noc_per_image.append(noc)
                                nocs.append(np.mean(noc_per_image))

    print(f"Pearson correlation coefficient between loss and NoC "
          f"for {label}, model {model_type}: {pearsonr(losses, nocs)}")

    f, ax = plt.subplots()
    # losses, nocs = zip(*sorted(zip(losses, nocs)))
    ax.plot(losses, nocs, '.', color="#e28743", lw=linewidth)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.grid(visible=True, which='both', axis='both')
    plt.xlabel('Validation loss', fontsize=fs)
    plt.ylabel(f'NoC@{int(noc_thr * 100)}', fontsize=fs)
    plt.show()

    epochs, losses, nocs = zip(*sorted(zip(epochs, losses, nocs)))
    ax1 = plt.subplot()

    ax1.plot(epochs, losses, color="#e28743", lw=linewidth, label='Loss')
    ax1.spines.top.set_visible(False)
    ax1.set_ylabel('Validation loss')
    ax2 = ax1.twinx()
    ax2.spines.top.set_visible(False)
    ax2.set_ylabel(f"NoC@{int(noc_thr * 100)}")
    ax2.plot(epochs, nocs, color='#332288', lw=linewidth, label='NoC')
    ax2.set_xlabel("Epoch")
    # dummy graph for legend
    ax2.plot([], [], color="#e28743", label='Loss')

    ax1.set_xlim([9, 219])
    ax2.set_xlim([9, 219])
    plt.grid(visible=True, which='both', axis='both')
    plt.legend(prop={'size': fs})
    plt.show()


def subjective_score(save=False):
    # 2nd SMA, 1st tumor, 3rd tumor, 1st aorta, 3rd aorta, 2st CBD
    subj_max = [0.95, 0.89, 0.85, 0.95, 0.85, 0.9]
    subj_min = [0.9, 0.7, 0.8, 0.95, 0.85, 0.85]
    dices = [0.84, 0.75, 0.73, 0.90, 0.89, 0.87]
    ious = [0.73, 0.60, 0.57, 0.83, 0.81, 0.77]
    labels = ['SMA', 'Tumor', 'Tumor', 'Aorta', 'Aorta', 'CBD']

    print(f"\nCorrelation coefficient between max subjective score and dice score: "
          f"{pearsonr(subj_max, dices)[0]}, p = {pearsonr(subj_max, dices)[1]}")
    print(f"Correlation coefficient between max subjective score and iou: "
          f"{pearsonr(subj_max, ious)[0]}, p = {pearsonr(subj_max, ious)[1]}\n")

    print(f"Mean absolute error between max subjective score and dice score: {mean_absolute_error(subj_max, dices)}")
    print(f"Mean absolute error between max subjective score and iou: {mean_absolute_error(subj_max, ious)}")

    print(f"\nCorrelation coefficient between min subjective score and dice score: "
          f"{pearsonr(subj_min, dices)[0]}, p = {pearsonr(subj_min, dices)[1]}")
    print(f"Correlation coefficient between min subjective score and iou: "
          f"{pearsonr(subj_min, ious)[0]}, p = {pearsonr(subj_min, ious)[1]}\n")

    print(f"Mean absolute error between min subjective score and dice score: {mean_absolute_error(subj_min, dices)}")
    print(f"Mean absolute error between min subjective score and iou: {mean_absolute_error(subj_min, ious)}")

    f, ax = plt.subplots()
    ax.plot(subj_min, label="Subjective score [min, max]", linestyle=linestyles[0], color=colors[0],
            linewidth=linewidth)
    ax.plot(subj_max, linestyle=linestyles[0], color=colors[0], linewidth=linewidth)
    ax.plot(dices, label="DSC", linestyle=linestyles[1], color=colors[1], linewidth=linewidth)
    ax.plot(ious, label="IoU", linestyle=linestyles[2], color=colors[2], linewidth=linewidth)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 5])
    ax.set_ylabel("Score", fontsize=fs)
    ax.set_xlabel("Structure", fontsize=fs)
    ax.set_xticks([i for i in range(6)])
    ax.set_xticklabels(labels, fontsize=fs)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.fill_between([i for i in range(6)], subj_min, subj_max, color=colors[0], alpha=0.5)
    plt.legend(prop={'size': fs})
    plt.grid(visible=True, which='both', axis='both')

    if save:
        plt.savefig("epoch_evaluations/metrics.png", dpi=300, transparent=True)
    else:
        plt.show()


def subjective_score_cat(save=False):
    subj_max = [0.95, 0.89, 0.85, 0.95, 0.85, 0.9]
    subj_min = [0.9, 0.7, 0.8, 0.95, 0.85, 0.85]
    dices = [0.84, 0.75, 0.73, 0.90, 0.89, 0.87]
    ious = [0.73, 0.60, 0.57, 0.83, 0.81, 0.77]
    labels = ['SMA', 'Tumor', 'Tumor', 'Aorta', 'Aorta', 'CBD']
    ms = 30

    f, ax = plt.subplots()
    ax.plot(subj_min, '_', label="Subjective score [min, max]", color=colors[0],
            linewidth=linewidth, markersize=ms)
    ax.plot(subj_max, '_', color=colors[0], linewidth=linewidth, markersize=ms)
    ax.plot(dices, '_', label="DSC", color=colors[1], linewidth=linewidth, markersize=ms)
    ax.plot(ious, '_', label="IoU", color=colors[2], linewidth=linewidth, markersize=ms)
    ax.set_ylim([0, 1])
    # ax.set_xlim([0, 5])
    ax.set_ylabel("Score", fontsize=fs)
    ax.set_xlabel("Structure", fontsize=fs)
    ax.set_xticks([i for i in range(6)])
    ax.set_xticklabels(labels, fontsize=fs)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # plt.fill_between([i for i in range(6)], subj_min, subj_max, color=colors[0], alpha=0.5)
    for i in range(6):
        plt.fill_between([i - 0.25, i + 0.25], subj_min[i], subj_max[i], color=colors[0], alpha=0.5)
    plt.legend(prop={'size': fs})
    plt.grid(visible=True, which='both', axis='y')

    if save:
        plt.savefig("epoch_evaluations/metrics_cat.png", dpi=300, transparent=True)
    else:
        plt.show()


if __name__ == "__main__":
    colors = ["#e28743", '#117733', '#332288', '#88CCEE', '#DDCC77', '#44AA99', '#CC6677', '#882255', '#AA4499']
    linestyles = ['dashdot', 'dashed', 'dotted', 'dashed', 'solid', 'dashdot', 'dotted', 'dashed', 'solid']
    linewidth = 2.5
    fs = 12
    metric = 'dice'
    assert metric == 'dice' or metric == 'iou', f"Expected metric dice or iou, but received {metric}"

    if metric == 'dice':
        sota = {'aorta': 0.95, 'arteria_mesenterica_superior': 0.88, 'common_bile_duct': None, 'gastroduodenalis': None,
                'pancreas': 0.91, 'pancreatic_duct': 0.5, 'tumour': 0.57}

    else:
        sota = {'aorta': None, 'arteria_mesenterica_superior': 0.74, 'common_bile_duct': 0.86, 'gastroduodenalis': None,
                'pancreas': None, 'pancreatic_duct': None, 'tumour': None}

    experiments_path_segformer = "./experiments/focalclick/"
    experiments_path_cdnet = "./experiments/cdnet/"
    experiments_path_hrnet = '../ritm_interactive_segmentation/experiments/iter_mask/'
    exp_paths = [experiments_path_segformer, experiments_path_cdnet, experiments_path_hrnet]

    structures_segformer = {'aorta': {'try': '000', 'epoch': 49, 'avg_mask': 3000},
                            'arteria_mesenterica_superior': {'try': '000', 'epoch': 39, 'avg_mask': 252},
                            'common_bile_duct': {'try': '000', 'epoch': 169, 'avg_mask': 501},
                            'gastroduodenalis': {'try': '000', 'epoch': 69, 'avg_mask': 61},
                            'pancreas': {'try': '000', 'epoch': 49, 'avg_mask': 979},
                            'pancreatic_duct': {'try': '000', 'epoch': 79, 'avg_mask': 162},
                            'tumour': {'try': '000', 'epoch': 109, 'avg_mask': 75}}

    structures_segformer_r1 = {'aorta': {'try': '001', 'epoch': 69, 'avg_mask': 3000},
                               'arteria_mesenterica_superior': {'try': '001', 'epoch': 99, 'avg_mask': 252},
                               'common_bile_duct': {'try': '001', 'epoch': 159, 'avg_mask': 501},
                               'gastroduodenalis': {'try': '001', 'epoch': 99, 'avg_mask': 61},
                               'pancreas': {'try': '001', 'epoch': 89, 'avg_mask': 979},
                               'pancreatic_duct': {'try': '001', 'epoch': 99, 'avg_mask': 162},
                               'tumour': {'try': '001', 'epoch': 199, 'avg_mask': 75}}

    structures_cdnet = {'aorta': {'try': '000', 'epoch': 19, 'avg_mask': 3000},
                        'arteria_mesenterica_superior': {'try': '000', 'epoch': 39, 'avg_mask': 252},
                        'common_bile_duct': {'try': '000', 'epoch': 29, 'avg_mask': 501},
                        'gastroduodenalis': {'try': '000', 'epoch': 59, 'avg_mask': 61},
                        'pancreas': {'try': '000', 'epoch': 19, 'avg_mask': 979},
                        'pancreatic_duct': {'try': '000', 'epoch': 169, 'avg_mask': 162},
                        'tumour': {'try': '000', 'epoch': 199, 'avg_mask': 75}}

    structures_cdnet_r1 = {'aorta': {'try': '001', 'epoch': 59, 'avg_mask': 3000},
                           'arteria_mesenterica_superior': {'try': '001', 'epoch': 39, 'avg_mask': 252},
                           'common_bile_duct': {'try': '001', 'epoch': 59, 'avg_mask': 501},
                           'gastroduodenalis': {'try': '001', 'epoch': 139, 'avg_mask': 61},
                           'pancreas': {'try': '001', 'epoch': 39, 'avg_mask': 979},
                           'pancreatic_duct': {'try': '001', 'epoch': 39, 'avg_mask': 162},
                           'tumour': {'try': '001', 'epoch': 209, 'avg_mask': 75}}

    structures_hrnet = {'aorta': {'try': '001', 'epoch': 169, 'avg_mask': 3001},
                        'arteria_mesenterica_superior': {'try': '001', 'epoch': 119, 'avg_mask': 252},
                        'common_bile_duct': {'try': '001', 'epoch': 110, 'avg_mask': 501},
                        'gastroduodenalis': {'try': '001', 'epoch': 29, 'avg_mask': 61},
                        'pancreas': {'try': '002', 'epoch': 149, 'avg_mask': 979},
                        'pancreatic_duct': {'try': '000', 'epoch': 179, 'avg_mask': 162},
                        'tumour': {'try': '000', 'epoch': 159, 'avg_mask': 75}}

    structures_hrnet_r1 = {'aorta': {'try': '013', 'epoch': 79, 'avg_mask': 3001},
                           'arteria_mesenterica_superior': {'try': '002', 'epoch': 189, 'avg_mask': 252},
                           'common_bile_duct': {'try': '004', 'epoch': 69, 'avg_mask': 501},
                           'gastroduodenalis': {'try': '002', 'epoch': 39, 'avg_mask': 61},
                           'pancreas': {'try': '003', 'epoch': 49, 'avg_mask': 979},
                           'pancreatic_duct': {'try': '001', 'epoch': 89, 'avg_mask': 162},
                           'tumour': {'try': '002', 'epoch': 19, 'avg_mask': 75}}

    structures_r5 = [structures_segformer, structures_cdnet, structures_hrnet]
    structures_r1 = [structures_segformer_r1, structures_cdnet_r1, structures_hrnet_r1]

    # loads test set results
    data_segformer = load_data_to_plot(structures_segformer, experiments_path_segformer, 'segformer')
    data_cdnet = load_data_to_plot(structures_cdnet, experiments_path_cdnet, 'cdnet')
    data_hrnet = load_data_to_plot(structures_hrnet, experiments_path_hrnet, 'hrnet')
    data_all_models = [data_segformer, data_cdnet, data_hrnet]

    # in_out_plot(save=True)
    # contrast_influence(save=False)
    # in_out_stats(0.8, 50)
    # cum_performance_plot(0.8, 50, save=True)
    # val_loss_vs_metrics('cdnet', 'common_bile_duct', n_clicks=50, noc_thr=0.8, save=False)
    # all_models_mask_influence(0.8, 50, save=False)
    # subjective_score(save=True)
    # subjective_score_cat(save=False)

    for structures, exp_path in zip(structures_r1, exp_paths):
        # print(exp_path)
        # create_latex_table(structures, exp_path, 'dice')
        # create_latex_table(structures, exp_path, 'iou')
        continue

    for thr in range(10, 110, 10):
        # c = all_models_mask_influence(0.8, thr, plot=True, save=False)
        # print(f"n clicks = {thr} results in coefficients of {c}")
        continue

    for k in structures_hrnet:
        # all_models_noc_histogram(k, n_clicks=50, noc_thr=0.8, save=True)
        # all_models_delta_relative(k, n_clicks=50, noc_thr=0.8, save=True)
        # all_models_miou(k, n_clicks=50, save=True)
        continue

    for data, m_type in zip(data_all_models, ['segformer', 'cdnet', 'hrnet']):
        combined_miou_plot(data, m_type, 10, save=False)
        for lbl in structures_hrnet:
            # val_loss_vs_metrics(m_type, lbl, n_clicks=50, noc_thr=0.8, save=False)
            # single_boxplot(data, m_type, lbl, 20)
            continue
