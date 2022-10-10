from termcolor import colored
import datetime
import json
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd
import regex as re
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from isegm.inference.utils import compute_noc_metric


def overlap(a_s, a_e, b_s, b_e):
    if a_s > b_e or b_s > a_e:
        return 0
    else:
        return min(a_e, b_e) - max(a_s, b_s)


# gives the average HU value from 10 pixels inwards and 10 pixels outside of the structure for each pixel layer
def structure_boundaries(structures: list):  # 243 images
    image_info_dict = {}

    for patient_nr in os.listdir(data_path):
        phase_path = os.path.join(data_path, patient_nr)
        for phase in os.listdir(phase_path):
            images_path = os.path.join(phase_path, phase, 'imagesTr')
            for scan in os.listdir(images_path):
                image_info_dict[scan] = {}
                print(f"Loading scan {scan}...")
                image = np.array(nib.load(os.path.join(images_path, scan)).get_fdata())
                print(f"Loaded.")
                for structure in structures:
                    label_name = scan.split('.')[0] + '_' + structure + scan[-7:]
                    label_path = os.path.join(phase_path, phase, 'labelsTr', label_name)
                    if os.path.exists(label_path):
                        print(f"Loading label {label_name}...")
                        label = np.array(nib.load(label_path).get_fdata())
                        print(f"Loaded.")
                        hu_values = boundary_mean(image, label, 10)
                        if structure not in image_info_dict[scan]:
                            image_info_dict[scan][structure] = []
                        image_info_dict[scan][structure].append(hu_values)

    with open('structure_boundary_hu.json', 'w') as f:
        json.dump(image_info_dict, f)


def boundary_mean(img, lbl, boundary_size):
    hu_list = []
    prev_label = lbl.copy()
    largest_mask_slice = np.argmax(np.sum(lbl, axis=(1, 2)))

    for i in range(boundary_size):
        eroded_label = binary_erosion(lbl, iterations=i + 1)

        ''' 
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(lbl[largest_mask_slice])
        ax[1].imshow(eroded_label[largest_mask_slice])
        ax[2].imshow((eroded_label != prev_label)[largest_mask_slice])
        plt.show()
        '''

        if np.sum(eroded_label) == 0:
            mean_hu = 40000
        else:
            mean_hu = np.mean(img[eroded_label != prev_label])
        hu_list.append(mean_hu)

        prev_label = eroded_label

    hu_list.reverse()
    prev_label = lbl.copy()

    for i in range(boundary_size):
        expanded_label = binary_dilation(lbl, iterations=i + 1)

        '''
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(lbl[largest_mask_slice])
        ax[1].imshow(expanded_label[largest_mask_slice])
        ax[2].imshow((expanded_label != prev_label)[largest_mask_slice])
        plt.show()
        '''

        mean_hu = np.mean(img[expanded_label != prev_label])
        hu_list.append(mean_hu)

        prev_label = expanded_label

    print(hu_list)
    return hu_list


# compares the mean and standard deviation of the HU within and x pixels outside the structure
def in_out_mean_std(img, lbl, boundary_size):
    dilated_label = binary_dilation(lbl, iterations=boundary_size)

    mean_hu_in = np.mean(img[lbl == 1])
    std_hu_in = np.std(img[lbl == 1])

    mean_hu_out = np.mean(img[lbl != dilated_label])
    std_hu_out = np.std(img[lbl != dilated_label])

    return mean_hu_in, std_hu_in, mean_hu_out, std_hu_out


def in_out_values(img, lbl, boundary_size):
    dilated_label = binary_dilation(lbl, iterations=boundary_size)

    in_values = list(img[lbl == 1])
    out_values = list(img[lbl != dilated_label])

    return in_values, out_values


def in_vs_out_statistics(structures: list, statistics: bool):
    image_info_dict = {}

    for patient_nr in os.listdir(data_path):
        phase_path = os.path.join(data_path, patient_nr)
        for phase in os.listdir(phase_path):
            images_path = os.path.join(phase_path, phase, 'imagesTr')
            for scan in os.listdir(images_path):
                image_info_dict[scan] = {}
                print(f"Loading scan {scan}...")
                image = np.array(nib.load(os.path.join(images_path, scan)).get_fdata())
                print(f"Loaded.")
                for structure in structures:
                    label_name = scan.split('.')[0] + '_' + structure + scan[-7:]
                    label_path = os.path.join(phase_path, phase, 'labelsTr', label_name)
                    if os.path.exists(label_path):
                        print(f"Loading label {label_name}...")
                        label = np.array(nib.load(label_path).get_fdata())
                        print(f"Loaded.")
                        if statistics:
                            if structure not in image_info_dict[scan]:
                                image_info_dict[scan][structure] = {'mean_in': [], 'std_in': [], 'mean_out': [],
                                                                    'std_out': []}
                            mean_hu_in, std_hu_in, mean_hu_out, std_hu_out = in_out_mean_std(image, label, 5)
                            image_info_dict[scan][structure]['mean_in'].append(mean_hu_in)
                            image_info_dict[scan][structure]['std_in'].append(std_hu_in)
                            image_info_dict[scan][structure]['mean_out'].append(mean_hu_out)
                            image_info_dict[scan][structure]['std_out'].append(std_hu_out)
                        else:
                            if structure not in image_info_dict[scan]:
                                image_info_dict[scan][structure] = {'in_values': [], 'out_values': []}
                            in_values, out_values = in_out_values(image, label, 5)
                            image_info_dict[scan][structure]['in_values'].append(in_values)
                            image_info_dict[scan][structure]['out_values'].append(out_values)

    if statistics:
        with open('in_vs_out_statistics.json', 'w') as f:
            json.dump(image_info_dict, f)
    else:
        with open('in_vs_out_values.json', 'w') as f:
            json.dump(image_info_dict, f)


def in_out_pickle(structures_dict, lw=0.5, font_size=12):  # in/out values into pickle file suitable for boxplot
    n_values = 0
    df_dict = {'structure': [], 'in': [], 'values': []}

    with open('in_vs_out_values.json', 'r') as f:
        data = json.load(f)

    for patient_case in data:  # patient case e.g. ct_1005
        print(patient_case)
        for structure in data[patient_case]:  # e.g. aorta
            for location in data[patient_case][structure]:  # location: in or out
                for value in data[patient_case][structure][location][0]:
                    n_values += 1
                    df_dict['structure'].append(structure)
                    df_dict['in'].append(location.split('_')[0])
                    df_dict['values'].append(value)

    df = pd.DataFrame.from_dict(df_dict)
    print(df.head())
    print(f"{n_values} values")
    df.to_pickle('in_out_df.pkl')


def process_results_txt(experiments_path):
    # finds all epoch evaluations and saves them in different formats
    # used to check to validation metrics
    results = {}
    txt_file = "epoch_evaluations/epoch_evaluations.txt"

    if os.path.exists(txt_file):
        os.remove(txt_file)
    for model in os.listdir(experiments_path):
        model_path = os.path.join(experiments_path, model)
        print(colored(f"Model name: {model}", "green"))
        with open(txt_file, "a") as write_file:
            write_file.write(f"Model name: {model}\n")
        results[model] = {}
        for try_folder in os.listdir(model_path):
            n_epochs = 0
            results_per_try = []
            epochs_per_try = []
            try_nr = try_folder.split("_")[0]
            print(colored(f"\tTry: {try_nr}", "red"))
            with open(txt_file, "a") as write_file:
                write_file.write(f"\tTry: {try_nr}\n")
            results[model][try_nr] = {}
            if 'hrnet' in try_folder and "radius" not in try_folder:
                epochs_path = os.path.join(model_path, try_folder, "evaluation_logs/others")
            else:
                epochs_path = os.path.join(model_path, try_folder, "evaluation_logs")

            if os.path.exists(epochs_path):
                for epoch in os.listdir(epochs_path):
                    if 'epoch' not in epoch:
                        continue
                    epoch_path = os.path.join(epochs_path, epoch)
                    epoch = epoch.split("-")[1]
                    print(colored(f"\t\tEpoch: {epoch}", "yellow"))
                    with open(txt_file, "a") as write_file:
                        write_file.write(f"\t\tEpoch: {epoch}\n")
                    results[model][try_nr][epoch] = {}

                    for file in os.listdir(epoch_path):
                        if ".txt" in file:
                            with open(os.path.join(epoch_path, file), "r") as f:
                                text = f.read()
                                values = re.findall(r"([0-9]{0,2}\.[0-9]{2})(?!\|)(?![0-9])", text)

                                # just print to terminal
                                print(f"\t\t{values}")

                                # save in comprehensive dict
                                results[model][try_nr][epoch] = values

                                # write to file
                                with open(txt_file, "a") as write_file:
                                    write_file.write(f"\t\t{values}\n")

                                # add to list/array
                                try:
                                    results_per_try.append([float(item) for item in values])
                                    epochs_per_try.append(float(epoch))
                                    n_epochs += 1
                                except ValueError:
                                    print("cannot reshape array probably because there was no evaluation data present")

            results_per_try = np.array(results_per_try)
            epochs_per_try = np.array(epochs_per_try)[:, None]

            try:
                results_per_try = results_per_try.reshape((n_epochs, -1))

                df = pd.DataFrame(np.concatenate((epochs_per_try, results_per_try), axis=1))
                df.to_excel(f"epoch_evaluations/excel_sheets/model_{model}_try_{try_nr}.xlsx", index=False,
                            header=False)
            except ValueError:
                print("cannot reshape array probably because there was no evaluation data present")


def contrast_multivar_reg():
    with open("./data/contrast_difference.json", "r") as f:
        contrast_dict = json.load(f)

    contrast_dict['iqr_in'] = [q3 - q1 for q1, q3 in zip(contrast_dict['q1s_in'], contrast_dict['q3s_in'])]
    contrast_dict['iqr_out'] = [q3 - q1 for q1, q3 in zip(contrast_dict['q1s_out'], contrast_dict['q3s_out'])]

    contrast_dict['iqr_diff'] = [abs(iqr_in - iqr_out) for iqr_in, iqr_out in
                                 zip(contrast_dict['iqr_in'], contrast_dict['iqr_out'])]
    contrast_dict['std_diff'] = [abs(std_in - std_out) for std_in, std_out in
                                 zip(contrast_dict['stds_in'], contrast_dict['stds_out'])]
    contrast_dict['mean_diff'] = [abs(mean_in - mean_out) for mean_in, mean_out in
                                  zip(contrast_dict['means_in'], contrast_dict['means_out'])]
    contrast_dict['q2_diff'] = [abs(q2_in - q2_out) for q2_in, q2_out in
                                zip(contrast_dict['q2s_in'], contrast_dict['q2s_out'])]
    contrast_dict['overlap'] = [overlap(q1_in, q3_in, q1_out, q3_out) for q1_in, q1_out, q3_in, q3_out in
                                zip(contrast_dict['q1s_in'], contrast_dict['q1s_out'], contrast_dict['q3s_in'],
                                    contrast_dict['q3s_out'])]

    for k in contrast_dict:
        print(k)

    print(f"Overlap: {contrast_dict['overlap']}")
    print(f"Std diff: {contrast_dict['std_diff']}")
    print(f"Std in: {contrast_dict['stds_in']}")
    print(f"Std out: {contrast_dict['stds_out']}")
    print(f"IQR in: {contrast_dict['iqr_in']}")
    print(f"IQR out: {contrast_dict['iqr_out']}")
    print(f"IQR diff: {contrast_dict['iqr_diff']}")

    df = pd.DataFrame.from_dict(contrast_dict)
    print(df.head())

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    print(df.corr())

    x = df[['iqr_diff', 'std_diff']]
    y = df['nocs_hrnet']

    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()
    predictions = model.predict(x)

    print(model.summary())


def click_types():
    with open('./data/click_types.txt', 'r') as f:
        lines = f.read()

    numbers = re.findall(r'[0-9]+', lines)
    pos_clicks = numbers[0::2]
    neg_clicks = numbers[1::2]

    ratio = [int(a)/int(b) for a, b in zip(pos_clicks, neg_clicks)]
    print("Evaluation statistics")
    print(pos_clicks)
    print(neg_clicks)
    print(ratio)

    print("\nTraining statistics")

    with open('./data/click_types_training.txt', 'r') as f:
        lines = f.read()

    pos_clicks = re.findall(r'[0-9]+(?= positive)', lines)
    neg_clicks = re.findall(r'[0-9]+(?= negative)', lines)
    pos_clicks = [int(pos_click) for pos_click in pos_clicks]
    neg_clicks = [int(neg_click) for neg_click in neg_clicks]
    print(pos_clicks)

    print("Positive clicks")
    for i in range(int(len(pos_clicks) / 3)):
        print(int(sum(pos_clicks[3 * i:3*(i+1)]) * 73.3))
    print("Negative clicks")
    for i in range(int(len(neg_clicks) / 3)):
        print(int(sum(neg_clicks[3 * i:3*(i+1)]) * 73.3))
    print("Ratio [p/n]")
    for i in range(int(len(pos_clicks) / 3)):
        print(sum(pos_clicks[3 * i:3*(i+1)]) / sum(neg_clicks[3 * i:3*(i+1)]))


def estimated_training_time():
    # times for three epochs
    ritm_times = [322.389, 249.532, 172.284, 139.694, 725.928, 381.912, 177.82]
    cdnet_times = [87.176, 66.115, 44.097, 119.500, 198.257, 174.898, 47.193]
    fc_times = [197.260, 149.764, 100.879, 82.670, 444.904, 225.321, 106.612]

    print("RITM times")
    for time in ritm_times:
        print(str(datetime.timedelta(seconds=int(time * 73.3))))

    print("CDNet times")
    for time in cdnet_times:
        print(str(datetime.timedelta(seconds=int(time * 73.3))))

    print("FocalClick times")
    for time in fc_times:
        print(str(datetime.timedelta(seconds=int(time * 73.3))))


if __name__ == '__main__':
    if os.name == "nt":
        data_path = "Z:/Pancreas/fullPixelAnnotRedo2/"
    elif os.name == "posix":
        data_path = "/home/014118_emtic_oncology/Pancreas/fullPixelAnnotRedo2/"
    else:
        raise OSError("Code only implemented for either Windows or Linux")

    labels = ['aorta', 'arteria_mesenterica_superior', 'common_bile_duct', 'gastroduodenalis', 'pancreas',
              'pancreatic_duct', 'tumour']

    structures_hrnet = {'aorta': {'try': '001', 'epoch': 169, 'avg_mask': 3001},
                        'arteria_mesenterica_superior': {'try': '001', 'epoch': 119, 'avg_mask': 252},
                        'common_bile_duct': {'try': '001', 'epoch': 110, 'avg_mask': 501},
                        'gastroduodenalis': {'try': '001', 'epoch': 29, 'avg_mask': 61},
                        'pancreas': {'try': '002', 'epoch': 149, 'avg_mask': 979},
                        'pancreatic_duct': {'try': '000', 'epoch': 179, 'avg_mask': 162},
                        'tumour': {'try': '000', 'epoch': 159, 'avg_mask': 75}}

    # in_vs_out_statistics(labels, statistics=False)
    # process_results_txt('../ritm_interactive_segmentation/experiments/iter_mask')
    contrast_multivar_reg()
    # click_types()
    # estimated_training_time()

