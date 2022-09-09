from termcolor import colored
import json
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd
import regex as re
from scipy.ndimage import binary_dilation, binary_erosion


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
    results = {}
    txt_file = "epoch_evaluations/epoch_evaluations.txt"

    if os.path.exists(txt_file):
        os.remove(txt_file)
    for model in os.listdir(experiments_path):
        if "hrnet" in model:
            continue
        model_path = experiments_path + model + "/"
        print(colored(f"Model name: {model}", "green"))
        with open(txt_file, "a") as write_file:
            write_file.write(f"Model name: {model}\n")
        results[model] = {}
        for try_folder in os.listdir(model_path):
            results_per_try = []
            epochs_per_try = []
            try_nr = try_folder.split("_")[0]
            print(colored(f"\tTry: {try_nr}", "red"))
            with open(txt_file, "a") as write_file:
                write_file.write(f"\tTry: {try_nr}\n")
            results[model][try_nr] = {}
            epochs_path = model_path + try_folder + "/evaluation_logs/"
            if os.path.exists(epochs_path):
                for epoch in os.listdir(epochs_path):
                    epoch_path = epochs_path + epoch + "/"
                    epoch = epoch.split("-")[1]
                    print(colored(f"\t\tEpoch: {epoch}", "yellow"))
                    with open(txt_file, "a") as write_file:
                        write_file.write(f"\t\tEpoch: {epoch}\n")
                    results[model][try_nr][epoch] = {}
                    for file in os.listdir(epoch_path):
                        if ".txt" in file:
                            with open(epoch_path + file, "r") as f:
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
                                results_per_try.append([float(item) for item in values])
                                epochs_per_try.append(float(epoch))

            results_per_try = np.array(results_per_try)
            epochs_per_try = np.array(epochs_per_try)[:, None]

            results_per_try = results_per_try.reshape((-1, 9))

            df = pd.DataFrame(np.concatenate((epochs_per_try, results_per_try), axis=1))
            df.to_excel(f"epoch_evaluations/model_{model}_try_{try_nr}.xlsx", index=False, header=False)


if __name__ == '__main__':
    if os.name == "nt":
        data_path = "Z:/Pancreas/fullPixelAnnotRedo2/"
    elif os.name == "posix":
        data_path = "/home/014118_emtic_oncology/Pancreas/fullPixelAnnotRedo2/"
    else:
        raise OSError("Code only implemented for either Windows or Linux")

    labels = ['aorta', 'arteria_mesenterica_superior', 'common_bile_duct', 'gastroduodenalis', 'pancreas',
              'pancreatic_duct', 'tumour']

    in_vs_out_statistics(labels, statistics=False)
