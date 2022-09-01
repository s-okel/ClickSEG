import json
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import binary_dilation, binary_erosion

# TODO: not only further from boundary, but also a bit inside


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


if __name__ == '__main__':
    if os.name == "nt":
        data_path = "Z:/Pancreas/fullPixelAnnotRedo2/"
    elif os.name == "posix":
        data_path = "/home/014118_emtic_oncology/Pancreas/fullPixelAnnotRedo2/"
    else:
        raise OSError("Code only implemented for either Windows or Linux")

    labels = ['aorta', 'arteria_mesenterica_superior', 'common_bile_duct', 'gastroduodenalis', 'pancreas',
              'pancreatic_duct', 'tumour']
    structure_boundaries(labels)
