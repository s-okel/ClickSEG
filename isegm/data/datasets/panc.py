import cv2
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


class PancDataset(ISDataset):
    def __init__(self, split, label, one_input_channel=False, data_path="./data/datasets/Panc/", **kwargs):
        super(PancDataset, self).__init__(**kwargs)
        assert split in ['train', 'val', 'test']
        self.name = "Panc"
        self.one_input_channel = one_input_channel

        self.data = torch.load(data_path + f"{label}_{split}_slices.pt")
        self.dataset_samples = range(len(self.data[0]))
        # self.dataset_samples = range(5)  # for testing purposes

    def get_sample(self, index) -> DSample:
        img = np.array(self.data[0, index]).astype("uint8")
        if not self.one_input_channel:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        mask = np.array(self.data[1, index]).astype("int32")

        return DSample(img, mask, objects_ids=[1], sample_id=index)


if __name__ == "__main__":
    lbl = "arteria_mesenterica_superior"
    dataset = PancDataset('val', lbl, one_input_channel=False,
                          data_path="C:/Users/320151982/source/repos/ClickSEG/data/datasets/Panc/")
    dataloader = DataLoader(dataset, shuffle=True)
    x = next(iter(dataloader))
    print(x['images'].shape)
    print(x['points'].shape)
    print(x['instances'].shape)
    print(x['images'][0].shape)

    image = torch.moveaxis(x['images'][0], 0, -1)
    instance = x['instances'][0, 0]

    f, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[1].imshow(image)
    axs[1].imshow(instance, alpha=0.2 * instance, cmap="Reds")
    plt.show()
