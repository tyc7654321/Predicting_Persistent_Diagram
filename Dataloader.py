from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
from ripser import lower_star_img
from persim import PersImage
from persim import PersistenceImager
import torch
import numpy as np

class CIFAR10dataset(Dataset):

    def __init__(self, dataset):
        self.data = dataset
        self.transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),
             #transforms.RandomRotation(30),
             transforms.ToTensor(),
             #GaussianBlur(kernel_size=3, sigma=(0.1,0.5)),
             transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = self.transform(image)
        sq_image = image.squeeze(0)
        pdgm = lower_star_img(sq_image)
        pdgm = pdgm[:-1,:]
        #lifetime = np.expand_dims((pdgm[:,1] - pdgm[:,0]), axis=1)
        #pdgm = np.concatenate((pdgm, lifetime), axis=1)
        #pdgm = np.asarray(sorted(pdgm, key=lambda x: x[1]-x[0], reverse=True))
        if pdgm.shape[0] >= 20:
            lifetime = -np.sort(pdgm[:,0] - pdgm[:,1])[:20]
        else:
            lifetime = np.concatenate((-np.sort(pdgm[:,0] - pdgm[:,1]),np.zeros(20 - pdgm.shape[0])))

        #if pdgm.shape[0] >= 20:
            #pdgm = torch.reshape(torch.tensor(pdgm[:20,:]), (1,40))
        #else:
            #pdgm = np.concatenate((pdgm, np.zeros((20 - pdgm.shape[0],2))))
            #pdgm = torch.reshape(torch.tensor(pdgm),(1, 40))

        #pdgm = torch.squeeze(pdgm, dim=0)

        #pimgr = PersistenceImager(pixel_size=0.02, birth_range=(-1,0))
        #pimgr.fit(pdgm, skew=True)
        #pimg = pimgr.transform(pdgm, skew=True)

        return image, lifetime, label
