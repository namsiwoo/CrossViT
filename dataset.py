import json
import os, random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

from PIL import Image
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from scipy.ndimage.morphology import binary_dilation
import skimage.morphology, skimage.measure

class Dataset_siwoo(torch.utils.data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.root_dir = os.path.expanduser(self.args.data_dir)  # /media/NAS/nas_187/PATHOLOGY_DATA/MoNuSeg
        self.split = split

        # self.mean = np.array([123.675, 116.28, 103.53])
        # self.std = np.array([58.395, 57.12, 57.375])

        # Imagenet norm?
        self.mean = np.array([0.485,0.456,0.406])
        self.std = np.array([0.229,0.224,0.225])

        # create image augmentation
        if self.split == 'train':
            self.transform = get_transforms({ #without color aug
                'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': 224,
                'to_tensor': 1, # number of img
                'normalize': np.array([self.mean, self.std])
            })
        else:
            self.transform = get_transforms({
                'to_tensor': 1,
                'normalize': np.array([self.mean, self.std])
            })

        # read samples
        self.samples = self.read_samples(self.root_dir, self.split)

        # set num samples
        self.num_samples = len(self.samples)

        print('{} dataset {} loaded'.format(self.split, self.num_samples))

    def read_samples(self, root_dir, split):
        '''
        /path/to/imagenet/
          train/
            class1/
              img1.jpeg
            class2/
              img2.jpeg
          val/
            class1/
              img3.jpeg
            class/2
              img4.jpeg
        :return: num_classes (0 ~ 999)
        '''
        # if split == 'train':
        #     samples = os.listdir(os.path.join(root_dir, 'images', split))
        #
        # else:

        if self.split == 'train':
            with open(os.path.join(self.root_dir, 'train_val_test.json')) as f:
                split_dict = json.load(f)
            filename_list = split_dict[split]
            samples = [os.path.join(f) for f in filename_list]
        else:
            samples = os.listdir(os.path.join(root_dir, 'images', split))


        # samples = os.listdir(os.path.join(root_dir, 'images', split))
        return samples

    def __getitem__(self, index):
        img_name = self.samples[index % len(self.samples)]

        if self.split == 'train':
            # 1) read image
            img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')
            # img = Image.open(os.path.join(self.root_dir, self.split, 'images', img_name)).convert('RGB')
            # img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/CoNSeP/CoNSeP_class/patches', 'images', img_name)).convert('RGB')

            if self.sup == True:

                gt_label = np.array(Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+self.ext)))
                # gt_label = np.array(Image.open(os.path.join('/media/NAS/nas_70/open_dataset/CoNSeP/CoNSeP_class/patches', 'labels_instance', img_name[:-4]+self.ext)))

                # gt_label = np.array(Image.open(os.path.join(self.root_dir,  self.split, 'labels_instance', img_name[:-4]+self.ext)))
                # class_label = Image.open(os.path.join(self.root_dir,  self.split, 'labels_class', img_name[:-4]+self.ext))


                gt_label = skimage.morphology.label(gt_label)
                gt_label = Image.fromarray(gt_label.astype(np.uint16))

                class_label = Image.open(os.path.join(self.root_dir, 'labels_class', self.split, img_name[:-4]+self.ext))
                # class_label = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/CoNSeP/CoNSeP_class/patches', 'labels_class', img_name[:-4]+self.ext))
                if self.data_name == 'CoNSeP':
                    class_label = np.array(class_label)
                    class_label[class_label == 4] = 3
                    class_label[class_label > 4] = 4
                    class_label = Image.fromarray(class_label.astype(np.uint8))

                # 3) do image augmentation
                sample = [img, gt_label, class_label]  # , new_mask
                sample = self.transform(sample)
            else:
                # 2) read label

                class_label = Image.open(os.path.join(self.root_dir, 'labels_class', self.split, img_name[:-4]+self.ext))
                # class_label = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/CoNSeP/CoNSeP_class/patches', 'labels_class', img_name[:-4]+self.ext))
                if self.data_name == 'CoNSeP':
                    class_label = np.array(class_label)
                    class_label[class_label == 4] = 3
                    class_label[class_label > 4] = 4
                    class_label = Image.fromarray(class_label.astype(np.uint8))

                    point = np.load(os.path.join(self.root_dir, 'labels_point_class', self.split, img_name[:-4] + '_label_point.npy'))
                    new_point = np.zeros((point.shape[0], point.shape[1]))

                    for i in range(point.shape[-1]):
                        # p = np.zeros_like(new_point)
                        p = binary_dilation(point[:, :, i], iterations=2)
                        new_point[p] = i + 1
                    point = Image.fromarray(new_point.astype(np.uint16))

                else:
                    point = Image.open(os.path.join(self.root_dir, 'labels_point', self.split, img_name)).convert('L')
                    point = binary_dilation(np.array(point), iterations=2)
                    point = Image.fromarray(point)


                sample = [img, class_label, point]#, cluster_label, voronoi_label]  # , new_mask
                sample = self.transform(sample)


        else:
            # mask = Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4]+'_label.png')).convert('L')
            # mask = np.array(mask)
            if self.data_name == 'CoNSeP':
                root_dir = self.root_dir.split('/')
                new_dir = ''
                for dir in root_dir[:-2]:
                    new_dir += dir + '/'

                img = Image.open(os.path.join(new_dir, 'images', img_name)).convert('RGB')
                mask = Image.open(os.path.join(new_dir, 'labels_instance', img_name[:-4] + self.ext))
                class_label = Image.open(os.path.join(new_dir, 'labels_class', img_name[:-4] + self.ext))
                class_label = np.array(class_label)
                class_label[class_label==4] = 3
                class_label[class_label>4] = 4
                class_label = Image.fromarray(class_label.astype(np.uint8))

            elif self.data_name == 'pannuke':
                img = Image.open(os.path.join(self.root_dir, 'images', self.split, img_name)).convert('RGB')
                mask = Image.open(os.path.join(self.root_dir, 'labels_instance', self.split, img_name[:-4] + self.ext))
                class_label = Image.open(os.path.join(self.root_dir, 'labels_class', self.split, img_name[:-4] + self.ext))

            else:
                img = Image.open(os.path.join(self.root_dir, self.split, 'ASAN_patches', img_name)).convert('RGB')
                mask = Image.open(os.path.join(self.root_dir, self.split, 'labels_instance', 'ASAN_instance', img_name[:-4] + self.ext))
                class_label = Image.open(os.path.join(self.root_dir, self.split, 'labels_class', 'ASAN_class', img_name[:-4] + self.ext))

            # img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/images/val', img_name)).convert('RGB')
            # mask = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/pannuke/Pannuke_patch/labels_instance/val', img_name))
            # img = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/images/val', img_name)).convert('RGB')
            # mask = Image.open(os.path.join('/media/NAS/nas_70/open_dataset/MoNuSAC/MoNuSAC/labels_instance/val2', img_name))


            sample = [img, mask, class_label]
            sample = self.transform(sample)

        return sample, str(img_name[:-4])

    def __len__(self):
        return self.num_samples