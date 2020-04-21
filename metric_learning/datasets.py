import torchvision.transforms as T
import numpy as np
import torch
import os

from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset
from PIL import Image

class MMFashion(Dataset):
    """CLASS:MMFasion:
        >- Dataset to extract only images and targets
    """
    def __init__(self, data_dir, img_txt, img_size, categories):
        # -- DEFINE TRANSFORMS -- #
        self.transform = T.Compose([
            T.RandomResizedCrop(img_size[0]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # -- CONSTANTS -- #
        self.data_dir = data_dir
        self.img_size = img_size
        self.categories = categories

        # -- COLLECT CLASS ID'S -- #
        fp = open(img_txt, 'r')
        self.img_list = [x.strip() for x in fp]
        self.class_ids = []
        for img in self.img_list:
            self.class_ids.append(self.categories[img.split("/")[2]])
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.img_list[index]))
        target = torch.tensor(self.class_ids[index])
        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = img.convert('RGB')
        img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.img_list)

class SiameseMMFashion(MMFashion):
    """CLASS::SiameseMMFashion:
        >- Siamese Dataset for MMfashion loading positive and negative examples
    """
    def __init__(self, data_dir, img_txt, img_size, categories, train_flag):
        super(SiameseMMFashion, self).__init__(data_dir, img_txt, img_size, categories)
        self.train_flag = train_flag
        if self.train_flag:
            self.train_labels = torch.tensor(self.class_ids)
            self.train_data = self.img_list
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            self.test_labels = torch.tensor(self.class_ids)
            self.test_data = self.img_list
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        index1 = None
        index2 = None
        if self.train_flag:
            target = np.random.randint(0, 2)
            index1 = index
            img1, label1 = self.train_data[index1], self.train_labels[index1].item()
            if target == 1:
                siamese_index = index1
                while siamese_index == index1:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            index2 = siamese_index
            img2 = self.train_data[index2]
        else:
            index1 = self.test_pairs[index][0]
            index2 = self.test_pairs[index][1]
            img1 = self.test_data[index1]
            img2 = self.test_data[index2]
            target = self.test_pairs[index][2]

        return self._create_images(img1, img2, index1, index2), target

    def _create_images(self, img1, img2, idx1, idx2):
        image1 = self.get_mmfashion_img(img1, idx1)
        image2 = self.get_mmfashion_img(img2, idx2)
        return image1, image2

    def get_mmfashion_img(self, img, idx):
        image = Image.open(os.path.join(self.data_dir, img))
        image.thumbnail(self.img_size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_list)

class TripletMMFashion(MMFashion):
    """CLASS:TripleMMFashion
        >- Triplet Dataset for MMFashion
    """
    def __init__(self, data_dir, img_txt, img_size, categories, train_flag):
        super(TripletMMFashion, self).__init__(data_dir, img_txt, img_size, categories)
        if self.train:
            self.train_labels = torch.tensor(self.class_ids)
            self.train_data = self.img_list
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = torch.tensor(self.class_ids)
            self.test_data = self.img_list
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets
    
    def __getitem__(self, index):
        index1 = None
        index2 = None
        index3 = None
        if self.train:
            index1 = index
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            index2 = positive_index
            index3 = negative_index
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            index1 = self.test_triplets[index][0]
            index2 = self.test_triplets[index][1]
            index3 = self.test_triplets[index][2]
            img1 = self.test_data[index1]
            img2 = self.test_data[index2]
            img3 = self.test_data[index3]

        return self._create_images(img1, img2, img3, index1, index2, index3), []

    def _create_images(self, img1, img2, img3, idx1, idx2, idx3):
        image1 = self.get_mmfashion_img(img1, idx1)
        image2 = self.get_mmfashion_img(img2, idx2)
        image3 = self.get_mmfashion_img(img3, idx3)
        return image1, image2, image3

    def get_mmfashion_img(self, img, idx):
        image = Image.open(os.path.join(self.data_dir, img))
        image.thumbnail(self.img_size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_list)

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size