import torch
from torch.utils.data import random_split
from torchinfo import summary
import os, json

import torchvision
from torchvision import transforms as tf

CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    )


def class_to_num(class_str):
    for idx, string in enumerate(CLASSES):
        if string == class_str: return idx

def num_to_class(number):
    for idx, string in enumerate(CLASSES):
        if idx == number: return string
    return 'none'

class VOCNormalize:
    """
    Normalize images and bounding boxes.
    The actual transformation is later done on the GPU using TransformGenerator class.
        :param image: 4D Tensor of shape [B, C, H, W]
        :param target: Tensor containing the bounding boxes for the given images.
            Shape: [CenterX, CenterY, W, H, Confidence, Class Label]
    """
    def __init__(self, train=True, only_person=False):
        self.only_person = only_person
        self.train = train

    def __call__(self,image, target):
        num_bboxes = 10
        width, height = 320, 320

        img_width, img_height = image.size

        scale = min(width/ img_width, height/img_height)
        new_width, new_height = int(img_width * scale), int( img_height * scale)

        diff_width, diff_height = width - new_width, height - new_height
        image = tf.functional.resize(image, size=(new_height, new_width))
        image = tf.functional.pad(image, padding = (diff_width//2,
                                                            diff_height//2,
                                                            diff_width//2 + diff_width % 2,
                                                            diff_height//2 + diff_height % 2))
        target = target['annotation']['object']

        target_vectors = []
        for item in target:
            x0 = int(item['bndbox']['xmin'])*scale + diff_width//2
            w = (int(item['bndbox']['xmax']) - int(item['bndbox']['xmin']))* scale
            y0 = int(item['bndbox']['ymin'])*scale + diff_height//2
            h = (int(item['bndbox']['ymax']) - int(item['bndbox']['ymin'])) * scale

            target_vector = [(x0 + w/2) / width,                # Center X (cx)
                            (y0 + h/2) / height,                # Center Y (cy)
                            w/width,                            # Width (w)
                            h/height,                           # Height (h)
                            1.0,                                # Objectness / Confidence
                            class_to_num(item['name'])]         # Class Label

            if self.only_person:
                if target_vector[5] == class_to_num("person"):
                    target_vector[5] = 0.0
                    target_vectors.append(target_vector)
            else:
                target_vectors.append(target_vector)

        target_vectors = list(sorted(target_vectors, key=lambda x: x[2]*x[3]))

        # Set target vectors if they are not empty, else fill with zeroes
        target_vectors = torch.tensor(target_vectors) if len(target_vectors) != 0 else torch.zeros((0, 6))

        if target_vectors.shape[0] < num_bboxes:
            zeros = torch.zeros((num_bboxes - target_vectors.shape[0], 6))
            zeros[:, -1] = -1
            target_vectors = torch.cat([target_vectors, zeros], 0)
        elif target_vectors.shape[0] > num_bboxes:
            target_vectors = target_vectors[:num_bboxes]

        return tf.functional.to_tensor(image), target_vectors


def VOCDataLoader(split="train", batch_size=32, shuffle=None):

    if shuffle is None:
        shuffle = (split == "train")
    
    image_set = "train" if split == "train" else "val"

    if not os.path.exists("data/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg"):
        dataset = torchvision.datasets.VOCDetection("data/", year="2012", image_set=image_set, download=True)

    dataset = torchvision.datasets.VOCDetection("data/", year="2012", image_set=image_set, download=False, transforms=VOCNormalize(train=(split == "train")))

    # split validation set into validation and test set
    if split in ["val", "test"]:
        total_size = len(dataset)
        val_size = total_size // 2
        test_size = total_size - val_size
        
        generator = torch.Generator().manual_seed(0)
        val_dataset, test_dataset = random_split(dataset, [val_size, test_size], generator=generator)
        
        dataset = val_dataset if split == "val" else test_dataset

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
def VOCDataLoaderPerson(split="train", batch_size=32, shuffle=None):

    if shuffle is None:
        shuffle = (split == "train")
    
    image_set = "train" if split == "train" else "val"

    if not os.path.exists("data/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg"):
        dataset = torchvision.datasets.VOCDetection("data/", year="2012", image_set=image_set, download=True)
        
    dataset = torchvision.datasets.VOCDetection("data/", year="2012", image_set=image_set, download=False,
                                transforms=VOCNormalize(train=(split == "train"), only_person=True))
    with open("data/person_indices.json", "r") as fd: indices = list(json.load(fd)[image_set])
    dataset = torch.utils.data.Subset(dataset, indices)

    # split validation set into validation and test set
    if split in ["val", "test"]:
        total_size = len(dataset)
        val_size = total_size // 2
        test_size = total_size - val_size
        
        generator = torch.Generator().manual_seed(0)
        val_dataset, test_dataset = random_split(dataset, [val_size, test_size], generator=generator)
        
        dataset = val_dataset if split == "val" else test_dataset

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)