import os
import pathlib
import zipfile

import requests
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def download_dataset(data_path, image_path, log):
    if image_path.is_dir():
        log.info(f"{image_path} directory exists.")
    else:
        log.info(f"Did not find {image_path} directory, creating one...")
        # image_path.mkdir(parents=True, exist_ok=True)

        with open(data_path/"japanese_food.zip", "wb") as f:
            request = requests.get(
                "https://github.com/myatmyintzuthin/models-in-pytorch/releases/download/v.1.0/japanese_food.zip")
            log.info("Downloading japanese food...")
            f.write(request.content)

        with zipfile.ZipFile(data_path/"japanese_food.zip", "r") as zip_ref:
            log.info("Unzipping japanese food...")
            zip_ref.extractall(data_path)


def find_classes(directory: str):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f'Couldn\'t find any classes in {directory}')

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None):
        self.paths = list(pathlib.Path(targ_dir).glob('*/*.jpg'))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int):
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

class CustomDataloader():
    def __init__(self,data_dir: str, img_path: str, BATCH_SIZE: int, log: str, num_worker: int, shuffle: bool = True) -> None:
    
        self.batchsize = BATCH_SIZE
        self.shuffle = shuffle
        self.num_worker = num_worker

        self.train_dir = img_path/'train'
        self.test_dir = img_path/'test'
        self.valid_dir = img_path/'valid'

        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        download_dataset(data_path=data_dir, image_path=img_path, log=log)
    
    def train_dataloader(self):
        train_custom_data = ImageFolderCustom(targ_dir=self.train_dir, transform=self.train_transforms)
        train_custom_dataloader = DataLoader(dataset=train_custom_data, batch_size=self.batchsize, num_workers=self.num_worker, shuffle=self.shuffle)
        return train_custom_dataloader

    def valid_dataloader(self):
        valid_custom_data = ImageFolderCustom(targ_dir=self.valid_dir, transform=self.test_transform)
        valid_custom_dataloader = DataLoader(dataset=valid_custom_data, batch_size=self.batchsize, num_workers=self.num_worker, shuffle=False)
        return valid_custom_dataloader

    def test_dataloader(self):
        test_custom_data = ImageFolderCustom(targ_dir=self.test_dir, transform=self.test_transform)
        test_custom_dataloader = DataLoader(dataset=test_custom_data, batch_size=self.batchsize, num_workers=self.num_worker, shuffle=False)
        return test_custom_dataloader
