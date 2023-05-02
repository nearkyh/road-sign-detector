import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class_idx = {
    'speedlimit': 0,
    'stop': 1,
    'crosswalk': 2,
    'trafficlight': 3
}

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}


def generate_df_dataset(anno_path):
    anno_files = glob(os.path.join(anno_path, '*.xml'))
    anno_list = []
    for anno_file in anno_files:
        xml_root = ET.parse(anno_file).getroot()
        anno = {}
        anno['filename'] = xml_root.find("./filename").text
        anno['width'] = xml_root.find("./size/width").text
        anno['height'] = xml_root.find("./size/height").text
        anno['class'] = xml_root.find("./object/name").text
        anno['xmin'] = int(xml_root.find("./object/bndbox/xmin").text)
        anno['ymin'] = int(xml_root.find("./object/bndbox/ymin").text)
        anno['xmax'] = int(xml_root.find("./object/bndbox/xmax").text)
        anno['ymax'] = int(xml_root.find("./object/bndbox/ymax").text)
        anno_list.append(anno)

    return pd.DataFrame(anno_list)


class RoadSignDataset(Dataset):

    def __init__(self, df, img_path, transforms=None):
        self.df = df
        self.img_path = img_path
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        df_obj = self.df.iloc[idx]
        img_name = df_obj['filename']
        img_file = os.path.join(self.img_path, img_name)
        label = class_idx[df_obj['class']]
        bbox = np.array([
            df_obj['xmin'], df_obj['ymin'], df_obj['xmax'], df_obj['ymax']
        ])

        img = Image.open(img_file).convert('RGB')
        if self.transforms:
            img = self.transforms(img)

        return img, label, bbox
