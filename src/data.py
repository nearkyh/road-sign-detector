import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class_idx = {
    'speedlimit': 0,
    'stop': 1,
    'crosswalk': 2,
    'trafficlight': 3
}

transform_config = {
    'default': transforms.Compose([
        transforms.ToTensor(),
    ]),
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
    ]),
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

    def __init__(self, df, img_path, mode='default'):
        self.df = df
        self.img_path = img_path
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_obj = self.df.iloc[idx]
        img_name = df_obj['filename']
        img_file = os.path.join(self.img_path, img_name)
        cls = class_idx[df_obj['class']]
        bbox = np.array([
            df_obj['xmin'], df_obj['ymin'], df_obj['xmax'], df_obj['ymax']
        ])

        img = Image.open(img_file).convert('RGB')
        img = transform_config[self.mode](img)

        return img, cls, bbox


class RoadSignDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=False):
        super().__init__(dataset, batch_size, shuffle)


if __name__ == '__main__':
    import cv2

    dataset_root = os.path.join(os.path.expanduser('~'), 'road-sign-dataset')
    img_path = os.path.join(dataset_root, 'images')
    anno_path = os.path.join(dataset_root, 'annotations')

    df_dataset = generate_df_dataset(anno_path)
    roadSignDS = RoadSignDataset(df_dataset, img_path)
    roadSignDL = RoadSignDataLoader(roadSignDS, batch_size=1)

    for _, batch_data in enumerate(roadSignDL):
        batch_img, batch_cls, batch_bbox = batch_data
        print(batch_img.shape, batch_cls.shape, batch_bbox.shape)

        img = batch_img[0].permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break