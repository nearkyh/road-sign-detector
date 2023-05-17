import os
from glob import glob
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision as tv
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class_id = {
    'speedlimit': 0,
    'stop': 1,
    'crosswalk': 2,
    'trafficlight': 3,
}
class_label = {
    0: 'speedlimit',
    1: 'stop',
    2: 'crosswalk',
    3: 'trafficlight',
}

transform_config = {
    'default': A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['cls_label'],
        )
    ),
    'train': A.Compose(
        [
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['cls_label'],
        )
    ),
    'demo': tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


def create_df_dataset(anno_path):
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


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img / 255.0 - mean) / std
    # Resize
    img = cv2.resize(img, (224, 224))
    # To tensor
    img = torch.tensor(img, dtype=torch.float32)
    # Add batch dimension
    img = img.unsqueeze(0)
    # (B, H, W ,C) --> (B, C, W, H)
    img = img.permute(0, 3, 2, 1)

    return img


class RoadSignDataset(Dataset):

    def __init__(self, df, img_path, mode='default'):
        self.df = df
        self.img_path = img_path
        self.mode = mode.lower()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_obj = self.df.iloc[idx]

        # Image
        img_name = df_obj['filename']
        img_file = os.path.join(self.img_path, img_name)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Object class
        cls_id = class_id[df_obj['class']]

        # Object bounding box
        bbox = [df_obj['xmin'], df_obj['ymin'], df_obj['xmax'], df_obj['ymax']]

        # Data augmentation
        if self.mode == 'train':
            mode = self.mode
        else:
            mode = 'default'
        transformed = transform_config[mode](
            image=img,
            bboxes=[bbox],
            cls_label=[cls_id]
        )
        img = transformed['image']
        bbox = transformed['bboxes'][0]
        cls_id = transformed['cls_label'][0]

        bbox = np.array(bbox).astype(np.int32)

        return img, cls_id, bbox


class RoadSignDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=False):
        super().__init__(dataset, batch_size, shuffle)


if __name__ == '__main__':
    dataset_root = os.path.join(os.path.expanduser('~'), 'road-sign-dataset')
    img_path = os.path.join(dataset_root, 'images')
    anno_path = os.path.join(dataset_root, 'annotations')

    df_dataset = create_df_dataset(anno_path)
    roadSignDS = RoadSignDataset(df_dataset, img_path, mode='test')
    roadSignDL = RoadSignDataLoader(roadSignDS, batch_size=1)

    for _, batch_data in enumerate(roadSignDL):
        batch_img, batch_cls, batch_bbox = batch_data
        print(batch_img)
        print(batch_img.shape, batch_cls, batch_bbox)

        img = batch_img[0].permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        xmin, ymin, xmax, ymax = batch_bbox[0].numpy()
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break