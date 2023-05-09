import os
import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

from settings import set_seed
from config import RANDOM_SEED
from config import BATCH_SIZE
from data import generate_df_dataset
from data import RoadSignDataset
from data import RoadSignDataLoader
from model import RoadSignModel
from callbacks import ModelCheckpoint
from callbacks import EarlyStopping


def train_on_epoch(model, data_loader, optimizer, C=1000):
    model.train()
    total_num = 0
    total_loss = 0
    total_acc = 0
    for _, batch_data in enumerate(data_loader):
        batch_img, batch_cls, batch_bbox = batch_data
        batch_size = batch_cls.shape[0]

        # Reset optimizer
        optimizer.zero_grad()

        # Forward
        out_cls, out_bbox = model(batch_img)

        # Loss
        loss_cls = F.cross_entropy(out_cls, batch_cls, reduction='sum')
        loss_reg = F.l1_loss(out_bbox, batch_bbox, reduction='none')
        loss_reg = loss_reg.sum(1)  # xmin + ymin + xmax + ymax
        loss_reg = loss_reg.sum()
        loss = loss_cls + loss_reg / C

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        _, pred = torch.max(out_cls, 1)
        acc = pred.eq(batch_cls).sum()

        total_num += batch_size
        total_loss += loss.item()
        total_acc += acc.item()

    train_loss = total_loss / total_num
    train_acc = total_acc / total_num

    return train_loss, train_acc


def valid_on_epoch(model, data_loader, C=1000):
    model.eval()
    total_num = 0
    total_loss = 0
    total_acc = 0
    for _, batch_data in enumerate(data_loader):
        batch_img, batch_cls, batch_bbox = batch_data
        batch_size = batch_cls.shape[0]

        # Forward
        out_cls, out_bbox = model(batch_img)

        # Loss
        loss_cls = F.cross_entropy(out_cls, batch_cls, reduction='sum')
        loss_reg = F.l1_loss(out_bbox, batch_bbox, reduction='none')
        loss_reg = loss_reg.sum(1)  # xmin + ymin + xmax + ymax
        loss_reg = loss_reg.sum()
        loss = loss_cls + loss_reg / C

        # Metrics
        _, pred = torch.max(out_cls, 1)
        acc = pred.eq(batch_cls).sum()

        total_num += batch_size
        total_loss += loss.item()
        total_acc += acc.item()

    valid_loss = total_loss / total_num
    valid_acc = total_acc / total_num

    return valid_loss, valid_acc


if __name__ == '__main__':
    set_seed(RANDOM_SEED)

    dataset_root = os.path.join(os.path.expanduser('~'), 'road-sign-dataset')
    img_path = os.path.join(dataset_root, 'images')
    anno_path = os.path.join(dataset_root, 'annotations')

    df_dataset = generate_df_dataset(anno_path)
    # train:valid:test=8:1:1
    df_train, df_valid = train_test_split(df_dataset, test_size=0.2, random_state=RANDOM_SEED)
    df_valid, df_test = train_test_split(df_valid, test_size=0.5, random_state=RANDOM_SEED)

    trainDS = RoadSignDataset(df_train, img_path, mode='train')
    validDS = RoadSignDataset(df_valid, img_path, mode='valid')
    trainDL = RoadSignDataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True)
    validDL = RoadSignDataLoader(validDS, batch_size=BATCH_SIZE)

    weights_file = os.path.join(os.path.expanduser('~'), 'road-sign-detector', 'weights', 'best.pt')
    if not os.path.isdir(os.path.dirname(weights_file)):
        os.makedirs(os.path.dirname(weights_file))

    model = RoadSignModel()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    modelCheckpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=True)
    earlyStopping = EarlyStopping(patience=7, verbose=True)

    total_epochs = 50
    for epoch in range(total_epochs):
        epoch = epoch + 1

        start_time = time.time()
        train_loss, train_acc = train_on_epoch(model, trainDL, optimizer)
        valid_loss, valid_acc = valid_on_epoch(model, validDL)
        elapsed_time = time.time() - start_time
        print('epoch - [{}/{}]  train_loss - {:.4f}  valid_loss - {:.4f}  train_acc - {:.4f}  valid_acc - {:.4f}  time - {:.4f}s'.format(
            epoch,
            total_epochs,
            train_loss,
            valid_loss,
            train_acc,
            valid_acc,
            elapsed_time
        ))

        earlyStopping(valid_loss)
        modelCheckpoint(valid_loss, model)