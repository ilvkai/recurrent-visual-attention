import os.path as osp

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import os.path as osp
import cv2
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from utils.serialization import load_checkpoint

def draw_curve(path, x_epoch, train_loss, train_prec):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="prec")
    ax0.plot(x_epoch, train_loss, 'bo-', label='train')
    ax1.plot(x_epoch, train_prec, 'bo-', label='train')
    ax0.legend()
    ax1.legend()
    fig.savefig(path)
    plt.close(fig)

def read_image(img_path, channels_first, color=True, color_mode='BGR', dtype=np.float32, resize_dim=None):

    """
    Reads and returns an image as a numpy array

    Parameters
    ----------
    img_path : string
        Path of the input image
    channels_first: bool
        If True, channel dimension is moved in first position
    color: bool, optional
        If True, image is loaded in color: grayscale otherwise
    color_mode: "RGB", "BGR", optional
        Whether to load the color image in RGB or BGR format
    dtype: dtype, optional
        Array is casted to this data type before being returned
    resize_dim: tuple, optional
        Resize size following convention (new_h, new_w) - interpolation is linear

    Returns
    -------
    image : np.array
        Loaded Image as numpy array of type dtype
    """

    if not osp.exists(img_path):
        raise ValueError('Provided path "{}" does NOT exist.'.format(img_path))

    image = cv2.imread(img_path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)

    if color and color_mode == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if resize_dim is not None:
        image = cv2.resize(image, dsize=resize_dim[::-1], interpolation=cv2.INTER_LINEAR)

    if color and channels_first:
        image = np.transpose(image, (2, 0, 1))

    return image.astype(dtype)

def get_data(name, data_dir, height, width, batch_size, workers,
             combine_trainval, crop, tracking_icams, fps, re=0, num_instances=0, camstyle=0, zju=0, colorjitter=0):
    root = osp.join(data_dir, name)
    if name == 'duke_tracking':
        if tracking_icams != 0:
            tracking_icams = [tracking_icams]
        else:
            tracking_icams = list(range(1, 9))
        dataset = datasets.create(name, root, type='tracking_gt', iCams=tracking_icams, fps=fps,
                                  trainval=combine_trainval)
    elif name == 'aic_tracking':
        dataset = datasets.create(name, root, type='tracking_gt', fps=fps, trainval=combine_trainval)
    else:
        dataset = datasets.create(name, root)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.ColorJitter(brightness=0.1 * colorjitter, contrast=0.1 * colorjitter, saturation=0.1 * colorjitter, hue=0),
        T.Resize((height, width)),
        T.RandomHorizontalFlip(),
        T.Pad(10 * crop),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=re),
    ])
    test_transformer = T.Compose([
        T.Resize((height, width)),
        # T.RectScale(height, width, interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    if zju:
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.train_path, transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            sampler=ZJU_RandomIdentitySampler(dataset.train, batch_size, num_instances) if num_instances else None,
            shuffle=False if num_instances else True, pin_memory=True, drop_last=False if num_instances else True)
    else:
        train_loader = DataLoader(
            Preprocessor(dataset.train, root=dataset.train_path, transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            sampler=RandomIdentitySampler(dataset.train, num_instances) if num_instances else None,
            shuffle=False if num_instances else True, pin_memory=True, drop_last=True)
    query_loader = DataLoader(
        Preprocessor(dataset.query, root=dataset.query_path, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=dataset.gallery_path, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    if camstyle <= 0:
        camstyle_loader = None
    else:
        camstyle_loader = DataLoader(
            Preprocessor(dataset.camstyle, root=dataset.camstyle_path,
                         transform=train_transformer),
            batch_size=camstyle, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)
    return dataset, num_classes, train_loader, query_loader, gallery_loader, camstyle_loader


def checkpoint_loader(model, path, eval_only=False):
    checkpoint = load_checkpoint(path)
    pretrained_dict = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        Parallel = 1
        model = model.module.cpu()
    else:
        Parallel = 0

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if eval_only:
        keys_to_del = []
        for key in pretrained_dict.keys():
            if 'classifier' in key:
                keys_to_del.append(key)
        for key in keys_to_del:
            del pretrained_dict[key]
        pass
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    start_epoch = checkpoint['epoch']
    best_top1 = checkpoint['best_top1']

    if Parallel:
        model = nn.DataParallel(model).cuda()

    return model, start_epoch, best_top1
