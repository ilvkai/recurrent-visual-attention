import torch

from trainer import Trainer
from config import get_config
from utils.utils import prepare_dirs, save_config
from data_loader import get_test_loader, get_train_valid_loader
from dataset.dreyeve_seq import Dreyeve as Dataset


def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    dataset = Dataset('train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=12,
                                               pin_memory=True)

    test_dataset = Dataset('test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=12,
                                                 pin_memory=True)


    # instantiate trainer
    trainer = Trainer(config, (train_loader, test_loader))

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
