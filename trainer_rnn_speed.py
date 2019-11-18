import torch
import numpy as np
import torch.nn.functional as F
import cv2

from torch.autograd import Variable
import torch.optim as optim

import os
import time
import shutil
import pickle

from tqdm import tqdm
from utils.utils import AverageMeter
from model_rnn_speed import RNN_network
from tensorboard_logger import configure, log_value
from utils.visualization import blend_map_with_focus_rectangle, blend_map_with_focus_circle

from config import w,h
from utils.data.my_utils import read_image

from config import dreyeve_dir, tmp_dir

# get mean location of fixation
# def get_mean_loc(fixs):
#     fixs = fixs.numpy()
#     gt_locs = np.zeros([fixs.shape[0], 2])
#     for index in range(fixs.shape[0]):
#         fix = fixs[index]
#         gt_locs[index] = np.mean(np.where(fix>0), axis=1)
#
#     return torch.from_numpy(gt_locs/w-0.5)


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config
        self.dis_R_thres = config.dis_R_thres

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden
        self.use_speed_course = config.use_speed_course

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = 2048

        # reinforce params
        self.std = config.std
        self.M = config.M
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader)*config.batch_size
            self.num_valid = len(self.valid_loader)*config.batch_size
        else:
            self.test_loader = data_loader[1]
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 10
        self.num_channels = 3

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = 'ram_{}_{}x{}_{}'.format(
            config.num_glimpses, config.patch_size,
            config.patch_size, config.glimpse_scale
        )

        self.plot_dir = './plots/' + self.model_name + '/'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RNN_network(
            self.std, self.hidden_size,
        )
        if self.use_gpu:
            self.model.cuda()
        # train resnet or not
        # self.model.sensor.feature_extractor.eval()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # # initialize optimizer and scheduler
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=self.lr, momentum=self.momentum,
        # )
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer, 'min', patience=self.lr_patience
        # )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=3e-4,
        )

    def reset(self):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        dtype = (
            torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        )

        h_t = torch.zeros(self.batch_size, self.hidden_size)
        h_t = Variable(h_t).type(dtype)

        l_t = torch.Tensor(self.batch_size, 2).uniform_(-1, 1)
        l_t = Variable(l_t).type(dtype)

        return h_t, l_t

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)
            # train_loss, train_acc = self.train_one_epoch(epoch)

            # # reduce lr if validation loss plateaus
            # self.scheduler.step(valid_loss)

            # is_best = valid_acc > self.best_valid_acc
            is_best = 1
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            is_best =1
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                 }, is_best
            )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()

        tic = time.time()
        count = 0
        totalCount = 10000
        with tqdm(total=self.num_train) as pbar:
            for i, (x, fixation, y, speeds, courses, scale_gt, indexSeq, frameEnd) in enumerate(self.train_loader):
                if count > totalCount:
                    return losses.avg, losses.avg
                count = count + 1
                y= y.squeeze().float()

                if self.use_gpu:
                    x, y, speeds, courses, scale_gt = x.cuda(), y.cuda(), speeds.cuda(), courses.cuda(), scale_gt.cuda()
                x, y, speeds, courses, scale_gt = Variable(x), Variable(y), Variable(speeds), Variable(courses), Variable(scale_gt)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()

                # save images
                imgs = []
                imgs.append(x[0:9])

                # extract the glimpses
                locs = []
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t = self.model(x, speeds, courses, l_t, h_t, t)

                # last iteration
                h_t, l_t = self.model(
                    x, speeds, courses, l_t, h_t, self.num_glimpses-1, last=True
                )

                # compute losses
                loss = F.mse_loss(l_t, y)
                losses.update(loss.data, x.size()[0])

                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                print("Epoch: {} - {}/{} - {:.1f}s - loss: {:.3f}"
                      .format(epoch, count, len(self.train_loader), (toc-tic), loss.data))

                # dump the glimpses and locs
                if plot:
                    if self.use_gpu:
                        imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                        locs = [l.cpu().data.numpy() for l in locs]
                    else:
                        imgs = [g.data.numpy().squeeze() for g in imgs]
                        locs = [l.data.numpy() for l in locs]
                    pickle.dump(
                        imgs, open(
                            self.plot_dir + "g_{}.p".format(epoch+1),
                            "wb"
                        )
                    )
                    pickle.dump(
                        locs, open(
                            self.plot_dir + "l_{}.p".format(epoch+1),
                            "wb"
                        )
                    )

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch*len(self.train_loader) + i
                    log_value('train_loss', losses.avg, iteration)

            return losses.avg, losses.avg

    def validate(self, epoch):
        losses = AverageMeter()
        accs = AverageMeter()
        countTotal = 1000

        count = 0

        for i, (x, fixs, y, speeds, courses, scale_gt, indexSeq, frameEnd) in enumerate(self.valid_loader):
            if count > countTotal:
                return losses.avg, losses.avg
            count = count + 1
            y = y.squeeze().float()

            if self.use_gpu:
                x, y, speeds, courses, scale_gt = x.cuda(), y.cuda(), speeds.cuda(), courses.cuda(), scale_gt.cuda()
            x, y, speeds, courses, scale_gt = Variable(x), Variable(y), Variable(speeds), Variable(courses), Variable(
                scale_gt)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            locs = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t = self.model(x, speeds, courses, l_t, h_t, t)

            # last iteration
            h_t, l_t = self.model(
                x, speeds, courses, l_t, h_t, self.num_glimpses - 1, last=True
            )

            # compute losses for differentiable modules
            loss = F.mse_loss(l_t, y)

            # compute accuracy
            acc = loss / len(y)

            # print('avg dist is {}'.format(acc))
            losses.update(loss.data, x.size()[0])

        return losses.avg, losses.avg

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        count = 0
        f = open('output_rnn_speed.txt', 'w')

        for i, (x, fixs, y, speeds, courses, scale_gt, indexSeq, frameEnd) in enumerate(self.test_loader):
            count = count + 1
            y = y.squeeze().float()

            if self.use_gpu:
                x, y, speeds, courses, scale_gt = x.cuda(), y.cuda(), speeds.cuda(), courses.cuda(), scale_gt.cuda()
            x, y, speeds, courses, scale_gt = Variable(x), Variable(y), Variable(speeds), Variable(courses), Variable(
                scale_gt)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            locs = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t = self.model(x, speeds, courses, l_t, h_t, t)

            # last iteration
            h_t, l_t = self.model(
                x, speeds, courses, l_t, h_t, self.num_glimpses - 1, last=True
            )
            for indexBlend in range(x.shape[0]):
                loc = l_t[indexBlend, :].cpu().detach().numpy()
                loc_gt = y[indexBlend].cpu().numpy()
                # scale_blend = scale[indexBlend].cpu().detach().numpy()
                scale_gt_blend = scale_gt[indexBlend].cpu().detach().numpy()

                line = '{:02} {:04d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n' \
                    .format(indexSeq[indexBlend], frameEnd[indexBlend], loc_gt[0], loc_gt[1], loc[0], loc[1],
                            float(scale_gt_blend))
                print('seq: {:02}- frame: {:04d} - loc_gt_h: {:.3f} - loc_gt_w: {:.3f} - loc_h: {:.3f} - loc_w: {:.3f} '
                      '- scale_gt: {:.3f}' \
                      .format(indexSeq[indexBlend], frameEnd[indexBlend], loc_gt[0], loc_gt[1], loc[0], loc[1],
                              float(scale_gt_blend)))
                f.writelines(line)

            # compute losses for differentiable modules
            loss = F.mse_loss(l_t, y)

            # compute accuracy
            acc = loss / len(y)

            # print('avg dist is {}'.format(acc))
            losses.update(loss.data, x.size()[0])
            print("Test:{}/{} - loss: {:.3f}"
                  .format( count, len(self.test_loader), acc))
        print('avg dist is {}'.format(losses.avg))



    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )

if __name__ == '__main__':
    # test get_mean_loc
    print('test get_mean_loc')
    pathImg = '/home/lk/data/DREYEVE_DATA/01/frames/000001.jpg'
    pathFix = '/home/lk/data/DREYEVE_DATA/01/saliency_fix/000001.png'

    img = read_image(pathImg, channels_first=True, color=True)/255
    fix = read_image(pathFix, channels_first=True, color=False) / 255
    # fix = torch.from_numpy(fix)
    np.mean(np.where(fix ==np.max(fix)), axis=1)



