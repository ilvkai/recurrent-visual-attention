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
from model import RecurrentAttention
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

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

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
        self.model = RecurrentAttention(
            self.patch_size, self.num_patches, self.glimpse_scale,
            self.num_channels, self.loc_hidden, self.glimpse_hidden,
            self.std, self.hidden_size, self.num_classes,
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
        accs = AverageMeter()

        losses_action = AverageMeter()
        countTotal = 1000

        tic = time.time()
        count=0
        with tqdm(total=self.num_train) as pbar:
            for i, (x, fixation, y, speeds, courses, scale_gt, indexSeq, frameEnd) in enumerate(self.train_loader):
                if count > countTotal:
                    return losses.avg, accs.avg
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
                log_pi = []
                baselines = []
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, speeds, courses, l_t, h_t, t)

                    # store
                    locs.append(l_t[0:9])
                    baselines.append(b_t)
                    log_pi.append(p)

                # last iteration
                h_t, l_t, b_t, l_t_final, p, scale = self.model(
                    x, speeds, courses, l_t, h_t, self.num_glimpses-1, last=True
                )
                log_pi.append(p)
                baselines.append(b_t)
                locs.append(l_t[0:9])

                # convert list to tensors and reshape
                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)

                # calculate reward
                # predicted = torch.max(log_probas, 1)[1]
                # R = (predicted.detach() == y).float()

                R = torch.zeros(y.shape[0])
                for index in range(y.shape[0]):
                    # get the distance of two locations
                    distance = torch.sqrt(torch.pow(l_t_final[index,0]-y[index,0], 2) + torch.pow(l_t_final[index,1]-y[index,1], 2)).float()
                    # R[index] = distance < self.dis_R_thres
                    # temp= distance < self.dis_R_thres
                    R[index] = 1 - distance
                # R = locs
                mean_R = torch.mean(R)
                R = R.unsqueeze(1).repeat(1, self.num_glimpses).to(self.device)

                # compute losses for differentiable modules
                # loss_action = F.nll_loss(log_probas, y)
                loss_action = F.mse_loss(l_t_final, y)
                loss_scale = F.mse_loss(scale, scale_gt)
                # if loss_action.data > 1:
                #     print('loss_action > 1 and l_t_final = {} y = {}'.format(l_t_final.data, y.data))
                loss_baseline = F.mse_loss(baselines, R)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)

                # sum up into a hybrid loss
                loss = loss_action * 10 + loss_baseline *10 + loss_reinforce + loss_scale
                # loss =  loss_baseline + loss_reinforce

                # compute accuracy
                # correct = dis.float()
                # acc = 100 * (correct.sum() / len(y))
                dist = distance

                # store
                losses.update(loss.data, x.size()[0])
                accs.update(dist.data, x.size()[0])
                losses_action.update(loss_action.data, x.size()[0])

                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                print("Epoch: {} - {}/{} - {:.1f}s - loss: {:.3f} - dis: {:.3f} - loss_action : {:.3f} - loss_scale: {:.3f} "
                      "- mean_R : {:.3f} - mean-baseline: {:.3f} - mean_adjusted_reward: {:.3f} "
                      .format(epoch, count, countTotal, (toc-tic), loss.data, dist.data, loss_action.data, loss_scale.data,
                              mean_R, torch.mean(baselines.data), torch.mean(adjusted_reward.data)))

                # pbar.set_description(
                #     (
                #         "{:.1f}s - loss: {:.3f} - dis: {:.3f} - sum_R : {} - loss_action : {:.3f}".format(
                #             (toc-tic), loss.data, dist.data, sum_R, loss_action.data
                #         )
                #     )
                # )
                # pbar.update(self.batch_size)

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
                    log_value('train_acc', accs.avg, iteration)

            return losses.avg, accs.avg

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        countTotal = 50

        count = 0
        is_blend = 1
        save_dir = os.path.join('logs', '{:02d}'.format(epoch))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for i, (x, fixs, y, speeds, courses, scale_gt, indexSeq, frameEnd) in enumerate(self.valid_loader):
            y = y.squeeze().float()

            if count > countTotal:
                return losses.avg, accs.avg
            count = count + 1

            if self.use_gpu:
                x, y, speeds, courses = x.cuda(), y.cuda(), speeds.cuda(), courses.cuda()
            x, y, speeds, courses = Variable(x), Variable(y), Variable(speeds), Variable(courses)

            # duplicate 10 times
            x = x.repeat(self.M,1, 1, 1, 1)
            # speeds = speeds.repeat(self.M, 1,1,)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x,speeds, courses, l_t, h_t, t)

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, l_t_final, p, scale = self.model(
                x, speeds, courses, l_t, h_t, self.num_glimpses-1, last=True
            )
            log_pi.append(p)
            baselines.append(b_t)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            l_t_final = l_t_final.view(
                self.M, -1, l_t_final.shape[-1]
            )
            l_t_final = torch.mean(l_t_final, dim=0)

            if is_blend:
                for indexBlend in range(x.shape[0]):
                    # img = x[indexBlend, :, -1, : , :].cpu().numpy()
                    # img = np.transpose(img, (1,2,0))*255
                    # img = img[:, :, [2, 1, 0]]
                    pathImg = os.path.join(dreyeve_dir, '{:02d}'.format(indexSeq[indexBlend]), 'frames', '{:06d}.jpg'
                                           .format(frameEnd[indexBlend]))
                    img = read_image(pathImg, channels_first= False, color=True)
                    # cv2.imwrite( 'temp.jpg', img)
                    pathFix = os.path.join(dreyeve_dir, '{:02d}'.format(indexSeq[indexBlend]), 'saliency_fix', '{:06d}.png'
                                           .format(frameEnd[indexBlend]))
                    map = read_image(pathFix, channels_first= False, color=False)
                    # map = fixs[indexBlend, :,:].cpu().numpy()
                    loc = l_t_final[indexBlend, :].cpu().detach().numpy()
                    loc_gt = y[indexBlend].cpu().numpy()
                    scale_blend = scale[indexBlend].cpu().detach().numpy()
                    scale_gt_blend = scale_gt[indexBlend].cpu().detach().numpy()
                    # blend = blend_map_with_focus_circle
                    # loc= np.array([0,0])

                    # draw target
                    blend = blend_map_with_focus_rectangle(img, map, loc, scale=scale_blend, color= (0, 0, 255))
                    #draw gt
                    if not (np.isnan(loc_gt[0]) or np.isnan(loc_gt[1])):
                        # loc_gt[0]=-0.9
                        # loc_gt[1]=0.2
                        blend = blend_map_with_focus_rectangle(blend, map, loc_gt, scale = scale_gt_blend, color=(0, 255, 0))
                        # blend = blend_map_with_focus_circle(img, map, loc_gt, color=(0, 255, 0))

                    print('scale is {:.3f} and scale_gt is {:.3f}'.format(float(scale_blend), float(scale_gt_blend)))
                    cv2.imwrite(os.path.join(save_dir, '{:06d}.jpg'.format(frameEnd[indexBlend])), blend)

            baselines = baselines.contiguous().view(
                self.M, -1, baselines.shape[-1]
            )
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(
                self.M, -1, log_pi.shape[-1]
            )
            log_pi = torch.mean(log_pi, dim=0)

            # # calculate reward
            # predicted = torch.max(log_probas, 1)[1]
            # R = (predicted.detach() == y).float()
            # R = R.unsqueeze(1).repeat(1, self.num_glimpses)
            dis = 0
            R = torch.zeros(y.shape[0])
            for index in range(y.shape[0]):
                # get the distance of two locations
                distance = torch.sqrt(
                    torch.pow(l_t_final[index, 0] - y[index, 0], 2) + torch.pow(l_t_final[index, 1] - y[index, 1], 2))
                dis = dis + distance
                # R[index] = distance < self.dis_R_thres
                R[index] = distance < self.dis_R_thres
            # R = locs
            R = R.unsqueeze(1).repeat(1, self.num_glimpses).to(self.device)

            # compute losses for differentiable modules
            loss_action = F.mse_loss(l_t_final, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action*100 + loss_baseline + loss_reinforce
            # loss =  loss_baseline + loss_reinforce

            # compute accuracy
            # compute accuracy
            correct = dis.float()
            # acc = 100 * (correct.sum() / len(y))
            acc = dis / len(y)

            print('avg dist is {}'.format(acc))

            # store
            losses.update(loss.data, x.size()[0])
            accs.update(acc.data, x.size()[0])

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch*len(self.valid_loader) + i
                log_value('valid_loss', losses.avg, iteration)
                log_value('valid_acc', accs.avg, iteration)

        return losses.avg, accs.avg

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        is_output = 1
        if is_output:
            f = open('output.txt', 'w')

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        for i, (x, fixs, y, speeds, courses, scale_gt, indexSeq, frameEnd) in enumerate(self.test_loader):
            y = y.squeeze().float()

            if self.use_gpu:
                x, y, speeds, courses = x.cuda(), y.cuda(), speeds.cuda(), courses.cuda()
            x, y, speeds, courses = Variable(x), Variable(y), Variable(speeds), Variable(courses)

            # duplicate 10 times
            x = x.repeat(self.M, 1, 1, 1, 1)
            # speeds = speeds.repeat(self.M, 1,1,)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, speeds, courses, l_t, h_t, t)

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, l_t_final, p, scale = self.model(
                x, speeds, courses, l_t, h_t, self.num_glimpses - 1, last=True
            )
            log_pi.append(p)
            baselines.append(b_t)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            l_t_final = l_t_final.view(
                self.M, -1, l_t_final.shape[-1]
            )
            l_t_final = torch.mean(l_t_final, dim=0)

            if is_output:
                for indexBlend in range(x.shape[0]):
                    loc = l_t_final[indexBlend, :].cpu().detach().numpy()
                    loc_gt = y[indexBlend].cpu().numpy()
                    scale_blend = scale[indexBlend].cpu().detach().numpy()
                    scale_gt_blend = scale_gt[indexBlend].cpu().detach().numpy()

                    line = '{:02} {:04d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'\
                        .format(indexSeq[indexBlend], frameEnd[indexBlend], loc_gt[0], loc_gt[1], loc[0], loc[1], float(scale_gt_blend), float(scale_blend))
                    print('seq: {:02}- frame: {:04d} - loc_gt_h: {:.3f} - loc_gt_w: {:.3f} - loc_h: {:.3f} - loc_w: {:.3f} '
                          '- scale_gt: {:.3f}- scale: {:.3f}'\
                        .format(indexSeq[indexBlend], frameEnd[indexBlend], loc_gt[0], loc_gt[1], loc[0], loc[1], float(scale_gt_blend), float(scale_blend)))
                    f.writelines(line)

            baselines = baselines.contiguous().view(
                self.M, -1, baselines.shape[-1]
            )
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(
                self.M, -1, log_pi.shape[-1]
            )
            log_pi = torch.mean(log_pi, dim=0)

            dis = 0
            R = torch.zeros(y.shape[0])
            for index in range(y.shape[0]):
                # get the distance of two locations
                distance = torch.sqrt(
                    torch.pow(l_t_final[index, 0] - y[index, 0], 2) + torch.pow(l_t_final[index, 1] - y[index, 1], 2))
                dis = dis + distance
                # R[index] = distance < self.dis_R_thres
                R[index] = distance < self.dis_R_thres
            # R = locs
            R = R.unsqueeze(1).repeat(1, self.num_glimpses).to(self.device)

            # compute losses for differentiable modules
            loss_action = F.mse_loss(l_t_final, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action * 100 + loss_baseline + loss_reinforce
            # loss =  loss_baseline + loss_reinforce

            # compute accuracy
            acc = dis / len(y)

            print('avg dist is {}'.format(acc))

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



