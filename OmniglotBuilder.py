# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: BoyuanJiang
# College of Information Science & Electronic Engineering,ZheJiang University
# Email: ginger188@gmail.com
# Copyright (c) 2017

# @Time    :17-8-29 16:20
# @FILE    :OmniglotBuilder.py
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from matching_networks import MatchingNetwork
# from MatchingNetwork import MatchingNetwork
import torch
import tqdm
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class OmniglotBuilder:
    def __init__(self, data):
        """
        Initializes the experiment
        :param data:
        """
        self.data = data

    def build_experiment(self, batch_size, num_channels, lr, image_size, classes_per_set, samples_per_class, keep_prob,
                         fce, optim, weight_decay, use_cuda):
        """

        :param batch_size:
        :param num_channels:
        :param lr:
        :param image_size:
        :param classes_per_set:
        :param samples_per_class:
        :param keep_prob:
        :param fce:
        :param optim:
        :param weight_decay:
        :param use_cuda:
        :return:
        """
        self.classes_per_set = classes_per_set
        self.sample_per_class = samples_per_class
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.optim = optim
        self.wd = weight_decay
        self.isCuadAvailable = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self.matchNet = MatchingNetwork(keep_prob, batch_size, num_channels, self.lr, fce, classes_per_set,
                                        samples_per_class, image_size, self.isCuadAvailable & self.use_cuda)
        self.total_iter = 0
        if self.isCuadAvailable & self.use_cuda:
            cudnn.benchmark = True  # set True to speedup
            torch.cuda.manual_seed_all(2017)
            self.matchNet.cuda()
        self.total_train_iter = 0
        self.optimizer = self._create_optimizer(self.matchNet, self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',verbose=True)

    def run_training_epoch(self, total_train_batches):
        """
        Run the training epoch
        :param total_train_batches: Number of batches to train on
        :return:
        """
        total_c_loss = 0.0
        total_accuracy = 0.0
        # optimizer = self._create_optimizer(self.matchNet, self.lr)

        with tqdm.tqdm(total=total_train_batches) as pbar:
            for i in range(total_train_batches):
                x_support_set, y_support_set, x_target, y_target = self.data.get_train_batch(True)
                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).float()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

                # convert to one hot encoding
                y_support_set = y_support_set.unsqueeze(2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = Variable(
                    torch.zeros(batch_size, sequence_length, self.classes_per_set).scatter_(2,
                                                                                            y_support_set.data,
                                                                                            1), requires_grad=False)

                # reshape channels and change order
                size = x_support_set.size()
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                if self.isCuadAvailable & self.use_cuda:
                    acc, c_loss = self.matchNet(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                                y_target.cuda())
                else:
                    acc, c_loss = self.matchNet(x_support_set, y_support_set_one_hot, x_target, y_target)

                # optimize process
                self.optimizer.zero_grad()
                c_loss.backward()
                self.optimizer.step()

                # TODO: update learning rate?

                iter_out = "tr_loss: {}, tr_accuracy: {}".format(c_loss.data[0], acc.data[0])
                pbar.set_description(iter_out)
                pbar.update(1)
                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                # self.total_train_iter+=1

            total_c_loss = total_c_loss / total_train_batches
            total_accuracy = total_accuracy / total_train_batches
            return total_c_loss, total_accuracy

    def _create_optimizer(self, model, lr):
        # setup optimizer
        if self.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.wd)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=self.wd)
        else:
            raise Exception("Not a valid optimizer offered: {0}".format(self.optim))
        return optimizer

    def _adjust_learning_rate(self, optimizer):
        """
        Update the learning rate after some epochs
        :param optimizer:
        :return:
        """

    def run_val_epoch(self, total_val_batches):
        """
        Run the training epoch
        :param total_train_batches: Number of batches to train on
        :return:
        """
        total_c_loss = 0.0
        total_accuracy = 0.0

        with tqdm.tqdm(total=total_val_batches) as pbar:
            for i in range(total_val_batches):
                x_support_set, y_support_set, x_target, y_target = self.data.get_val_batch(False)
                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).float()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

                # convert to one hot encoding
                y_support_set = y_support_set.unsqueeze(2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = Variable(
                    torch.zeros(batch_size, sequence_length, self.classes_per_set).scatter_(2,
                                                                                            y_support_set.data,
                                                                                            1), requires_grad=False)

                # reshape channels and change order
                size = x_support_set.size()
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                if self.isCuadAvailable & self.use_cuda:
                    acc, c_loss = self.matchNet(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                                y_target.cuda())
                else:
                    acc, c_loss = self.matchNet(x_support_set, y_support_set_one_hot, x_target, y_target)

                # TODO: update learning rate?

                iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss.data[0], acc.data[0])
                pbar.set_description(iter_out)
                pbar.update(1)
                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                # self.total_train_iter+=1

            total_c_loss = total_c_loss / total_val_batches
            total_accuracy = total_accuracy / total_val_batches
            self.scheduler.step(total_c_loss)
            return total_c_loss, total_accuracy

    def run_test_epoch(self, total_test_batches):
        """
        Run the training epoch
        :param total_train_batches: Number of batches to train on
        :return:
        """
        total_c_loss = 0.0
        total_accuracy = 0.0

        with tqdm.tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):
                x_support_set, y_support_set, x_target, y_target = self.data.get_test_batch(False)
                x_support_set = Variable(torch.from_numpy(x_support_set)).float()
                y_support_set = Variable(torch.from_numpy(y_support_set), requires_grad=False).long()
                x_target = Variable(torch.from_numpy(x_target)).float()
                y_target = Variable(torch.from_numpy(y_target), requires_grad=False).squeeze().long()

                # convert to one hot encoding
                y_support_set = y_support_set.unsqueeze(2)
                sequence_length = y_support_set.size()[1]
                batch_size = y_support_set.size()[0]
                y_support_set_one_hot = Variable(
                    torch.zeros(batch_size, sequence_length, self.classes_per_set).scatter_(2,
                                                                                            y_support_set.data,
                                                                                            1), requires_grad=False)

                # reshape channels and change order
                size = x_support_set.size()
                x_support_set = x_support_set.permute(0, 1, 4, 2, 3)
                x_target = x_target.permute(0, 3, 1, 2)
                if self.isCuadAvailable & self.use_cuda:
                    acc, c_loss = self.matchNet(x_support_set.cuda(), y_support_set_one_hot.cuda(), x_target.cuda(),
                                                y_target.cuda())
                else:
                    acc, c_loss = self.matchNet(x_support_set, y_support_set_one_hot, x_target, y_target)

                # TODO: update learning rate?

                iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss.data[0], acc.data[0])
                pbar.set_description(iter_out)
                pbar.update(1)
                total_c_loss += c_loss.data[0]
                total_accuracy += acc.data[0]
                # self.total_train_iter+=1

            total_c_loss = total_c_loss / total_test_batches
            total_accuracy = total_accuracy / total_test_batches
            return total_c_loss, total_accuracy