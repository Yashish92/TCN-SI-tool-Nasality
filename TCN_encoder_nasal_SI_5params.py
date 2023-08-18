#!/usr/bin/env python
# coding: utf-8

# Authors: Yashish Maduwantha

from __future__ import print_function
import argparse

import matplotlib

SERVER = True  # This variable enable or disable matplotlib, set it to true when you use the server!
if SERVER:
    matplotlib.use('Agg')

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
#from utils import correlation_coefficient_loss as myloss

import os
import numpy as np
import h5py
import time
import subprocess
import logging

# from IPython.display import Audio
# import pyworld as pw
# import librosa

from datetime import date
import matplotlib.pyplot as plt
import pickle as pkl
import datetime
from utils_PPMC_nasal_SI import compute_corr_score
import nsltools as nsl


# get_ipython().run_line_magic('matplotlib', 'inline')

# get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES = 0')

# In[61]:
tv_dim = 5 #32 # 64
# ap_enc_dim = 1

fs = 16000
# fftlen = pw.get_cheaptrick_fft_size(fs)

class BoDataset(Dataset):
    """
        Wrapper for the WSJ Dataset.
    """

    def __init__(self, path):
        super(BoDataset, self).__init__()

        self.h5pyLoader = h5py.File(path, 'r')
        self.wav = self.h5pyLoader['speaker']
        self.spec = self.h5pyLoader['spec513']
        # self.specWorld = self.h5pyLoader['spec513_pw']
        self.hidden = self.h5pyLoader['hidden']
        self._len = self.wav.shape[0]
        # print (path)
        print('h5py loader', type(self.h5pyLoader))
        print('wav', self.wav)
        print('spec', self.spec)
        # print('self.specWorld', self.specWorld)
        print('self.hidden', self.hidden)

    def __getitem__(self, index):
        # print (index)
        wav_tensor = torch.from_numpy(self.wav[index])  # raw waveform with normalized power
        # print (len(self.hidden[index]))
        # spec_tensor = torch.from_numpy(self.spec[index])         # magnitude of spectrogram of raw waveform
        if len(self.hidden[index]) > 0:
            hidden_tensor = torch.from_numpy(self.hidden[index])  # parameters of world
        else:
            hidden_tensor = []
        spec_tensor = torch.from_numpy(self.spec[index])

        return wav_tensor, [], hidden_tensor, spec_tensor

    def __len__(self):
        return self._len


def log_print(content):
    print(content)
    logging.info(content)


# In[70]:
EPS = 1e-8

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        return gLN_y


class CNN1D(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel,
                 dilation=1, stride=1, padding=0, ds=2):
        super(CNN1D, self).__init__()

        self.causal = False

        if self.causal:
            self.padding1 = (kernel - 1) * dilation
            self.padding2 = (kernel - 1) * dilation * ds
        else:
            self.padding1 = (kernel - 1) // 2 * dilation
            self.padding2 = (kernel - 1) // 2 * dilation * ds

        # self.conv1x1 = nn.Conv1d(input_channel, hidden_channel, 1, bias=False)

        self.conv1d1 = nn.Conv1d(input_channel, hidden_channel, kernel,
                                 stride=stride, padding=self.padding1,
                                 dilation=dilation)
        self.conv1d2 = nn.Conv1d(input_channel, hidden_channel, kernel,
                                 stride=stride, padding=self.padding2,
                                 dilation=dilation * ds)

        self.reg1 = nn.BatchNorm1d(hidden_channel, eps=1e-16)
        self.reg2 = nn.BatchNorm1d(hidden_channel, eps=1e-16)
        # self.reg1 = GlobalLayerNorm(hidden_channel)
        # self.reg2 = GlobalLayerNorm(hidden_channel)
        self.nonlinearity = nn.ReLU()
        # self.nonlinearity = nn.PReLU()

    def forward(self, input):
        # res=input

        # input_1=self.nonlinearity(self.reg1(self.conv1x1(input)))
        output = self.nonlinearity(self.reg1(self.conv1d1(input)))
        if self.causal:
            output = output[:, :, :-self.padding1]
        output = self.reg2(self.conv1d2(output))
        if self.causal:
            output = output[:, :, :-self.padding2]
        output = self.nonlinearity(output + input)
        # output=output+res

        return output


# In[71]:

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.AE_stride = 80
        self.AE_channel = 256
        self.BN_channel = 256 # changed from 256

        self.AE_win = 320
        self.AE_stride = 128
        # self.AE_channel = 128
        # self.BN_channel = 128

        self.kernel = 3
        self.CNN_layer = 4
        self.stack = 3

        self.L1 = nn.Sequential(
            #nn.Conv1d(128, 128, 1),
            nn.Conv1d(128, 256, 1), # changed from 256
            nn.GroupNorm(1, 256, eps=1e-16),
            nn.Conv1d(256, 256, 1)  # added new
            # nn.BatchNorm1d(128, 128, 1),  # added new
            # nn.ReLU()
        )

        #self.encoder = nn.Conv1d(128, self.AE_channel, 1)

        # self.encoder = nn.Conv1d(1, self.AE_channel, self.AE_win,
        #                          stride=self.AE_stride, bias=False)
        #self.enc_Norm = nn.GroupNorm(1, self.AE_channel, eps=1e-16)

        #self.L1 = nn.Conv1d(self.AE_channel, self.AE_channel, 1)

        self.CNN = nn.ModuleList([])
        for s in range(self.stack):
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 0))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 2))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 4))

        #self.f0_linear = nn.Linear(16064,16064)

        self.sp_seq = nn.Sequential(
            nn.Conv1d(self.BN_channel, 256, 1), # changed from 256
            nn.BatchNorm1d(256, 256, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),
            # nn.AvgPool1d(5, stride=5),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128, 128, 1),
            nn.ReLU(),
            nn.AvgPool1d(5, stride=5),
            nn.Conv1d(128, tv_dim, 1),
            #nn.Conv1d(256, 513, 1),
            nn.BatchNorm1d(tv_dim, tv_dim, 1),
            #nn.AvgPool1d(5, stride=5),
            nn.Tanh()
            #nn.Linear(sp_enc_dim, sp_enc_dim)
            # nn.Tanh()
            # nn.ReLU(),

        )
        self.sp_seq.apply(initialize_weights)

        #self.ap_linear = nn.Linear()

    def forward(self, input):
        # input shape: B, T
        batch_size = input.size(0)
        nsample = input.size(1)
        # print (input.size(), ' encoder input.size() encoder') #64-32320

        # pad_num = 160
        # pad_aux = Variable(torch.zeros(batch_size, pad_num)).type(input.type())
        # input = torch.cat([pad_aux, input, pad_aux], 1)
        # output = input.unsqueeze(1)  # B, 1, T=32320
        # print (output.size(), 'encoder output')

        # encoder
        this_input = self.L1(input)  # B, C, T #T=[{(32320-320)/80}+1]=401
        #this_input = input

        # print (this_input.size(), 'this_input after L1, before stack. Encoder') #64-256-401
        for i in range(len(self.CNN)):
            this_input = self.CNN[i](this_input)   # changed from this_input

        sp = self.sp_seq(this_input)

        return sp


# In[72]:


# class decoder(nn.Module):
#     def __init__(self):
#         super(decoder, self).__init__()
#
#         self.AE_win = 320
#         self.AE_stride = 80
#         self.AE_channel = 256
#         self.BN_channel = 256
#         # self.BN_channel = 128
#         self.kernel = 3
#         self.CNN_layer = 4
#         self.stack = 3
#
#         # self.encoder = nn.Conv1d(1, self.AE_channel, self.AE_win,
#         #                          stride=self.AE_stride, bias=False)
#
#         # self.L1 = nn.Sequential(
#         #                         nn.Conv1d(1+513*2, 256, 1)
#
#         # )
#
#         self.L1 = nn.Sequential(
#             # nn.Upsample(scale_factor=2),
#             # nn.Conv1d(4, 128, 1),  # changed 1 + 513 * 2 to 16
#             # nn.BatchNorm1d(128, 128, 1),
#             # nn.ReLU(),
#             # nn.Upsample(scale_factor=5),
#             # nn.Conv1d(128, 256, 1),  # changed 1 + 513 * 2 to 16
#             # nn.BatchNorm1d(256, 256, 1),
#             # nn.ReLU(),
#             # nn.Upsample(scale_factor=5),
#             # nn.Conv1d(256, 256, 1),  # changed 1 + 513 * 2 to 16
#             # nn.BatchNorm1d(256, 256, 1),
#             # nn.ReLU()
#
#             #nn.AvgPool1d(4, stride=4),
#             #nn.Upsample(scale_factor=5),
#             nn.Conv1d(tv_dim, 128, 1, bias=False),  # changed 1 + 513 * 2 to 16
#             nn.BatchNorm1d(128, 128, 1),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=5),
#             nn.Conv1d(128, 256, 1, bias=False),  # changed 1 + 513 * 2 to 16
#             nn.BatchNorm1d(256, 256, 1),
#             nn.ReLU(),
#             #nn.Upsample(scale_factor=5),
#             nn.AvgPool1d(4, stride=4),
#             nn.Conv1d(256, 256, 1, bias=False),  # changed 1 + 513 * 2 to 16
#             nn.BatchNorm1d(256, 256, 1),
#             nn.ReLU()
#
#             # nn.Upsample(scale_factor=12)
#             # nn.ConvTranspose1d(4, 128, kernel_size=51),  # changed 1 + 513 * 2 to 16
#             # nn.BatchNorm1d(128, 128, 1),
#             # nn.ReLU(),
#             # nn.ConvTranspose1d(128, 256, kernel_size=99),  # changed 1 + 513 * 2 to 16
#             # nn.BatchNorm1d(256, 256, 1),
#             # nn.ReLU(),
#             # nn.ConvTranspose1d(256, 256, kernel_size=100),  # changed 1 + 513 * 2 to 16
#             # nn.BatchNorm1d(256, 256, 1),
#             # nn.ReLU()
#
#         )
#
#         self.CNN = nn.ModuleList([])
#         for s in range(self.stack):
#             self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
#                                   dilation=2 ** 0))
#             self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
#                                   dilation=2 ** 2))
#             self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
#                                   dilation=2 ** 4))
#
#         # self.L2 = nn.Sequential(
#         #                         nn.Conv1d(self.AE_channel, 256, 1)
#
#         # )
#
#         self.L2 = nn.Sequential(
#             nn.Conv1d(256, 256, 1),
#             nn.GroupNorm(1, self.AE_channel, eps=1e-16),
#             nn.Conv1d(256, 128, 1),
#             nn.Conv1d(128, 128, 1)  # added new
#             # nn.BatchNorm1d(128, 128, 1),  # added new
#             # nn.ReLU()    		# added new
#
#         )
#
#     def forward(self, input):
#         # input shape: B, T
#         batch_size = input.size(0)
#         nsample = input.size(1)
#
#         #
#         # print (input.size(), 'decoder input.size()')    #64-1027-251 shape of latent space
#         # print(input.size)
#         # input = input.unsqueeze(dim=3)
#         this_input = self.L1(input)  # 64-128-250
#         # print(this_input.shape)
#         # print (this_input.size(),'this_input before CNN stack decoder--------' )
#         for i in range(len(self.CNN)):
#             this_input = self.CNN[i](this_input)  # 64-128-250
#             # print (this_input.size(), 'Size(this_input) inside CNN stack decoder')
#         # print(this_input.shape)
#         this_input = self.L2(this_input)
#         # print (this_input.size()) #output is 64-128-250
#
#         return this_input


criteron = nn.MSELoss()


def new_training_technique(epoch, train_E=False, init=False):
    '''
    This function train the networks for one epoch using the new training technique.
    train_D and train_E can be specified to train only one network or both in the same time.
    More details about the loss functions and the architecture in the README.md
    '''
    start_time = time.time()

    # D.eval()
    E.eval()

    # if train_D:
    #     D.train()
    if train_E:
        E.train()

    # train_loss = 0.
    # train_loss2 = 0.
    train_loss = 0.
    new_loss = 0.
    # E2_train = 0.  # Decoder Error ||W(H_hat), D(H_Hat)||
    E1_train = 0.  # Encoder Error || D(E(X)), X ||

    pad = 40
    # mel_basis = librosa.filters.mel(16000, 1024, n_mels=256)
    # if args.cuda:
    #     mel_basis = torch.from_numpy(mel_basis).cuda().float().unsqueeze(0)
    # else:
    #     mel_basis = torch.from_numpy(mel_basis).float().unsqueeze(0)
    if init:
        for (batch_idx, data_random) in enumerate(train_loader):

            batch_wav_random = Variable(data_random[0]).contiguous().float()
            batch_h_random = Variable(data_random[2]).contiguous().float()
            batch_spec_random = Variable(data_random[3]).contiguous().float()

            if args.cuda:
                batch_wav_random = batch_wav_random.cuda()
                batch_spec_random = batch_spec_random.cuda()
                batch_h_random = batch_h_random.cuda()

            fs = 16000

            # predict parameters through waveform
            # note: we can predict h through batch_spec, too. But I found waveform is better. batch_spec is derived from batch_wav.

            if train_E:
                E_optimizer.zero_grad()

                if init:
                    enc_out = E(batch_spec_random)
                    loss1 = criteron(enc_out, batch_h_random)

                loss1.backward()
                E_optimizer.step()
                train_loss += loss1.data.item()

            # if train_E and train_D:
            #     loss = loss1 + loss2
            #     train_loss += loss.data.item()

            # loss.backward()
            # all_optimizer.step()

            if (batch_idx + 1) % args.log_step == 0:
                elapsed = time.time() - start_time
                log_print(
                    '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | train loss {:5.8f}|'.format(
                        epoch, batch_idx + 1, len(train_loader),
                               elapsed * 1000 / (batch_idx + 1), train_loss / (batch_idx + 1)))

        val_loss = 0.0
        E.eval()

        for (batch_idx_val, data_random_val) in enumerate(dev_loader):
            # Transfer Data to GPU if available
            batch_wav_random = Variable(data_random_val[0]).contiguous().float()
            batch_h_random = Variable(data_random_val[2]).contiguous().float()
            batch_spec_random = Variable(data_random_val[3]).contiguous().float()

            if args.cuda:
                batch_wav_random = batch_wav_random.cuda()
                batch_spec_random = batch_spec_random.cuda()
                batch_h_random = batch_h_random.cuda()

            loss = criteron(E(batch_spec_random), batch_h_random)
            # Calculate Loss
            val_loss += loss.item()


    train_loss /= (batch_idx + 1)
    # train_loss2 /= (batch_idx + 1)  # Decoder error per epoch.
    # train_loss /= (batch_idx + 1)  # total loss
    new_loss /= (batch_idx + 1)
    # E2_train /= (batch_idx+1)
    E1_train /= (batch_idx + 1)
    val_loss /= (batch_idx_val + 1)
    # train_loss1 /= len(train_loader)
    # train_loss2 /= len(train_loader)
    # train_loss /= len(train_loader)
    # print (len(train_loader), 'len(train_loader)')

    log_print('-' * 99)
    log_print(
        '    | end of training epoch {:3d} | time: {:5.2f}s | train loss {:5.8f} | val loss: {:5.8f} |'.format(
            epoch, (time.time() - start_time), train_loss, val_loss))

    return train_loss, new_loss, val_loss


def trainTogether_newTechnique(epochs=None, save_better_model=False, loader_eval="evaluation", train_E=False, init=False, name=""):
    '''
    Glob function for the new training technique, it iterates over all the epochs,
    process the learning rate decay, save the weights and call the evaluation funciton.
    '''
    reduce_epoch = []

    training_loss_encoder = []
    training_loss_decoder = []

    validation_loss_encoder = []
    validation_loss_decoder = []

    validation_temp_encoder = []
    validation_temp_decoder = []

    # E2_loss_decoder=[]
    E1_loss_encoder = []

    new_loss = []

    decay_cnt1 = 0
    decay_cnt2 = 0

    if epochs is None:
        epochs = args.epochs

    # print()

    # if train_E and not train_D:
    #     print("TRAINING E ONLY")
    # if not train_E and train_D:
    #     print("TRAINING D ONLY")
    # if train_E and train_D:
    #     print("TRAINING E AND D")
    # if not train_E and not train_D:
    #     print("TRAINING NOTHING ...")

    # Training all
    for epoch in range(1, epochs + 1):
        print(epoch, 'epoch')
        if args.cuda:
            E.cuda()
            # D.cuda()

        error = new_training_technique(epoch, train_E=train_E, init=init)

        training_loss_encoder.append(error[0])
        # training_loss_decoder.append(error[2])
        new_loss.append(error[1])  # E2 error for decoder during training
        print("Difference between D en W:", error[1])

        # if train_D and not init:
        #     E2_loss_decoder.append(error[5])

        if train_E and not init:
            E1_loss_encoder.append(error[2])

        decay_cnt2 += 1

        if np.min(training_loss_encoder) not in training_loss_encoder[-3:] and decay_cnt1 >= 3:
            E_scheduler.step()
            decay_cnt1 = 0
            log_print('      Learning rate decreased for E.')
            log_print('-' * 99)

        # if np.min(training_loss_decoder) not in training_loss_decoder[-3:] and decay_cnt2 >= 3:
        #     D_scheduler.step()
        #     decay_cnt2 = 0
        #     log_print('      Learning rate decreased for D.')
        #     log_print('-' * 99)

#        if train_D and epoch % 5==0:
#            print("EVAL DECOER")
#            total_loss_eval, encoder_loss_eval, decoder_loss_eval = quickEval(epoch, loader="dev",
#                                                                              train_D=train_D, train_E=train_E,
#                                                                              ideal_h=True)
#            print(
#                '--------------------------------------------------------------------------------------------------------')
#            print('Decoder:---', decoder_loss_eval)
#            validation_temp_decoder.append(decoder_loss_eval)
#            # print (validation_temp_decoder, 'validation_temp_decoder')
#
#        if train_E and epoch % 5==0:
#            print("DEV ENCODER")
#            total_loss_eval, encoder_loss_eval, decoder_loss_eval = quickEval(epoch, loader="dev",
#                                                                              train_D=train_D, train_E=train_E,
#                                                                              ideal_h=False)
#            print(
#                '--------------------------------------------------------------------------------------------------------')
#            print('Encoder:---', encoder_loss_eval)
#            # print("___________________")
#            validation_temp_encoder.append(encoder_loss_eval)
#            # print (validation_temp_encoder, 'validation_temp_encoder')
#
#
def generate_figures(mode="evaluation", name="", load_weights=("", "")):
    '''
    Generates a set of figures that help to evaluate the training, it allows to see the generated ap, f0 and sp for example.
    '''

    if mode == "evaluation":
        loader = validation_loader
    elif mode == "train":
        loader = train_loader
    elif mode == "train_random":
        #loader = train_random
        loader = dev_loader
    else:
        print("We did not understand the mode, we skip this function.")
        return

    if name is not "":
        local_out_init = out + "/" + name + "_" + mode
    else:
        local_out_init = out + "/" + mode

    if not os.path.exists(local_out_init):
        os.makedirs(local_out_init)

    if load_weights != ("", ""):
        E.load_state_dict(torch.load(load_weights[0]))
        # D.load_state_dict(torch.load(load_weights[1]))

    E.eval()
    # D.eval()

    if args.cuda:
        E.cuda()
        # D.cuda()

    pad = 40
    # mel_basis = librosa.filters.mel(16000, 1024, n_mels=256)
    # if args.cuda:
    #     mel_basis = torch.from_numpy(mel_basis).cuda().float().unsqueeze(0)
    # else:
    #     mel_basis = torch.from_numpy(mel_basis).float().unsqueeze(0)

    spectrograms = []

    for batch_idx, data in enumerate(loader):

        batch_wav = Variable(data[0]).contiguous().float()
        batch_h = Variable(data[2]).contiguous().float()
        batch_spec = Variable(data[3]).contiguous().float()

        if args.cuda:
            batch_wav = batch_wav.cuda()
            batch_spec = batch_spec.cuda()
            batch_h = batch_h.cuda()

        # ## mel-spectrogram
        # #Not needed for AudSpec
        # batch_spec = batch_spec ** 2
        # mel_basis_ep = mel_basis.expand(batch_spec.size(0),mel_basis.size(1),mel_basis.size(2))
        # batch_spec = torch.bmm(mel_basis_ep, batch_spec)
        # batch_spec = torch.log(batch_spec)
        # batch_spec[batch_spec == float("-Inf")] = -50

        # predict parameters through waveform
        fs = 16000

        batch_sp_code = E(batch_spec)
        # batch_ap = pw.decode_aperiodicity(batch_ap_code, fs, fftlen)
        # batch_sp = pw.decode_spectral_envelope(batch_sp_code, fs, fftlen)
        ## do decoding here to match dimensions

        # batch_h_hat = torch.cat([batch_f0, batch_sp, batch_ap], 1)

        ############# World ###################################################################################################
        # mel_b = librosa.filters.mel(16000, 1024, n_mels=256)
        # f0_temp = batch_h_hat[:, 0:1, :].data.cpu().numpy().astype('float64')
        # sp_temp = batch_h_hat[:, 1:514, :].data.cpu().numpy().astype('float64')
        # ap_temp = batch_h_hat[:, 514:, :].data.cpu().numpy().astype('float64')
        # f0_temp = batch_f0.data.cpu().numpy().astype('float64')
        sp_temp = batch_sp_code[:, 0:tv_dim, :].data.cpu().numpy().astype('float64')
        # ap_temp = batch_ap_code.data.cpu().numpy().astype('float64')

        # for latent space visualization
        # ap_temp_2 = np.zeros((f0_temp.shape[0], ap_enc_dim, 251), dtype=np.float64)
        # sp_temp_2 = np.zeros((sp_temp.shape[0], tv_dim, 200), dtype=np.float64)

        # f0_temp = batch_h[:,0:1,:].data.cpu().numpy().astype('float64')
        # sp_temp = batch_h[:,1:129,:].data.cpu().numpy().astype('float64')
        # ap_temp = batch_h[:,129:,:].data.cpu().numpy().astype('float64')
        # if not os.path.exists(local_out_init+"/waveforms"):
        # os.makedirs(local_out_init+"/waveforms")

        spec_wav = np.zeros((batch_wav.shape[0], 128, 250)).astype('float64')
        # spec_wav = np.zeros((batch_wav.shape[0], 128, 250)).astype('float64')
        frmlen = 8
        tc = 8
        sampling_rate = 16000
        paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000)]

        # f0 = batch_h[:, 0:1, :].data.cpu().numpy().astype('float64')
        sp = batch_h[:, 0:tv_dim, :].data.cpu().numpy().astype('float64')
        # ap = batch_h[:, (tv_dim + 1):, :].data.cpu().numpy().astype('float64')
        # f0 = batch_h[:,0:1,:].data.cpu().numpy().astype('float64')
        # sp = batch_h[:,1:129,:].data.cpu().numpy().astype('float64')
        # ap = batch_h[:,129:,:].data.cpu().numpy().astype('float64')
        ## SAVE ALL FIGURES

        # f0_random = batch_f0 + (torch.rand(f0_temp.shape[0],f0_temp.shape[1], f0_temp.shape[2]) - 0.5)*10
        # sp_random = batch_sp + (torch.rand(sp_temp.shape[0], sp_temp.shape[1], sp_temp.shape[2]) - 0.5)*5
        # ap_random = batch_ap + torch.rand(ap_temp.shape[0], ap_temp.shape[1], ap_temp.shape[2]) - 0.5

        # batch_h_random = torch.cat([f0_random, sp_random, ap_random], 1)
        wave_files = batch_wav[:, :].data.cpu().numpy().astype('float64')

        local_out_init += "/lentent_space/"

        if mode == "evaluation":
            corr_avg, avg_corr = compute_corr_score(sp_temp, sp, tv_dim)

            # write a .txt with results and model params
            lines = ['Corr_Average:' + str(corr_avg), 'Corr_Average_all:' + str(avg_corr)]
            with open(out_dir + 'log.txt', 'w') as f:
                f.write('\n'.join(lines))

        # local_out_init += "/lentent_space/"
        # print (f0.shape[0])
        num = np.min((sp_temp.shape[0], 10))
        # print (num, 'num')
        # Save ap
        # Save Sp
        # ap_hat_saved = np.zeros((num, ap_enc_dim, 251), dtype=np.float64)
        # ap_ideal_saved = np.zeros((num, ap_enc_dim, 251), dtype=np.float64)
        wave_files_saved = np.zeros((num, wave_files.shape[1]), dtype=np.float64)
        sp_hat_saved = np.zeros((num, tv_dim, 200), dtype=np.float64)
        sp_ideal_saved = np.zeros((num, tv_dim, 200), dtype=np.float64)
        # f0_hat_saved = np.zeros((num, 1, 251), dtype=np.float64)
        # f0_ideal_saved = np.zeros((num, 1, 251), dtype=np.float64)

        # print (num)

        for i in range(num):
            # ap_hat_saved[i, :, :] = ap_temp_2[i, :, :]
            # ap_ideal_saved[i, :, :] = ap[i, :, :]
            wave_files_saved[i, :] = wave_files[i, :]
            sp_hat_saved[i, :, :] = sp_temp[i, :, :]
            sp_ideal_saved[i, :, :] = sp[i, :, :]
            # f0_ideal_saved[i, :, :] = f0[i, :, :]
            # f0_hat_saved[i, :, :] = f0_temp[i, :, :]

        if not os.path.exists(local_out_init):
            os.makedirs(local_out_init)

        # with open(local_out_init + 'ap_hat.pkl', 'wb') as f:
        #     pkl.dump(ap_hat_saved, f)
        #
        # with open(local_out_init + 'ap_ideal.pkl', 'wb') as f:
        #     pkl.dump(ap_ideal_saved, f)

        with open(local_out_init + 'wave_files.pkl', 'wb') as f:
            pkl.dump(wave_files_saved, f)

        with open(local_out_init + 'sp_hat.pkl', 'wb') as f:
            pkl.dump(sp_hat_saved, f)

        with open(local_out_init + 'sp_ideal.pkl', 'wb') as f:
            pkl.dump(sp_ideal_saved, f)

        # with open(local_out_init + 'f0_ideal.pkl', 'wb') as f:
        #     pkl.dump(f0_ideal_saved, f)
        #
        # with open(local_out_init + 'f0_hat.pkl', 'wb') as f:
        #     pkl.dump(f0_hat_saved, f)

        if not os.path.exists(local_out_init):
            os.makedirs(local_out_init)

        for i in range(min(sp.shape[0], 10)):

            local_out = local_out_init + str(i) + "/"

            if not os.path.exists(local_out):
                os.makedirs(local_out)

            plt.title("Naslance")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(sp_temp[i, 0, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(sp[i, 0, :])
            plt.savefig(local_out + "nasalance.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("glottal TV")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(sp_temp[i, 1, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(sp[i, 1, :])
            plt.savefig(local_out + "glottal.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("Ap")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(sp_temp[i, 2, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(sp[i, 2, :])
            plt.savefig(local_out + "Ap.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("Per")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(sp_temp[i, 3, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(sp[i, 3, :])
            plt.savefig(local_out + "Per.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("Pitch")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(sp_temp[i, 4, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(sp[i, 4, :])
            plt.savefig(local_out + "Pitch.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

           # plt.title("TTCD")
           # plt.subplot(121)
           # plt.title("from encoder (E(x))")
           # plt.plot(sp_temp[i, 5, :])
           # plt.subplot(122)
           # plt.title("ground truth")
           # plt.plot(sp[i, 5, :])
           # plt.savefig(local_out + "TTCD.eps")

           # if not SERVER:
           #     plt.show()
           # else:
           #     plt.close()
           # #
           # # plt.title("Aperiodicity")
            # plt.subplot(121)
            # plt.title("from encoder (E(x))")
            # plt.plot(music_tmp[i, 6, :])
            # plt.subplot(122)
            # plt.title("ground truth")
            # plt.plot(music[i, 6, :])
            # plt.savefig(local_out + "Aperio.eps")
            #
            # if not SERVER:
            #     plt.show()
            # else:
            #     plt.close()
            #
            # plt.title("Periodicity")
            # plt.subplot(121)
            # plt.title("from encoder (E(x))")
            # plt.plot(music_tmp[i, 7, :])
            # plt.subplot(122)
            # plt.title("ground truth")
            # plt.plot(music[i, 7, :])
            # plt.savefig(local_out + "Perio.eps")
            #
            # if not SERVER:
            #     plt.show()
            # else:
            #     plt.close()
            #
            # plt.title("Pitch (scaled)")
            # plt.subplot(121)
            # plt.title("from encoder (E(x))")
            # plt.plot(music_tmp[i, 8, :])
            # plt.subplot(122)
            # plt.title("ground truth")
            # plt.plot(music[i, 8, :])
            # plt.savefig(local_out + "pitch.eps")
            #
            # if not SERVER:
            #     plt.show()
            # else:
            #     plt.close()



            #print(sp_temp_2[i,:,:])
            #print(sp[i, :, :])
            # for j in range(0,tv_dim):
            #     plt.title("TV plot")
            #     plt.subplot(121)
            #     plt.title("from encoder (E(x))")
            #     plt.plot(sp_temp[i, j, :])
            #     plt.subplot(122)
            #     plt.title("from world (ideal h)")
            #     plt.plot(sp[i, j, :])
            #     # plt.colorbar()
            #     plt.savefig(local_out + "TV_" + str(j+1) + ".eps")
            #
            #     if not SERVER:
            #         plt.show()
            #     else:
            #         plt.close()

            # plt.imshow((sp[i, :, :]), cmap='jet', aspect='auto', origin='lower')
            # plt.title("SP original")
            # plt.colorbar()
            # plt.savefig(local_out + "/sp_only.eps")
            #
            # if not SERVER:
            #     plt.show()
            # else:
            #     plt.close()

            plt.title("Spectogram")
            plt.subplot(121)
            plt.imshow(batch_spec[i, :, :].cpu().numpy(), cmap='jet', origin='lower')
            plt.title("original")
            # plt.subplot(122)
            #plt.imshow(D(batch_sp_code).detach().cpu().numpy()[i, :, :], cmap='jet', origin='lower')
            # plt.title("generated")
            plt.colorbar()
            plt.savefig(local_out + "/spectrogram.eps")
            #
            if not SERVER:
                 plt.show()
            else:
                 plt.close()

        return  # Do only the first batch


# def getSpectrograms(mode="evaluation"):
#     '''
#     Generates spectrogram for the original sound (x), D(ideal_h), D(E(x)), world(E(x)) in order to evaluate
#     the accuracy and the decoder and the encoder separatly.
#     '''
#     if mode == "train":
#         loader = train_loader
#
#     elif mode == "evaluation":
#         loader = validation_loader
#
#     elif mode == "train_random":
#         #loader = train_random
#         loader = dev_loader
#
#     else:
#         print("The mode is not understood, therefore we don't compute spectrograms.")
#         return [([], [], [], [])]
#
#     E.eval()
#     # D.eval()
#
#     if args.cuda:
#         E.cuda()
#         # D.cuda()
#
#     pad = 40
#
#     frmlen = 8
#     tc = 8
#     sampling_rate = 16000
#     paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000)]
#
#     # mel_basis = librosa.filters.mel(16000, 1024, n_mels=256)
#     # if args.cuda:
#     #     mel_basis = torch.from_numpy(mel_basis).cuda().float().unsqueeze(0)
#     # else:
#     #     mel_basis = torch.from_numpy(mel_basis).float().unsqueeze(0)
#
#     spectrograms = []
#
#     for batch_idx, data in enumerate(loader):
#
#         batch_wav = Variable(data[0]).contiguous().float()
#         batch_h = Variable(data[2]).contiguous().float()
#         batch_spec = Variable(data[3]).contiguous().float()
#
#         if args.cuda:
#             batch_wav = batch_wav.cuda()
#             batch_spec = batch_spec.cuda()
#             batch_h = batch_h.cuda()
#
#         # ## mel-spectrogram
#         # #If we are using Aud Spec, we do no need to use this blocl of code and let batch_spec be the way it is
#         # batch_spec = batch_spec ** 2
#         # mel_basis_ep = mel_basis.expand(batch_spec.size(0),mel_basis.size(1),mel_basis.size(2))
#         # batch_spec = torch.bmm(mel_basis_ep, batch_spec)
#         # batch_spec = torch.log(batch_spec)
#         # batch_spec[batch_spec == float("-Inf")] = -50
#
#         # predict parameters through waveform
#         # batch_f0, batch_sp, batch_ap = E(batch_spec)
#         fs = 16000
#         batch_sp_code = E(batch_spec)
#         # batch_ap = pw.decode_aperiodicity(batch_ap_code, fs, fftlen)
#         # batch_sp = pw.decode_spectral_envelope(batch_sp_code, fs, fftlen)
#
#         # batch_h_hat = torch.cat([batch_f0, batch_sp, batch_ap], 1)
#
#         ############# World ###################################################################################################
#         # mel_b = librosa.filters.mel(16000, 1024, n_mels=256)
#         # f0_temp = batch_h_hat[:, 0:1, :].data.cpu().numpy().astype('float64')
#         # sp_temp = batch_h_hat[:, 1:514, :].data.cpu().numpy().astype('float64')
#         # ap_temp = batch_h_hat[:, 514:, :].data.cpu().numpy().astype('float64')
#         # f0_temp = batch_f0.data.cpu().numpy().astype('float64')
#         sp_temp = batch_sp_code.data.cpu().numpy().astype('float64')
#         # ap_temp = batch_ap_code.data.cpu().numpy().astype('float64')
#         # f0_temp = batch_h_hat[:,0:1,:].data.cpu().numpy().astype('float64')
#         # sp_temp = batch_h_hat[:,1:129,:].data.cpu().numpy().astype('float64')
#         # ap_temp = batch_h_hat[:,129:,:].data.cpu().numpy().astype('float64')
#
#         spec_wav = np.zeros((batch_wav.shape[0], 128, 250)).astype('float64')
#         # spec_wav = np.zeros((batch_wav.shape[0], 128, 250)).astype('float64')
#
#         #####################################################################################################################
#
#         realSpectrogram = np.array(batch_spec.detach().cpu().numpy())
#         # decoderSpectrogram = np.array(D(batch_h).detach().cpu().numpy())
#         # modelSpectrogram = np.array(D(batch_sp_code).detach().cpu().numpy())
#         # worldSpectrogram = spec_wav
#
#         spectrograms.append((realSpectrogram, decoderSpectrogram, modelSpectrogram))
#         # In order to only save a few spectrograms (64)
#         return spectrograms
#
#     return spectrograms


# def plotSpectrograms(spectrograms, name="", MAX_examples=10):
#     '''
#     Plot or save the spectrograms.
#     '''
#
#     if not os.path.exists(out + "/spectrograms/" + name):
#         os.makedirs(out + "/spectrograms/" + name)
#
#     if name == 'POST_TRAIN_evaluation_data':
#         realSpectrogram, decoderSpectrogram, modelSpectrogram = spectrograms[0]
#         # # with open('Pitch_1.pkl', 'wb') as f:
#         # #     pkl.dump(X_pitch, f)
#         for i in range(max(len(spectrograms[0][0]), MAX_examples)):
#             with open(out + "/spectrograms/" + "RealSpectrogram%d.pkl" % (i), 'wb') as f:
#                 pkl.dump(realSpectrogram[i], f)
#
#             with open(out+"/spectrograms/"+"decoderSpectrogram%d.pkl" %(i), 'wb') as f:
#                pkl.dump(decoderSpectrogram[i], f)
#
#             with open(out + "/spectrograms/" + "modelSpectrogram%d.pkl" % (i), 'wb') as f:
#                 pkl.dump(modelSpectrogram[i], f)
#
#
#     for i in range(min(len(spectrograms[0][0]), MAX_examples)):
#
#         realSpectrogram, decoderSpectrogram, modelSpectrogram = spectrograms[0]
#
#         plt.subplot(221)
#         plt.imshow(realSpectrogram[i], cmap='jet', origin='lower')  # Ground Truth
#         plt.title('Real Spectrogram contrasted')
#
#         plt.subplot(222)
#         plt.imshow(decoderSpectrogram[i], cmap='jet', origin='lower')
#         plt.title('Decoder Spectrogram (using ideal h)')  # D(ideal_h)
#
#         plt.subplot(223)
#         plt.imshow(modelSpectrogram[i], cmap='jet', origin='lower')
#         plt.title('Model Spectrogram: D(E(x))')  # (D(E(X)))
#
#
#         plt.savefig(out + "/spectrograms/" + name + "/" + str(i) + ".eps")
#
#         if not SERVER:
#             plt.show()
#         else:
#             plt.close()


# prepare exp folder
exp_name = 'MirrorNet'
descripiton = 'train MirrorNet without ideal hiddens'
exp_new = 'tmp/'
base_dir = './'
# exp_prepare='/hdd/cong/exp_prepare_folder.sh'
# net_dir=base_dir + exp_new + '/' + exp_name+'/net'
# subprocess.call(exp_prepare+ ' ' + exp_name + ' ' + base_dir + ' ' + exp_new, shell=True)

### Set the name of the folder
out = "figs/NEW_with_only_1_loss_for_speech_nasal_glottal_TVs_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour)
out_dir = "./figs/NEW_with_only_1_loss_for_speech_nasal_glottal_TVs_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour) + "/"

model_dir = "./figs/NEW_with_only_1_loss_for_E_DATE_2021-06-02_H_1/"

# Create sub directory is dont exist
if not os.path.exists(out_dir + exp_new):
    os.makedirs(out_dir + exp_new)

# Create weights directory if don't exist
if not os.path.exists(out_dir + exp_new + 'net/'):
    os.makedirs(out_dir + exp_new + 'net/')


if not os.path.exists(base_dir + exp_new):
    os.makedirs(base_dir + exp_new)

# Create log directory if don't exist
if not os.path.exists(base_dir + exp_new + "log/"):
    os.makedirs(base_dir + exp_new + "log/")

# Create weights directory if don't exist
#if not os.path.exists(base_dir + exp_new + '/net/'):
#   os.makedirs(base_dir + exp_new + '/net/')

# setting logger   NOTICE! it will overwrite current log
log_dir = base_dir + exp_new + "log/" + str(date.today()) + ".log"
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    filename=log_dir,
                    filemode='a')

# global params

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
# parser.add_argument('--decoder_lr', type=float, default=1e-3, # changed from 3e-4
#                     help='learning rate')
parser.add_argument('--encoder_lr', type=float, default=1e-4, # changed from 3e-4
                    help='learning rate')
parser.add_argument('--lr', type=float, default=1e-3,    # changed from 1e-3
                    help='learning rate')
parser.add_argument('--seed', type=int, default=20190101,
                    help='random seed')
parser.add_argument('--val-save', type=str, default=base_dir + exp_new + '/' + exp_name + '/net/cv/model_weight.pt',
                    help='path to save the best model')

parser.add_argument('--train_data', type=str,  default=base_dir+"Nasal_glottal_SI/train_audio_nasal_glottal_2sec.data",
                    help='path to training data')

parser.add_argument('--test_data', type=str,  default=base_dir+"Nasal_glottal_SI/test_audio_nasal_glottal_2sec.data",   #female_voice.data",
                    help='path to testing data')

parser.add_argument('--dev_data', type=str,  default=base_dir+"Nasal_glottal_SI/dev_audio_nasal_glottal_2sec.data",
                    help='path to initialization data')

parser.add_argument('--train_random_data', type=str,  default=base_dir+"SI_autoencoder/EMA_train_files/val_audio_EMA_ext_2sec.data",
                    help='path to train random data')


args, _ = parser.parse_known_args()
print(type(args))
print(args)
np.save(base_dir + exp_new + '/model_arch', args)
args.cuda = args.cuda and torch.cuda.is_available()

np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    print("CUDA ACTIVATED")
else:
    kwargs = {}
    print("CUDA DISABLED")

training_data_path = args.train_data
validation_data_path = args.test_data
dev_data_path = args.dev_data
train_random_data_path = args.train_random_data

dev_loader = DataLoader(BoDataset(dev_data_path),
                        batch_size=args.batch_size,
                        shuffle=False,
                        **kwargs)

train_loader = DataLoader(BoDataset(training_data_path),
                          batch_size=args.batch_size,
                          shuffle=False,
                          **kwargs)
# print ('----------------------', list(train_loader))
# print ('.,.,.,.,.,.,.,.,.,.', (list(train_loader)[0]))

train_random = DataLoader(BoDataset(train_random_data_path),
                          batch_size=args.batch_size,
                          shuffle=False,
                          **kwargs)

validation_loader = DataLoader(BoDataset(validation_data_path),
                               batch_size=args.batch_size,
                               shuffle=False,
                               **kwargs)

eval_byPieces_loader = DataLoader(BoDataset(validation_data_path),
                                  batch_size=1,
                                  shuffle=False,
                                  **kwargs)

args.dataset_len = len(train_loader)
args.log_step = args.dataset_len // 2

E = encoder()
# D = decoder()
# print ('. . . . . . . . . . ', list(E.parameters()))

if args.cuda:
    E.to('cuda')
    # D.to('cuda')
    print("converted model to cuda")

# E.load_state_dict(torch.load('tmp/net/Gui_model_weight_E1.pt'))
# D.load_state_dict(torch.load('tmp/net/Gui_model_weight_D1.pt'))


current_lr = args.lr
E_optimizer = optim.Adam(E.parameters(), lr=args.encoder_lr)
E_scheduler = torch.optim.lr_scheduler.ExponentialLR(E_optimizer, gamma=0.5)   # changed from 0.5

# D_optimizer = optim.Adam(D.parameters(), lr=args.decoder_lr, eps=1e-4)
# D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.5)   # changed from 0.5

parameters = [p for p in E.parameters()]
all_optimizer = optim.Adam(parameters, lr=args.lr)
all_scheduler = torch.optim.lr_scheduler.ExponentialLR(all_optimizer, gamma=0.5)

#total_params = sum(p.numel() for p in E.parameters() if p.requires_grad)
#print('Trainable parameters :', total_params)

# output info

log_print('Experiment start time is {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
log_print('-' * 70)
log_print('Discription:'.format(descripiton))
log_print('-' * 70)
log_print(args)

log_print('-' * 70)
s = 0
for param in E.parameters():
    s += np.product(param.size())
# for param in D.parameters():
#     s += np.product(param.size())

log_print('# of parameters: ' + str(s))

log_print('-' * 70)
log_print('Training Set is {}'.format(training_data_path))
log_print('CV set is {}'.format(validation_data_path))
log_print('-' * 70)

### Set the name of the folder
out = "figs/NEW_with_only_1_loss_for_speech_nasal_glottal_TVs_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour)

if not os.path.exists(out):
    os.makedirs(out)

if not os.path.exists(out + "/loss"):
    os.makedirs(out + "/loss")

print(out)

# # generated the spectrograms before anything to make sure there is no issue with already trained weights or something.
# spec = getSpectrograms("train")
# plotSpectrograms(spec, "before_training_train_data")

'''
       Training
'''
print("Training")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
#
trainTogether_newTechnique(epochs=70, init=True, train_E=True, loader_eval="train", save_better_model=True, name='init')

end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end))  # milliseconds

#
# generate_figures("train_random", name="init") ## evaluation or train
###generate_figures("train", name="init") ## evaluation or train
##
generate_figures("train", name="end")  ## evaluation or train
generate_figures("evaluation", name="end")  ## evaluation or train

# spec = getSpectrograms("train")
# plotSpectrograms(spec, "TRAIN_train_data")
#
# spec = getSpectrograms("evaluation")
# plotSpectrograms(spec, "TRAIN_evaluation_data")

## save trained model
#with open(out_dir + exp_new + '/net/music_model_weight_E1.pt', 'wb') as f:
#    torch.save(E.cpu().state_dict(), f)
#    print("We saved encoder!")
#
#with open(out_dir + exp_new + '/net/music_model_weight_D1.pt', 'wb') as f:
#    torch.save(D.cpu().state_dict(), f)
#    print("We saved decoder!")

# save model with checkpoint
with open(out_dir + exp_new + '/net/music_model_weight_E1.pt', 'wb') as f:
    torch.save({'model_state_dict': E.cpu().state_dict(), 'optimizer_state_dict': E_optimizer.state_dict()}, f)
    print("We saved encoder!")

# with open(out_dir + exp_new + '/net/music_model_weight_D1.pt', 'wb') as f:
#     torch.save({'model_state_dict': D.cpu().state_dict(), 'optimizer_state_dict': D_optimizer.state_dict()}, f)
#     print("We saved decoder!")


print("Done.")