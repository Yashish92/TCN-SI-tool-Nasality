"""
This script writes the predicted latent space and ideal latent space parameters
to .mat files to be visualized using the matlab TV plotting scripts

"""


from __future__ import print_function
import argparse

import matplotlib

SERVER = True  # This variable enable or disable matplotlib, set it to true when you use the server!
if SERVER:
    matplotlib.use('Agg')
import sys
import os
import numpy as np
# from pink_tromb_lib.pynkTrombone.voc import Voc
# from pink_tromb_lib.examples import voc_synthesize_sm as voc_syn
import pickle as pkl
import datetime
import scipy
from scipy.io.wavfile import write
import inspect
import nsltools as nsl
import scipy.io as scio
from scipy.io import savemat

# speaker_index = 0
# print(speaker_index, '-->Speaker Index')
analyze_sp = False
analyze_sectrograms = False
recon_from_latent = True
frmlen = 8
tc = 8
sampling_rate = 16000
paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000.0)]

fs = sampling_rate
figs_folder = 'SI_model_outputs/NEW_with_only_1_loss_for_speech_nasal_only_TVs_2023-03-04_H_18'
# name = figs_folder + 'Speaker%d_results/' % (speaker_index)

if recon_from_latent:
    print('recon from latent')

    # f0_ideal = np.squeeze(f0_ideal[speaker_index, :, :])
    # sp_ideal = np.squeeze(sp_ideal[speaker_index, :, :])
    for speaker_index in range(0, 10):
        name = figs_folder + '/Speaker%i_results/' % (speaker_index)
        if not os.path.exists(name):
            os.makedirs(name)

        with open(figs_folder + '/end_evaluation/lentent_space/sp_ideal.pkl', 'rb') as f:
          ap_ideal = pkl.load(f)

        with open(figs_folder + '/end_evaluation/lentent_space/sp_hat.pkl', 'rb') as f:
            ap_hat = pkl.load(f)

        with open(figs_folder + '/end_evaluation/lentent_space/wave_files.pkl', 'rb') as f:
            wave_file = pkl.load(f)

        ap_ideal = np.squeeze(ap_ideal[speaker_index, :, :])
        tv_ideal = ap_ideal[:].copy(order='C')  # 406, 513

        # save to a .mat file
        mdic = {"tv": tv_ideal}
        savemat(name + 'tv_ideal.mat', mdic)

        # # f0_hat = np.squeeze(f0_hat[speaker_index, :, :])
        ap_hat = np.squeeze(ap_hat[speaker_index, :, :])
        tv_hat = ap_hat[:].copy(order='C')  # 406, 513
        # sp_hat = np.squeeze(sp_hat[speaker_index, :, :])

        # save to a .mat file
        mdic = {"tv": tv_hat}
        savemat(name + 'tv_predict.mat', mdic)

        wave_file = np.squeeze(wave_file[speaker_index,:])
        wave = wave_file[:].copy(order='C')  # 406, 513

        scipy.io.wavfile.write(name + 'wave_orig.wav', sampling_rate, wave)
