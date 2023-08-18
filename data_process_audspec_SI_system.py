#!/usr/bin/env python
# coding: utf-8
'''
This performs the data_process script to produce audspecs

Authors : Yashish

'''
import matplotlib

SERVER = True  # This variable enable or disable matplotlib, set it to true when you use the server!
if SERVER:
    matplotlib.use('Agg')

import numpy as np
# from scipy import signal
import os
# import string
import librosa
# import time
# import random
import h5py
# import pyworld as pw
from tqdm import tqdm
import sys
# import matplotlib.pyplot as plt
# import matplotlib
# import matplotlib.pyplot as plt
# import scipy.io.wavfile
# from random_generation import get_f0, get_ap, get_random_h
import nsltools as nsl
# from scipy.io import loadmat

np.set_printoptions(threshold=sys.maxsize)
tv_dim = 5
tv_timesteps = 200

def min_max_norm(X):
    # X_samples = X.shape[0]
    # X_pitch = X.flatten()
    min_val = np.min(X)
    max_val = np.max(X)
    X_pitch = (X - min_val) / (max_val - min_val)
    #print(std)
    return X_pitch


def generate_data(audio, path_params, audio_time=2, sampling_rate=16000, random=False):

    audio_name = audio[:-4]
    audspec_out = audio_name + "_audspec" + ".data"  # save data to this path

    audio_len = np.int(audio_time * sampling_rate)
    # cur_sample = 0  # current sample generated
    file_id = 0

    s_wav = audio
    s_wav, s_sr = librosa.load(s_wav, sr=sampling_rate)
    file_len = len(s_wav)

    # chunk_num = len(s_wav) // audio_len
    # extra_len = len(s_wav) % audio_len
    # if extra_len > audio_len // 2:
    #     trim_len = (chunk_num + 1) * audio_len
    # else:
    #     trim_len = chunk_num * audio_len

    first_test = True
    if len(s_wav) <= audio_len:
        no_segs = 1
        spk_wav = np.concatenate([s_wav, np.zeros(audio_len - len(s_wav))])
        spk_wav = np.expand_dims(spk_wav, axis=0)
    elif len(s_wav) > audio_len:
        # spk_wav = s_wav[:audio_len]
        no_segs = len(s_wav) // audio_len + 1
        for i in range(0, no_segs):
            if first_test:
                spk_data_seg = s_wav[:audio_len]
                spk_data_seg = np.expand_dims(spk_data_seg, axis=0)
                spk_wav = spk_data_seg
                first_test = False
            elif i < no_segs - 1:
                spk_data_seg = s_wav[audio_len * i:audio_len * (i + 1)]
                spk_data_seg = np.expand_dims(spk_data_seg, axis=0)
                spk_wav = np.vstack((spk_wav, spk_data_seg))
            elif i == no_segs - 1:
                pad_amt = (audio_len * no_segs) - file_len
                spk_data_seg = np.concatenate([s_wav[(audio_len * i):], np.zeros(pad_amt)])
                spk_data_seg = np.expand_dims(spk_data_seg, axis=0)
                spk_wav = np.vstack((spk_wav, spk_data_seg))

    print('trim finished')  # spk_wav_tmp.shape[0]=Number of files in data directory or Num of examples= N

    spk_wav_tmp = spk_wav
    # h_tmp = h_tmp[:spk_wav_tmp.shape[0], :, :]
    # spk_wav_tmp.shape= (Num of examples (N), 32000)

    spk_tmp = np.zeros((spk_wav_tmp.shape[0], spk_wav_tmp.shape[1]))  # raw speech with normalized power
    # h_tmp = np.zeros((spk_tmp.shape[0], tv_dim, 250))  # ideal hiddens from world
    spec_tmp_513 = np.zeros(
        (spk_wav_tmp.shape[0], 128, 250))  # dimensions of the AudSpec - needs to be softcoded for scalability
    #spec_tmp_513_pw = np.zeros((spk_wav_tmp.shape[0], 128,
    #                           250))  # dimensions of the Reconstructed AudSpec - needs to be softcoded for scalability
    print(spec_tmp_513.shape, 'spec_tmp_513')

    '''Parameters for AudSpectrogram'''
    frmlen = 8
    tc = 8
    paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000)]
    print(sampling_rate)

    pad = 40
    for i in tqdm(range(spk_wav_tmp.shape[0])):
        if i % 100 == 0:
            print(i)
        wav = spk_wav_tmp[i, :].copy().astype('float64')
        wav = wav.reshape(-1)
        # wav=nsl.unitseq(wav)    #THis here causes the problem: RuntimeWarning: overflow encountered in exp

        wav = wav / np.sqrt(np.sum(wav ** 2))  # power normalization

        wav = nsl.unitseq(wav)  # THis here causes the problem: RuntimeWarning: overflow encountered in exp

        # #wav.shape=(32000,)
        if not random:
            spk_tmp[i, :] = wav  # this is saved

        spec513 = nsl.wav2aud(wav, paras_c)  # audSpec
        # print (spec513.shape, 'spec513--line 116')
        spec_tmp_513[i, :, 0:250] = spec513.T  # AudSpec
        # print (spec_tmp_513[i,:,:])

    # write data
    # print (spec_tmp_513)
    dset = h5py.File(audspec_out, 'w')
    print(spk_tmp.shape)
    spk_set = dset.create_dataset('speaker', shape=(spk_tmp.shape[0], spk_tmp.shape[1]), dtype=np.float64)
    # hid_set = dset.create_dataset('hidden', shape=(spk_tmp.shape[0], tv_dim, tv_timesteps), dtype=np.float64)
    # spec513_set = dset.create_dataset('spec513', shape=(spk_tmp.shape[0], 513, 401), dtype=np.float64)
    # spec_513_pw_set = dset.create_dataset('spec513_pw', shape=(spk_tmp.shape[0], 513, 401), dtype=np.float64)

    spec513_set = dset.create_dataset('spec513', shape=(spk_tmp.shape[0], 128, 250), dtype=np.float64)
    # spec_513_pw_set = dset.create_dataset('spec513_pw', shape=(spk_tmp.shape[0], 128, 251), dtype=np.float64)

    spk_set[:, :] = spk_tmp
    # hid_set[:, :, :] = h_tmp
    # spec513_set[:,:,:] = d
    # spec513_set = []
    spec513_set[:, :, :] = spec_tmp_513
    dset.close()

    return audspec_out, no_segs, file_len


if __name__ == "__main__":
    audio_dir = 'sample_audio'
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            audio_file = audio_dir + '/' + file
            generate_data(audio_file)

    print('finished')

