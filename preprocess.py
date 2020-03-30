#!/usr/bin/python3
#refactored and added audio length normalization and spectrogram code to https://github.com/manashmandal/DeadSimpleSpeechRecognizer
import librosa
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import json

from util import *

SR=16000

def wav2mfcc(file_path, n_mfcc, max_sample_length=None):
    wave, sr = librosa.load(file_path, mono=True, sr=SR)
    #TODO see https://github.com/manashmandal/DeadSimpleSpeechRecognizer
    #I think this is like a primitive way of lowering the sample rate?
    #Smaller input would mean better perf but would it hurt accuracy?
    #If included, need to move this to data_to_mfcc because udp_listen uses that fn
    # wave = wave[::3]

    if max_sample_length:
        wave = random_pad(wave, max_sample_length)

    return data_to_mfcc(wave, n_mfcc) 

def data_to_mfcc(data, n_mfcc):
    #upstream if we call data[::3] we need to call this or get NaN fortran continuous error
    wave = np.asfortranarray(data)

    #default n_mfcc is 20, 40 is used in kaggle tutorial
    mfcc = librosa.feature.mfcc(wave, sr=SR, n_mfcc=n_mfcc)

    return mfcc

def wav2spec(file_path, max_sample_length=None):
    wave, sr = librosa.load(file_path, mono=True, sr=SR)
    # wave = wave[::3]

    if max_sample_length:
        wave = random_pad(wave, max_sample_length)

    return data_to_spec(wave)

def data_to_spec(data):
    wave = np.asfortranarray(data)
    spectrogram = librosa.feature.melspectrogram(wave, sr=SR)
    log_spec = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spec

def save_data_to_array(path=DATA_PATH, n_mfcc=40):
    labels, _, _ = get_labels(path)
    
    max_sample_length = max_sample_len()

    for label in labels:
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, n_mfcc=n_mfcc, max_sample_length=max_sample_length)
            mfcc_vectors.append(mfcc)
        np.save(PREPROC_PATH + label + '.npy', mfcc_vectors)

def save_raw_data_to_array(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    
    max_sample_length = max_sample_len()

    for label in labels:
        raw_audio = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving audio of label - '{}'".format(label)):
            wave, sr = librosa.load(wavfile, mono=True, sr=SR)
            wave = random_pad(wave, max_sample_length)
            raw_audio.append(wave)
        np.save(PREPROC_PATH + label + '.npy', raw_audio)

def save_spec_data_to_array(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    
    max_sample_length = max_sample_len()

    for label in labels:
        spectrograms = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving spectrograms of label - '{}'".format(label)):
            spec = wav2spec(wavfile, max_sample_length=max_sample_length)
            spectrograms.append(spec)
        np.save(PREPROC_PATH + label + '.npy', spectrograms)

def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(PREPROC_PATH + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(PREPROC_PATH + label + '.npy')
        #stack is [X,x]
        X = np.vstack((X, x))
        #append spreads [...y, ...np.full]
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)
