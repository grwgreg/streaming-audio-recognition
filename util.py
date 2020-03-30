#!/usr/bin/python3
import librosa
import os
from keras.utils import to_categorical
import numpy as np
import json

DATA_PATH = "./data/"
PREPROC_PATH = "./preproc/"
MODELS_PATH = "./models/"
STATS_PATH = "./stats/"

def save_model(model, name="example"):
    # serialize model to JSON
    model_json = model.to_json()
    with open(MODELS_PATH + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(MODELS_PATH + name + ".h5")

#https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
#if you get this error you are trying to load a model that was saved with the keras python package, not the tensorflow.keras package
#I have it using tensorflow.keras so tensorboard works but no harm in changing to top level keras if you're not using tensorboard
def load_model(name="example"):
    from tensorflow.keras.models import model_from_json
    json_file = open(MODELS_PATH + name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(MODELS_PATH + name + ".h5")
    return loaded_model

#when reading in mic data we need to standardize using same as the data we trained on
def save_stats(name, **stats):
    # np datatypes are annoying
    # https://stackoverflow.com/questions/53082708/typeerror-object-of-type-float32-is-not-json-serializable
    copy = {}
    for k,v in stats.items():
        copy[k] = str(v)
    data = json.dumps(copy)
    with open(STATS_PATH + name + ".json", "w") as json_file:
            json_file.write(data)
                
def load_stats(name):
    json_file = open(STATS_PATH + name + ".json", "r")
    stats_json = json_file.read()
    json_file.close()
    return json.loads(stats_json)

def max_sample_len(path=DATA_PATH, sr=16000):
    lens = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if not (filename.startswith('.')):
                filepath = os.path.join(dirname, filename)
                try:
                    signal, sr = librosa.load(filepath, sr=sr)
                except NoBackendError as e:
                    print("Could not open audio file {}".format(filepath))
                    raise e
                lens.append(signal.shape[0])
    return max(lens)

def random_pad(signal, max_len):
    max_offset = max_len - len(signal)
    offset = np.random.randint(max_offset+1)
    return np.pad(signal, (offset, max_len - len(signal) - offset), "constant")

def random_pad_with(signal, pad_values):
    max_len = len(pad_values)
    max_offset = max_len - len(signal)
    offset = np.random.randint(max_offset+1)
    return np.concatenate((pad_values[:offset],signal,pad_values[len(signal)+offset:]))

#this assumes all files in silence dir are the max length
def random_silence_wav():
    for _, _, filenames in os.walk('./data/silence'):
        filename = filenames[np.random.randint(len(filenames))]
        wav, _ = librosa.core.load('./data/silence/' + filename, sr=16000)
        return wav

def random_silence_pad(signal):
  silence_wav = random_silence_wav()
  return random_pad_with(signal, silence_wav)

#https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
#TODO copied from above, range is -.5 to .5, do negative values help relu or something?
def audio_norm(data, max_data, min_data):
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    #to_categorical return one hot encoded labels
    return labels, label_indices, to_categorical(label_indices)
