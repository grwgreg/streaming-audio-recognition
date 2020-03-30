#!/usr/bin/python3
from tensorflow import keras
import socket
import argparse
import librosa
import audioop
import json
import numpy as np
import io
import soundfile as sf
from preprocess import *

# for testing, use wave lib to save the incoming mic data as audio files on disk
# import wave

def udp_listen(model_type, model_name, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", port))

    model = load_model(model_name)

    #mfcc and spec models currently use mean and stdev for standardizing input
    #1d conv only normalizes uses max and min values from the audio data
    #TODO make a single load_stats(model_name, model_type) fn in util.py?
    stats = {}
    if model_type == '1dconv':
        saved_stats = load_stats(model_name)
        stats['max_data'] = float(saved_stats['max_data'])
        stats['min_data'] = float(saved_stats['min_data'])
    else:
        saved_stats = np.load('stats/' + model_name + '.npy')
        stats['input_std'], stats['input_mean'] = saved_stats

    max_sample_length = max_sample_len()

    data = np.zeros((1,), dtype='int16')

    #call eval every 4 chunks of mic data, chunk/samplerate units are data/(data/second) = second
    #1024/16k = .064, times 4 chunks should be roughly .25 seconds
    #if we only eval every .5seconds but input is .4seconds we miss .1 seconds
    #if we eval every .1 seconds but input is .4 seconds, we eval part of each audio sample 4 times
    #I think it's better to have some overlap for highest recognition. If sound appears on edge I
    #would expect it to miss it often
    eval_every = 4
    count = 0

    while 1:

        count += 1
        chunk, ip = sock.recvfrom(1024)

        next_data = np.frombuffer(chunk, dtype='int16')
        data = np.append(data,next_data)

        if count % eval_every == 0:

            #TODO might get some perf improvement by manually handling the data buffer instead of growing/slicing
            #Could try a python dequeue like a circular buffer? but then have to merge chunks before evaling.
            #Could also allocate a big buffer initially and reset index at some max size
            #ie: to_send = data[offset*CHUNK:(offset+1)*CHUNK]
            #if offset > x: data[0:max_sample_length] = data[-max_sample_length] && offset = 0
            data = data[-max_sample_length:]
            result = eval_data(data, model, model_type, max_sample_length, stats)
            label = get_labels()[0][
                    np.argmax(result)
            ]
            msg = {'label': label, 'probability': float(max(result[0]))}
            sock.sendto(str.encode(json.dumps(msg)), ip)

def eval_data(data, model, model_type, max_sample_length, stats):

    # https://stackoverflow.com/questions/52369925/creating-wav-file-from-bytes
    # https://pysoundfile.readthedocs.io/en/0.9.0/#raw-files
    # https://github.com/bastibe/SoundFile/blob/0.9.0/soundfile.py
    # ctrl f for subtype, I think pcm_16 same as pyaudio.paInt16
    audio_data, sr = sf.read(io.BytesIO(data), channels=1, samplerate=16000, subtype='PCM_16', format='RAW')

    # If bad mic predictions but good from wav files, write above audio_data to a file to make sure it isn't garbled

    # if io.BytesIO method causing problems, use code below to write mic data to a temp wav file and then it open via librosa
    # Can also use this trick to save all incoming mic data for testing, ie make sure the overlap/stride is what is expected
    # filename = "tmp/tmp.wav"
    # wf = wave.open(filename, 'wb')# wf has no rewind() method so we have to open/close on each eval...
    # wf.setnchannels(1)
    # wf.setsampwidth(2)#2 bytes because pyaudio.paint16
    # wf.setframerate(16000)
    # wf.writeframes(data)
    # wf.close()
    # audio_data, sr = librosa.load(filename, mono=True, sr=None)

    if audio_data.shape[0] < max_sample_length:
        audio_data = random_pad(audio_data, max_sample_length)

    if model_type == 'mfcc':
        features = data_to_mfcc(audio_data, n_mfcc=40)#n_mfcc should be saved in stats TODO
    elif model_type == 'spec':
        features = data_to_spec(audio_data)
    elif model_type == '1dconv':
        features = audio_data
    else:
        raise Exception('unknown model type')


    #reshape and normalize/standardize, first dimension added is for 1 sample, last is for channel
    if model_type == 'mfcc' or model_type == 'spec':
        features = features.reshape(1, features.shape[0], features.shape[1], 1)
        features = (features - stats['input_mean']) / stats['input_std']
    elif model_type == '1dconv':
        features = features.reshape(1, features.shape[0], 1)
        features = audio_norm(features, stats['max_data'], stats['min_data'])
    else:
        raise Exception('unknown model type')

    result = model.predict(features)
    # labels = get_labels()[0][
    #         np.argmax(result)
    # ]
    # print('result',result)
    # print('labels',labels)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='keras audio recognition udp listener')
    parser.add_argument('-1', '--onedconv', action="store_true", help="Use the 1d convolution model type, requires a stats file with min/max of data")
    parser.add_argument('-s', '--spectrogram', action="store_true", help="Use the 2d convolution model type with spectrogram input, requires stats file with mean and standard deviation")
    parser.add_argument('-m', '--mfcc', action="store_true", help="Use the 2d convolution model type with mfcc input, requires stats file with mean and standard deviation")
    parser.add_argument('-n', '--model-name', help="The filename without extension of the model and stats file. ie [-n mymodel] for models/mymodel.json, models/mymodel.h5, stats/mymodel.npy|json")
    parser.add_argument('-p', '--port', default="4444", type=int, help="The udp port to listen on")
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.onedconv:
        model_type='1dconv'
    elif args.spectrogram:
        model_type='spec'
    elif args.mfcc:
        model_type='mfcc'
    
    udp_listen(model_type, args.model_name, port=args.port)
