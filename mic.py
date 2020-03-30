#!/usr/bin/python3

import pyaudio
import wave
import os
import json
import threading
import argparse
import audioop
import librosa
import random
import string
import sys
import numpy as np
import time
from util import *

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

PRE_DATA = 'predata/'
#silence tolerance:
RMS_SILENCE = 30
UDP_IP = "localhost"
UDP_PORT = 4444

#repeatedly record audio of fixed length (default length is longest audio file in the PRE_DATA directory)
#note ctrl+c will produce a final file shorter than max_len, close stream and exit before save logic
def record_fixed(dirname, size=None):

    if not size:
        max_len = max_sample_len(path=PRE_DATA)
    else:
        max_len = size

    buffer_len = max_len / CHUNK

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    print ("\n" * 30)
    print("RECORDING, press ctrl+c to stop recording")
    frames = [[]]

    current = 0
    buffer_count = 0
    while 1:
        try:
            #this prints a countdown for each recording
            print(buffer_len - buffer_count)
            data = stream.read(CHUNK)
            buffer_count += 1

            frames[current].append(data)
            if buffer_count >= buffer_len:
                print("========================")
                print("========================")
                print(current)
                print("========================")
                print("========================")
                current += 1
                frames.append([])
                buffer_count = 0

        except KeyboardInterrupt:
            print("STOPPING")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    recid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))

    for i, frames in enumerate(frames):
        filename = PRE_DATA+dirname+"/"+recid+str(i)+".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    print("Finished recording to: ", dirname)
    print('File prefix: ',recid)

    return

#record using silence to split into files
#note this will append quiet_buffers(7) frames on to end of all recordings,
#can use slice to remove these before saving if need small files or offset is a problem
def record(dirname):

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    print ("\n" * 30)
    print("RECORDING, press ctrl+c to stop recording")
    frames = []

    #this many quiet buffers before splitting
    #note this value of 7 is very short because I'm splitting on quick short sounds
    #CHUNK/RATE is seconds of each buffer/chunk, bits / (bits/seconds) = seconds
    #1024/16000 * 7 is a little less than half a second
    quiet_buffers = 7
    prev_quiet = True
    quiets = 0
    current = -1
    while 1:
        try:
            data = stream.read(CHUNK)

#https://stackoverflow.com/questions/47253648/what-do-fragment-and-width-mean-in-audioop-rmsfragment-width
#I think arg of '2' is for 2 bytes because 16bit, ie FORMAT = pyaudio.paInt16
            rms = audioop.rms(data,2)
            if (rms > RMS_SILENCE):
                if (prev_quiet):
                    current += 1
                    frames.append([])
                    prev_quiet = False
                    quiets = 0
                print("Recording: ", current)
                frames[current].append(data)
            else:
                quiets += 1
                if (not prev_quiet and quiets < quiet_buffers):
                    print("Recording: ", current)
                    frames[current].append(data)
                else:
                    prev_quiet = True
                    print("waiting")

        except KeyboardInterrupt:
            print("STOPPING")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    recid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))

    for i, frames in enumerate(frames):
        filename = PRE_DATA+dirname+"/"+recid+str(i)+".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    print("Finished recording to:", dirname)
    print('File prefix: ',recid)

    return

def udp_listener(sock):
    hits = {}
    while True:
        try:
            data, server = sock.recvfrom(1024)
            data = json.loads(data)
            # print('data: ', data)
            # print('hits: ', hits)
            if not data['label'] in hits:
                hits[data['label']] = time.time()
            if time.time() - hits[data['label']] > 2:
                if data['probability'] < 0.6: continue
                print('hit', data)
                hits[data['label']] = time.time()
            # else:
            #     print(f"too soon {time.time() - hits[data['label']]}")
        except KeyboardInterrupt:
            print("STOPPING")
            break

#send mic data to udp
def stream():
    import socket

    sock = socket.socket(socket.AF_INET, # Internet
                                 socket.SOCK_DGRAM) # UDP

    listen_thread = threading.Thread(target=udp_listener, args=(sock,))
    listen_thread.daemon = True
    listen_thread.start()

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    print ("\n" * 30)
    print("Streaming, press ctrl+c to stop recording")

    while 1:
        try:
            chunk = stream.read(CHUNK)
            rms = audioop.rms(chunk,2)
            if rms < RMS_SILENCE: continue
            sock.sendto(chunk, (UDP_IP, UDP_PORT))
        except KeyboardInterrupt:
            print("STOPPING")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    return

def rms():

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    print ("\n" * 30)
    print("Streaming, press ctrl+c to stop ")

    while 1:
        try:
            chunk = stream.read(CHUNK)
            rms = audioop.rms(chunk,2)
            print("rms: ", rms)

        except KeyboardInterrupt:
            print("STOPPING")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='audio recognition tools')
    #TODO add input_device_index arg on pyaudio for using different mics
    # parser.add_argument('-d', '--device', default="-1", dest="device", type=int, help="Select a different microphone (give device ID)")
    parser.add_argument('-l', '--label', help="The name of the directory to output the files. This dir must already exist")
    parser.add_argument('-f', '--fixed-size', action="store_true", help="Record fixed number of frames per file, default is result of --maxlen, can override by providing final argument (-f 1028) ")
    parser.add_argument('-m', '--maxlen', action="store_true", help="The maximum length of audio in the predata dir, units is hardcoded as int16 so double this for number of bytes")
    parser.add_argument('-s', '--stream', action="store_true", help="Stream audio to udp port (hardcoded port 4444)")
    parser.add_argument('-r', '--rms', action="store_true", help="Output the RMS of the mic data. Set the RMS_SILENCE constant to match the ambient background noise on your mic to split recordings on silence")
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(args)
    if args.maxlen:
        print(max_sample_len(path=PRE_DATA))
    elif args.rms:
        rms()
    elif args.stream:
        stream()
    elif args.fixed_size:
        if not args.label:
            print("Must provide the --label argument") 
        elif not os.path.isdir(PRE_DATA+args.label):
            print("The label argument must be a preexisiting subdirectory in the predata dir")
        else:
            if len(args.args) > 0:
                record_fixed(args.label,int(args.args[0]))
            else:
                record_fixed(args.label)
    elif args.label:
        if not os.path.isdir(PRE_DATA+args.label):
            print("The label argument must be a preexisiting subdirectory in the predata dir")
        else:
            record(args.label)
