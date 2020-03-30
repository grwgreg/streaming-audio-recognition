# Keras Streaming Audio Recognition

## Introduction

This project contains scripts and jupyter notebooks for recording audio, training a keras model and streaming mic data to a process for evaluation. Much of the code is based on https://github.com/manashmandal/DeadSimpleSpeechRecognizer and https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data. The default setup assumes udpserver.py is run from within a docker container with udp port 4444 exposed, but the udpserver.py file can be run on another machine.

```
$ docker run --name tenflow2020 -it -v ~/tensorvol:/tensorvol -p 8888:8888 -p 6006:6006 -p 4444:4444/udp tensorflow/tensorflow:latest-py3-jupyter /bin/bash
```

Port 8888 is for the jupyter notebook, 6006 for tensorboard and 4444/udp for the mic. See https://www.tensorflow.org/install/docker for the current tensorflow docker installation. This doc assumes the project is located at /tensorvol.

## Data preparation

Create a directory in /predata for each sound you want the model to be able to recognize. We will use the mic.py script to split the mic input into multiple files, splitting after a period of silence.

```
$ python3 mic.py -r
```

This will output the rms value (essentially the average volume) of your mic. The default is 30. Set the RMS_SILENCE constant in the mic.py file to your silence threshold.

```
$ python3 mic.py -l snap
```

If we have a directory /predata/snap it will record into this directory. Do a few recordings for each sound/label, later we will record many more with a fixed length.

```
$ python3 mic.py -m
```

This will output the length of the longest recording in your predata directory. You don't need to save this anywhere but its good to know about this command because if a larger file sneaks into your training data it may mess things up. The max_sample_len function in util.py is responsible for this output and is used throughout the code to get the size of the input for the keras model both during training and from the streaming mic input.

With the length of the input known, the next step is to make many recordings for each sound. How many you need ultimately depends on how many parameters your model has. If you have a deep model with many params and too little data, it will overfit and memorize the training data instead of learning the actual underlying pattern. But if you use a tiny model then you can expect the recognition accuracy to not be so great as it has less freedom to learn how to recognize the pattern. For the models included in the jupyter notebooks, 150 samples per sound got around 92% validation accuracy and 350 samples got up to 98-99% validation accuracy. 

```
$ python3 mic.py -l snap -f
```

The -f flag will record fixed length files using the size of the largest file in the /predata dir. You may want to open your file manager or use some method to filter out the smaller files from when recording was cut off or from the initial recordings. The code does pad inputs to be the same size but I haven't tested if it affects training. When done preparing the data, copy it to the /data directory.

## Model creation and training

If using the docker container, get a shell and run the startup script that starts the jupyter process.

```
$ docker exec -it tenflow2020 bash
#now within the container:
$ cd /tensorvol
$ ./startup.sh
```

There is a jupyter notebook for each of the three model types. There are two 2d convolution models, one uses mfcc as input and another uses spectrograms. The third model is a 1d conv net on the raw audio data. The steps in each file are basically the same. First the audio data is processed into whatever form is needed for the model and saved in /preproc. The stddev, mean, min, and max of the data is saved into stats/ because we normalize the data before training and will need to repeat the same preprocessing steps on the incoming mic data. The trained model data is saved into models/ using the NAME variable set within the notebook and a timestamp so you don't overwrite a previous model. You can skip over the tensorboard stuff if you want by removing the tensorboard callback from the model.fit(...) arguments and skipping the cells that setup tensorboard.

## Streaming mic data for evaluation

From within the docker container:

```
# within docker container
$ cd /tensorvol
$ python3 udpserver.py -m -n mfccmodel1584667400
```

The -n argument takes the name of the model without any extension, it should match the model and stats filenames saved from the jupyter notebook. The -m is for the mfcc model, if the model was trained on spectrograms we would instead pass -s and for the 1d convolution model we would pass -1. The default port to listen on is 4444 but the -p flag will override this value.

Now from our machine with the microphone, run the mic.py program to stream the mic data and print out the results,

```
$ python3 mic.py -s
```

If all is well you will begin seeing the results printed to the console,

```
hit {'label': 'snap', 'probability': 0.7509004473686218}
```

See the udp_listener function in the mic.py file for how the incoming results are handled. The code is set up to only recognize a single label once every 2 seconds and only if the probability is over 0.6.
