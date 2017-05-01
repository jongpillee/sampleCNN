import os
import os.path
import numpy as np
import time
import cPickle as cP
import librosa
import re

np.random.seed(0)

class Options(object):
	def __init__(self):
		self.train_size = 15244 #15244
		self.valid_size = 1529 #1529
		self.test_size = 4332 #4332
		self.num_tags = 50
		self.batch_size = 23 #16 #23
		self.num_frames_per_song = 1250
		self.conv_window_size = [0]
		self.hop_size = [2]
		self.nb_epoch = 1000
		self.lr = [0.01,0.002,0.0004,0.00008,0.000016] #0.04
		self.lrdecay = 1e-6
		self.gpu_use = 1
		self.model_list = ['8192frames_power2_input743_ETE'] #['15625frames_power5_ETE_doubleFilter']
		self.activ = 'relu'
		self.regul = 'l2(1e-7)'
		self.init = 'he_uniform'
		self.patience = 3 #4
		self.partition = 37 #37
		self.max_q_size = 1

options = Options()
theano_config = 'mode=FAST_RUN,device=gpu%d,floatX=float32,lib.cnmem=0.4' % options.gpu_use #lib.cnmem=0.7
os.environ['THEANO_FLAGS'] = theano_config

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten,Reshape,Permute
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.layers.convolutional import ZeroPadding1D,Convolution1D,MaxPooling1D,AveragePooling1D,UpSampling1D
from keras.models import model_from_json,Model
from keras.callbacks import Callback,ModelCheckpoint,EarlyStopping
from keras.layers import Input,merge
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import sys
from eval_tags import *

activ = options.activ
regul = eval(options.regul)
init = options.init

feature_path = '/media/ssd2/pubMagnatagatune_mp3s_to_npy/'
label_path = '../pubtagatune1DCNN_beat/'

#load data
train_list = cP.load(open(label_path + 'train_list_pub.cP','r'))
valid_list = cP.load(open(label_path + 'valid_list_pub.cP','r'))
test_list = cP.load(open(label_path + 'test_list_pub.cP','r'))

def calculate_window_size(hop_size,window):
	return hop_size*(window+1)

def calculate_sample_length(hop_size,num_frames_input):
	return hop_size*(num_frames_input)

def calculate_num_segment(hop_size,num_frame_input):
	return int((int(640000/hop_size))/num_frame_input)


def build_model8192frames_power2_input743_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv11 = Convolution1D(256,2,border_mode='same',init=init)(MP10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	MP11 = MaxPooling1D(pool_length=2)(activ11)

	conv13 = Convolution1D(256,2,border_mode='same',init=init)(MP11)
	bn13 = BatchNormalization()(conv13)
	activ13 = Activation(activ)(bn13)
	MP13 = MaxPooling1D(pool_length=2)(activ13)

	conv14 = Convolution1D(512,2,border_mode='same',init=init)(MP13)
	bn14 = BatchNormalization()(conv14)
	activ14 = Activation(activ)(bn14)
	MP14 = MaxPooling1D(pool_length=2)(activ14)

	conv15 = Convolution1D(512,1,border_mode='same',init=init)(MP14)
	bn15 = BatchNormalization()(conv15)
	activ15 = Activation(activ)(bn15)
	dropout1 = Dropout(0.5)(activ15)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model



def build_model19683frames_power3_ETE_DoubleLayer_AllModel(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(128,3,subsample_length=3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)

	conv3 = Convolution1D(128,3,border_mode='same',init=init)(activ1_1)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(128,3,subsample_length=3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)

	conv4 = Convolution1D(256,3,border_mode='same',init=init)(activ3_1)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(256,3,subsample_length=3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)

	conv5 = Convolution1D(256,3,border_mode='same',init=init)(activ4_1)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(256,3,subsample_length=3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)

	conv6 = Convolution1D(256,3,border_mode='same',init=init)(activ5_1)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(256,3,subsample_length=3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(activ6_1)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,subsample_length=3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(activ7_1)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,subsample_length=3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)

	conv9 = Convolution1D(256,3,border_mode='same',init=init)(activ8_1)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(256,3,subsample_length=3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)

	conv10 = Convolution1D(512,3,border_mode='same',init=init)(activ9_1)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	conv10_1 = Convolution1D(512,3,subsample_length=3,border_mode='same',init=init)(activ10)
	bn10_1 = BatchNormalization()(conv10_1)
	activ10_1 = Activation(activ)(bn10_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(activ10_1)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model128frames_power2_input1486_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv13 = Convolution1D(512,2,border_mode='same',init=init)(MP5)
	bn13 = BatchNormalization()(conv13)
	activ13 = Activation(activ)(bn13)
	MP13 = MaxPooling1D(pool_length=2)(activ13)

	conv14 = Convolution1D(512,2,border_mode='same',init=init)(MP13)
	bn14 = BatchNormalization()(conv14)
	activ14 = Activation(activ)(bn14)
	MP14 = MaxPooling1D(pool_length=2)(activ14)

	conv15 = Convolution1D(512,1,border_mode='same',init=init)(MP14)
	bn15 = BatchNormalization()(conv15)
	activ15 = Activation(activ)(bn15)
	dropout1 = Dropout(0.5)(activ15)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model256frames_power2_input1486_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv13 = Convolution1D(512,2,border_mode='same',init=init)(MP6)
	bn13 = BatchNormalization()(conv13)
	activ13 = Activation(activ)(bn13)
	MP13 = MaxPooling1D(pool_length=2)(activ13)

	conv14 = Convolution1D(512,2,border_mode='same',init=init)(MP13)
	bn14 = BatchNormalization()(conv14)
	activ14 = Activation(activ)(bn14)
	MP14 = MaxPooling1D(pool_length=2)(activ14)

	conv15 = Convolution1D(512,1,border_mode='same',init=init)(MP14)
	bn15 = BatchNormalization()(conv15)
	activ15 = Activation(activ)(bn15)
	dropout1 = Dropout(0.5)(activ15)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model512frames_power2_input1486_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv13 = Convolution1D(512,2,border_mode='same',init=init)(MP7)
	bn13 = BatchNormalization()(conv13)
	activ13 = Activation(activ)(bn13)
	MP13 = MaxPooling1D(pool_length=2)(activ13)

	conv14 = Convolution1D(512,2,border_mode='same',init=init)(MP13)
	bn14 = BatchNormalization()(conv14)
	activ14 = Activation(activ)(bn14)
	MP14 = MaxPooling1D(pool_length=2)(activ14)

	conv15 = Convolution1D(512,1,border_mode='same',init=init)(MP14)
	bn15 = BatchNormalization()(conv15)
	activ15 = Activation(activ)(bn15)
	dropout1 = Dropout(0.5)(activ15)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model1024frames_power2_input1486_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv13 = Convolution1D(512,2,border_mode='same',init=init)(MP8)
	bn13 = BatchNormalization()(conv13)
	activ13 = Activation(activ)(bn13)
	MP13 = MaxPooling1D(pool_length=2)(activ13)

	conv14 = Convolution1D(512,2,border_mode='same',init=init)(MP13)
	bn14 = BatchNormalization()(conv14)
	activ14 = Activation(activ)(bn14)
	MP14 = MaxPooling1D(pool_length=2)(activ14)

	conv15 = Convolution1D(512,1,border_mode='same',init=init)(MP14)
	bn15 = BatchNormalization()(conv15)
	activ15 = Activation(activ)(bn15)
	dropout1 = Dropout(0.5)(activ15)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model2048frames_power2_input1486_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv13 = Convolution1D(512,2,border_mode='same',init=init)(MP9)
	bn13 = BatchNormalization()(conv13)
	activ13 = Activation(activ)(bn13)
	MP13 = MaxPooling1D(pool_length=2)(activ13)

	conv14 = Convolution1D(512,2,border_mode='same',init=init)(MP13)
	bn14 = BatchNormalization()(conv14)
	activ14 = Activation(activ)(bn14)
	MP14 = MaxPooling1D(pool_length=2)(activ14)

	conv15 = Convolution1D(512,1,border_mode='same',init=init)(MP14)
	bn15 = BatchNormalization()(conv15)
	activ15 = Activation(activ)(bn15)
	dropout1 = Dropout(0.5)(activ15)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model4096frames_power2_input1486_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv13 = Convolution1D(512,2,border_mode='same',init=init)(MP10)
	bn13 = BatchNormalization()(conv13)
	activ13 = Activation(activ)(bn13)
	MP13 = MaxPooling1D(pool_length=2)(activ13)

	conv14 = Convolution1D(512,2,border_mode='same',init=init)(MP13)
	bn14 = BatchNormalization()(conv14)
	activ14 = Activation(activ)(bn14)
	MP14 = MaxPooling1D(pool_length=2)(activ14)

	conv15 = Convolution1D(512,1,border_mode='same',init=init)(MP14)
	bn15 = BatchNormalization()(conv15)
	activ15 = Activation(activ)(bn15)
	dropout1 = Dropout(0.5)(activ15)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model8192frames_power2_input1486_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv11 = Convolution1D(256,2,border_mode='same',init=init)(MP10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	MP11 = MaxPooling1D(pool_length=2)(activ11)

	conv13 = Convolution1D(512,2,border_mode='same',init=init)(MP11)
	bn13 = BatchNormalization()(conv13)
	activ13 = Activation(activ)(bn13)
	MP13 = MaxPooling1D(pool_length=2)(activ13)

	conv14 = Convolution1D(512,2,border_mode='same',init=init)(MP13)
	bn14 = BatchNormalization()(conv14)
	activ14 = Activation(activ)(bn14)
	MP14 = MaxPooling1D(pool_length=2)(activ14)

	conv15 = Convolution1D(512,1,border_mode='same',init=init)(MP14)
	bn15 = BatchNormalization()(conv15)
	activ15 = Activation(activ)(bn15)
	dropout1 = Dropout(0.5)(activ15)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model16384frames_power2_input1486_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv11 = Convolution1D(256,2,border_mode='same',init=init)(MP10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	MP11 = MaxPooling1D(pool_length=2)(activ11)

	conv12 = Convolution1D(256,2,border_mode='same',init=init)(MP11)
	bn12 = BatchNormalization()(conv12)
	activ12 = Activation(activ)(bn12)
	MP12 = MaxPooling1D(pool_length=2)(activ12)

	conv13 = Convolution1D(512,2,border_mode='same',init=init)(MP12)
	bn13 = BatchNormalization()(conv13)
	activ13 = Activation(activ)(bn13)
	MP13 = MaxPooling1D(pool_length=2)(activ13)

	conv14 = Convolution1D(512,2,border_mode='same',init=init)(MP13)
	bn14 = BatchNormalization()(conv14)
	activ14 = Activation(activ)(bn14)
	MP14 = MaxPooling1D(pool_length=2)(activ14)

	conv15 = Convolution1D(512,1,border_mode='same',init=init)(MP14)
	bn15 = BatchNormalization()(conv15)
	activ15 = Activation(activ)(bn15)
	dropout1 = Dropout(0.5)(activ15)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model243frames_power3_ETE_window729(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	zero0 = ZeroPadding1D(padding=244)(pool_input)
	conv0 = Convolution1D(128,729,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(zero0)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(512,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(512,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP5)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model

def build_model729frames_power3_ETE_window243(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	zero0 = ZeroPadding1D(padding=82)(pool_input)
	conv0 = Convolution1D(128,243,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(zero0)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(512,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(512,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP6)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model19683frames_power3_ETE_256(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=3)(activ7)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=3)(activ8)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=3)(activ9)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model19683frames_power3_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=3)(activ7)

	conv8 = Convolution1D(512,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=3)(activ8)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=3)(activ9)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model

def build_model19683frames_power3_ETE_TripleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(128,3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	conv1_2 = Convolution1D(128,3,border_mode='same',init=init)(activ1_1)
	bn1_2 = BatchNormalization()(conv1_2)
	activ1_2 = Activation(activ)(bn1_2)
	MP1 = MaxPooling1D(pool_length=3)(activ1_2)

	conv3 = Convolution1D(128,3,border_mode='same',init=init)(MP1)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(128,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	conv3_2 = Convolution1D(128,3,border_mode='same',init=init)(activ3_1)
	bn3_2 = BatchNormalization()(conv3_2)
	activ3_2 = Activation(activ)(bn3_2)
	MP3 = MaxPooling1D(pool_length=3)(activ3_2)

	conv4 = Convolution1D(256,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(256,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	conv4_2 = Convolution1D(256,3,border_mode='same',init=init)(activ4_1)
	bn4_2 = BatchNormalization()(conv4_2)
	activ4_2 = Activation(activ)(bn4_2)
	MP4 = MaxPooling1D(pool_length=3)(activ4_2)

	conv5 = Convolution1D(256,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(256,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	conv5_2 = Convolution1D(256,3,border_mode='same',init=init)(activ5_1)
	bn5_2 = BatchNormalization()(conv5_2)
	activ5_2 = Activation(activ)(bn5_2)
	MP5 = MaxPooling1D(pool_length=3)(activ5_2)

	conv6 = Convolution1D(256,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(256,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	conv6_2 = Convolution1D(256,3,border_mode='same',init=init)(activ6_1)
	bn6_2 = BatchNormalization()(conv6_2)
	activ6_2 = Activation(activ)(bn6_2)
	MP6 = MaxPooling1D(pool_length=3)(activ6_2)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	conv7_2 = Convolution1D(256,3,border_mode='same',init=init)(activ7_1)
	bn7_2 = BatchNormalization()(conv7_2)
	activ7_2 = Activation(activ)(bn7_2)
	MP7 = MaxPooling1D(pool_length=3)(activ7_2)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	conv8_2 = Convolution1D(256,3,border_mode='same',init=init)(activ8_1)
	bn8_2 = BatchNormalization()(conv8_2)
	activ8_2 = Activation(activ)(bn8_2)
	MP8 = MaxPooling1D(pool_length=3)(activ8_2)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	conv9_2 = Convolution1D(512,3,border_mode='same',init=init)(activ9_1)
	bn9_2 = BatchNormalization()(conv9_2)
	activ9_2 = Activation(activ)(bn9_2)
	MP9 = MaxPooling1D(pool_length=3)(activ9_2)

	conv10 = Convolution1D(512,3,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	conv10_1 = Convolution1D(512,3,border_mode='same',init=init)(activ10)
	bn10_1 = BatchNormalization()(conv10_1)
	activ10_1 = Activation(activ)(bn10_1)
	conv10_2 = Convolution1D(512,3,border_mode='same',init=init)(activ10_1)
	bn10_2 = BatchNormalization()(conv10_2)
	activ10_2 = Activation(activ)(bn10_2)
	MP10 = MaxPooling1D(pool_length=3)(activ10_2)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP10)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model19683frames_power3_ETE_DoubleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(128,3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	MP1 = MaxPooling1D(pool_length=3)(activ1_1)

	conv3 = Convolution1D(128,3,border_mode='same',init=init)(MP1)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(128,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	MP3 = MaxPooling1D(pool_length=3)(activ3_1)

	conv4 = Convolution1D(256,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(256,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	MP4 = MaxPooling1D(pool_length=3)(activ4_1)

	conv5 = Convolution1D(256,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(256,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	MP5 = MaxPooling1D(pool_length=3)(activ5_1)

	conv6 = Convolution1D(256,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(256,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	MP6 = MaxPooling1D(pool_length=3)(activ6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	MP7 = MaxPooling1D(pool_length=3)(activ7_1)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	MP8 = MaxPooling1D(pool_length=3)(activ8_1)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	MP9 = MaxPooling1D(pool_length=3)(activ9_1)

	conv10 = Convolution1D(512,3,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	conv10_1 = Convolution1D(512,3,border_mode='same',init=init)(activ10)
	bn10_1 = BatchNormalization()(conv10_1)
	activ10_1 = Activation(activ)(bn10_1)
	MP10 = MaxPooling1D(pool_length=3)(activ10_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP10)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model19683frames_power3_ETE_TripleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(128,3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	conv1_2 = Convolution1D(128,3,border_mode='same',init=init)(activ1_1)
	bn1_2 = BatchNormalization()(conv1_2)
	activ1_2 = Activation(activ)(bn1_2)
	MP1 = MaxPooling1D(pool_length=3)(activ1_2)

	conv2 = Convolution1D(128,3,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	conv2_1 = Convolution1D(128,3,border_mode='same',init=init)(activ2)
	bn2_1 = BatchNormalization()(conv2_1)
	activ2_1 = Activation(activ)(bn2_1)
	conv2_2 = Convolution1D(128,3,border_mode='same',init=init)(activ2_1)
	bn2_2 = BatchNormalization()(conv2_2)
	activ2_2 = Activation(activ)(bn2_2)
	MP2 = MaxPooling1D(pool_length=3)(activ2_2)

	conv3 = Convolution1D(256,3,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(256,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	conv3_2 = Convolution1D(256,3,border_mode='same',init=init)(activ3_1)
	bn3_2 = BatchNormalization()(conv3_2)
	activ3_2 = Activation(activ)(bn3_2)
	MP3 = MaxPooling1D(pool_length=3)(activ3_2)

	conv4 = Convolution1D(256,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(256,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	conv4_2 = Convolution1D(256,3,border_mode='same',init=init)(activ4_1)
	bn4_2 = BatchNormalization()(conv4_2)
	activ4_2 = Activation(activ)(bn4_2)
	MP4 = MaxPooling1D(pool_length=3)(activ4_2)

	conv5 = Convolution1D(256,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(256,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	conv5_2 = Convolution1D(256,3,border_mode='same',init=init)(activ5_1)
	bn5_2 = BatchNormalization()(conv5_2)
	activ5_2 = Activation(activ)(bn5_2)
	MP5 = MaxPooling1D(pool_length=3)(activ5_2)

	conv6 = Convolution1D(256,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(256,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	conv6_2 = Convolution1D(256,3,border_mode='same',init=init)(activ6_1)
	bn6_2 = BatchNormalization()(conv6_2)
	activ6_2 = Activation(activ)(bn6_2)
	MP6 = MaxPooling1D(pool_length=3)(activ6_2)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	conv7_2 = Convolution1D(256,3,border_mode='same',init=init)(activ7_1)
	bn7_2 = BatchNormalization()(conv7_2)
	activ7_2 = Activation(activ)(bn7_2)
	MP7 = MaxPooling1D(pool_length=3)(activ7_2)

	conv8 = Convolution1D(512,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(512,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	conv8_2 = Convolution1D(512,3,border_mode='same',init=init)(activ8_1)
	bn8_2 = BatchNormalization()(conv8_2)
	activ8_2 = Activation(activ)(bn8_2)
	MP8 = MaxPooling1D(pool_length=3)(activ8_2)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	conv9_2 = Convolution1D(512,3,border_mode='same',init=init)(activ9_1)
	bn9_2 = BatchNormalization()(conv9_2)
	activ9_2 = Activation(activ)(bn9_2)
	MP9 = MaxPooling1D(pool_length=3)(activ9_2)

	conv11 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	dropout1 = Dropout(0.5)(activ11)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model125frames_power5_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,5,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=5)(activ1)

	conv3 = Convolution1D(256,5,border_mode='same',init=init)(MP1)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=5)(activ3)

	conv5 = Convolution1D(512,5,border_mode='same',init=init)(MP3)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=5)(activ5)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP5)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model625frames_power5_input3543_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,5,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=5)(activ1)

	conv2 = Convolution1D(128,5,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=5)(activ2)

	conv3 = Convolution1D(256,5,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=5)(activ3)

	conv5 = Convolution1D(512,5,border_mode='same',init=init)(MP3)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=5)(activ5)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP5)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model3125frames_power5_input3543_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,5,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=5)(activ1)

	conv2 = Convolution1D(128,5,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=5)(activ2)

	conv3 = Convolution1D(256,5,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=5)(activ3)

	conv4 = Convolution1D(256,5,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=5)(activ4)

	conv5 = Convolution1D(512,5,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=5)(activ5)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP5)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model

def build_model31250frames_power5_2_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,5,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=5)(activ1)

	conv2 = Convolution1D(128,5,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=5)(activ2)

	conv3 = Convolution1D(256,5,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=5)(activ3)

	conv4 = Convolution1D(256,5,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=5)(activ4)

	conv5 = Convolution1D(256,5,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=5)(activ5)

	conv6 = Convolution1D(512,5,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=5)(activ6)

	conv7 = Convolution1D(512,2,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP7)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model

def build_model62500frames_power5_4_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,5,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=5)(activ1)

	conv2 = Convolution1D(128,5,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=5)(activ2)

	conv3 = Convolution1D(256,5,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=5)(activ3)

	conv4 = Convolution1D(256,5,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=5)(activ4)

	conv5 = Convolution1D(256,5,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=5)(activ5)

	conv6 = Convolution1D(512,5,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=5)(activ6)

	conv7 = Convolution1D(512,4,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=4)(activ7)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP7)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model125000frames_power5_42_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,5,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=5)(activ1)

	conv2 = Convolution1D(128,5,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=5)(activ2)

	conv3 = Convolution1D(256,5,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=5)(activ3)

	conv4 = Convolution1D(256,5,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=5)(activ4)

	conv5 = Convolution1D(256,5,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=5)(activ5)

	conv6 = Convolution1D(256,5,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=5)(activ6)

	conv7 = Convolution1D(512,4,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=4)(activ7)

	conv8 = Convolution1D(512,2,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP8)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model15625frames_power5_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,5,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=5)(activ1)

	conv2 = Convolution1D(128,5,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=5)(activ2)

	conv3 = Convolution1D(256,5,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=5)(activ3)

	conv4 = Convolution1D(256,5,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=5)(activ4)

	conv5 = Convolution1D(256,5,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=5)(activ5)

	conv6 = Convolution1D(512,5,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=5)(activ6)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP6)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model

def build_model15625frames_power5_ETE_doubleFilter(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,5,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(128,5,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	MP1 = MaxPooling1D(pool_length=5)(activ1_1)

	conv2 = Convolution1D(128,5,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	conv2_1 = Convolution1D(128,5,border_mode='same',init=init)(activ2)
	bn2_1 = BatchNormalization()(conv2_1)
	activ2_1 = Activation(activ)(bn2_1)
	MP2 = MaxPooling1D(pool_length=5)(activ2_1)

	conv3 = Convolution1D(256,5,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(256,5,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	MP3 = MaxPooling1D(pool_length=5)(activ3_1)

	conv4 = Convolution1D(256,5,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(256,5,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	MP4 = MaxPooling1D(pool_length=5)(activ4_1)

	conv5 = Convolution1D(256,5,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(256,5,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	MP5 = MaxPooling1D(pool_length=5)(activ5_1)

	conv6 = Convolution1D(512,5,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(512,5,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	MP6 = MaxPooling1D(pool_length=5)(activ6_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP6)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model64frames_power4_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,4,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=4)(activ1)

	conv3 = Convolution1D(256,4,border_mode='same',init=init)(MP1)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=4)(activ3)

	conv6 = Convolution1D(512,4,border_mode='same',init=init)(MP3)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=4)(activ6)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP6)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model256frames_power4_input2972_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,4,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=4)(activ1)

	conv3 = Convolution1D(256,4,border_mode='same',init=init)(MP1)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=4)(activ3)

	conv5 = Convolution1D(256,4,border_mode='same',init=init)(MP3)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=4)(activ5)

	conv6 = Convolution1D(512,4,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=4)(activ6)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP6)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model1024frames_power4_input2972_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,4,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=4)(activ1)

	conv2 = Convolution1D(128,4,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=4)(activ2)

	conv3 = Convolution1D(256,4,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=4)(activ3)

	conv5 = Convolution1D(256,4,border_mode='same',init=init)(MP3)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=4)(activ5)

	conv6 = Convolution1D(512,4,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=4)(activ6)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP6)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model4096frames_power4_input2972_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,4,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=4)(activ1)

	conv2 = Convolution1D(128,4,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=4)(activ2)

	conv3 = Convolution1D(256,4,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=4)(activ3)

	conv4 = Convolution1D(256,4,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=4)(activ4)

	conv5 = Convolution1D(256,4,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=4)(activ5)

	conv6 = Convolution1D(512,4,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=4)(activ6)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP6)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model

def build_model16384frames_power4_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,4,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=4)(activ1)

	conv2 = Convolution1D(128,4,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=4)(activ2)

	conv3 = Convolution1D(256,4,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=4)(activ3)

	conv4 = Convolution1D(256,4,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=4)(activ4)

	conv5 = Convolution1D(256,4,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=4)(activ5)

	conv6 = Convolution1D(256,4,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=4)(activ6)

	conv7 = Convolution1D(512,4,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=4)(activ7)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP7)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model6561frames_3s_ETE_FilterDoubleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(16,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(32,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(32,3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	MP1 = MaxPooling1D(pool_length=3)(activ1_1)

	conv2 = Convolution1D(32,3,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	conv2_1 = Convolution1D(32,3,border_mode='same',init=init)(activ2)
	bn2_1 = BatchNormalization()(conv2_1)
	activ2_1 = Activation(activ)(bn2_1)
	MP2 = MaxPooling1D(pool_length=3)(activ2_1)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	MP3 = MaxPooling1D(pool_length=3)(activ3_1)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	MP4 = MaxPooling1D(pool_length=3)(activ4_1)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	MP5 = MaxPooling1D(pool_length=3)(activ5_1)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	MP6 = MaxPooling1D(pool_length=3)(activ6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	MP7 = MaxPooling1D(pool_length=3)(activ7_1)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	MP8 = MaxPooling1D(pool_length=3)(activ8_1)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	MP9 = MaxPooling1D(pool_length=3)(activ9_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model



def build_model19683frames_3s_ETE_FilterDoubleMoreLayerOnTail(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(16,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(32,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(32,3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	MP1 = MaxPooling1D(pool_length=3)(activ1_1)

	conv2 = Convolution1D(32,3,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	conv2_1 = Convolution1D(32,3,border_mode='same',init=init)(activ2)
	bn2_1 = BatchNormalization()(conv2_1)
	activ2_1 = Activation(activ)(bn2_1)
	MP2 = MaxPooling1D(pool_length=3)(activ2_1)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	MP3 = MaxPooling1D(pool_length=3)(activ3_1)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	MP4 = MaxPooling1D(pool_length=3)(activ4_1)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	MP5 = MaxPooling1D(pool_length=3)(activ5_1)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	MP6 = MaxPooling1D(pool_length=3)(activ6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	conv7_2 = Convolution1D(256,3,border_mode='same',init=init)(activ7_1)
	bn7_2 = BatchNormalization()(conv7_2)
	activ7_2 = Activation(activ)(bn7_2)
	MP7 = MaxPooling1D(pool_length=3)(activ7_2)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	conv8_2 = Convolution1D(256,3,border_mode='same',init=init)(activ8_1)
	bn8_2 = BatchNormalization()(conv8_2)
	activ8_2 = Activation(activ)(bn8_2)
	MP8 = MaxPooling1D(pool_length=3)(activ8_2)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	conv9_2 = Convolution1D(512,3,border_mode='same',init=init)(activ9_1)
	bn9_2 = BatchNormalization()(conv9_2)
	activ9_2 = Activation(activ)(bn9_2)
	MP9 = MaxPooling1D(pool_length=3)(activ9_2)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model19683frames_3s_ETE_FilterDoubleMoreLayerOnHead(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(16,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(32,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(32,3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	conv1_2 = Convolution1D(32,3,border_mode='same',init=init)(activ1_1)
	bn1_2 = BatchNormalization()(conv1_2)
	activ1_2 = Activation(activ)(bn1_2)
	MP1 = MaxPooling1D(pool_length=3)(activ1_2)

	conv2 = Convolution1D(32,3,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	conv2_1 = Convolution1D(32,3,border_mode='same',init=init)(activ2)
	bn2_1 = BatchNormalization()(conv2_1)
	activ2_1 = Activation(activ)(bn2_1)
	conv2_2 = Convolution1D(32,3,border_mode='same',init=init)(activ2_1)
	bn2_2 = BatchNormalization()(conv2_2)
	activ2_2 = Activation(activ)(bn2_2)
	MP2 = MaxPooling1D(pool_length=3)(activ2_2)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	conv3_2 = Convolution1D(64,3,border_mode='same',init=init)(activ3_1)
	bn3_2 = BatchNormalization()(conv3_2)
	activ3_2 = Activation(activ)(bn3_2)
	MP3 = MaxPooling1D(pool_length=3)(activ3_2)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	MP4 = MaxPooling1D(pool_length=3)(activ4_1)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	MP5 = MaxPooling1D(pool_length=3)(activ5_1)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	MP6 = MaxPooling1D(pool_length=3)(activ6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	MP7 = MaxPooling1D(pool_length=3)(activ7_1)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	MP8 = MaxPooling1D(pool_length=3)(activ8_1)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	MP9 = MaxPooling1D(pool_length=3)(activ9_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model243frames_3s_ETE_FilterTripleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(32,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(activ0)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	conv3_2 = Convolution1D(64,3,border_mode='same',init=init)(activ3_1)
	bn3_2 = BatchNormalization()(conv3_2)
	activ3_2 = Activation(activ)(bn3_2)
	MP3 = MaxPooling1D(pool_length=3)(activ3_2)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP3)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	conv6_2 = Convolution1D(128,3,border_mode='same',init=init)(activ6_1)
	bn6_2 = BatchNormalization()(conv6_2)
	activ6_2 = Activation(activ)(bn6_2)
	MP6 = MaxPooling1D(pool_length=3)(activ6_2)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	conv7_2 = Convolution1D(256,3,border_mode='same',init=init)(activ7_1)
	bn7_2 = BatchNormalization()(conv7_2)
	activ7_2 = Activation(activ)(bn7_2)
	MP7 = MaxPooling1D(pool_length=3)(activ7_2)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	conv8_2 = Convolution1D(256,3,border_mode='same',init=init)(activ8_1)
	bn8_2 = BatchNormalization()(conv8_2)
	activ8_2 = Activation(activ)(bn8_2)
	MP8 = MaxPooling1D(pool_length=3)(activ8_2)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	conv9_2 = Convolution1D(512,3,border_mode='same',init=init)(activ9_1)
	bn9_2 = BatchNormalization()(conv9_2)
	activ9_2 = Activation(activ)(bn9_2)
	MP9 = MaxPooling1D(pool_length=3)(activ9_2)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model729frames_3s_ETE_FilterTripleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(32,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(activ0)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	conv3_2 = Convolution1D(64,3,border_mode='same',init=init)(activ3_1)
	bn3_2 = BatchNormalization()(conv3_2)
	activ3_2 = Activation(activ)(bn3_2)
	MP3 = MaxPooling1D(pool_length=3)(activ3_2)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP3)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	conv5_2 = Convolution1D(128,3,border_mode='same',init=init)(activ5_1)
	bn5_2 = BatchNormalization()(conv5_2)
	activ5_2 = Activation(activ)(bn5_2)
	MP5 = MaxPooling1D(pool_length=3)(activ5_2)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	conv6_2 = Convolution1D(128,3,border_mode='same',init=init)(activ6_1)
	bn6_2 = BatchNormalization()(conv6_2)
	activ6_2 = Activation(activ)(bn6_2)
	MP6 = MaxPooling1D(pool_length=3)(activ6_2)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	conv7_2 = Convolution1D(256,3,border_mode='same',init=init)(activ7_1)
	bn7_2 = BatchNormalization()(conv7_2)
	activ7_2 = Activation(activ)(bn7_2)
	MP7 = MaxPooling1D(pool_length=3)(activ7_2)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	conv8_2 = Convolution1D(256,3,border_mode='same',init=init)(activ8_1)
	bn8_2 = BatchNormalization()(conv8_2)
	activ8_2 = Activation(activ)(bn8_2)
	MP8 = MaxPooling1D(pool_length=3)(activ8_2)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	conv9_2 = Convolution1D(512,3,border_mode='same',init=init)(activ9_1)
	bn9_2 = BatchNormalization()(conv9_2)
	activ9_2 = Activation(activ)(bn9_2)
	MP9 = MaxPooling1D(pool_length=3)(activ9_2)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model2187frames_3s_ETE_FilterTripleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(32,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(activ0)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	conv3_2 = Convolution1D(64,3,border_mode='same',init=init)(activ3_1)
	bn3_2 = BatchNormalization()(conv3_2)
	activ3_2 = Activation(activ)(bn3_2)
	MP3 = MaxPooling1D(pool_length=3)(activ3_2)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	conv4_2 = Convolution1D(64,3,border_mode='same',init=init)(activ4_1)
	bn4_2 = BatchNormalization()(conv4_2)
	activ4_2 = Activation(activ)(bn4_2)
	MP4 = MaxPooling1D(pool_length=3)(activ4_2)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	conv5_2 = Convolution1D(128,3,border_mode='same',init=init)(activ5_1)
	bn5_2 = BatchNormalization()(conv5_2)
	activ5_2 = Activation(activ)(bn5_2)
	MP5 = MaxPooling1D(pool_length=3)(activ5_2)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	conv6_2 = Convolution1D(128,3,border_mode='same',init=init)(activ6_1)
	bn6_2 = BatchNormalization()(conv6_2)
	activ6_2 = Activation(activ)(bn6_2)
	MP6 = MaxPooling1D(pool_length=3)(activ6_2)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	conv7_2 = Convolution1D(256,3,border_mode='same',init=init)(activ7_1)
	bn7_2 = BatchNormalization()(conv7_2)
	activ7_2 = Activation(activ)(bn7_2)
	MP7 = MaxPooling1D(pool_length=3)(activ7_2)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	conv8_2 = Convolution1D(256,3,border_mode='same',init=init)(activ8_1)
	bn8_2 = BatchNormalization()(conv8_2)
	activ8_2 = Activation(activ)(bn8_2)
	MP8 = MaxPooling1D(pool_length=3)(activ8_2)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	conv9_2 = Convolution1D(512,3,border_mode='same',init=init)(activ9_1)
	bn9_2 = BatchNormalization()(conv9_2)
	activ9_2 = Activation(activ)(bn9_2)
	MP9 = MaxPooling1D(pool_length=3)(activ9_2)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model6561frames_3s_ETE_FilterTripleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(32,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(32,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(32,3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	conv1_2 = Convolution1D(32,3,border_mode='same',init=init)(activ1_1)
	bn1_2 = BatchNormalization()(conv1_2)
	activ1_2 = Activation(activ)(bn1_2)
	MP1 = MaxPooling1D(pool_length=3)(activ1_2)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(MP1)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	conv3_2 = Convolution1D(64,3,border_mode='same',init=init)(activ3_1)
	bn3_2 = BatchNormalization()(conv3_2)
	activ3_2 = Activation(activ)(bn3_2)
	MP3 = MaxPooling1D(pool_length=3)(activ3_2)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	conv4_2 = Convolution1D(64,3,border_mode='same',init=init)(activ4_1)
	bn4_2 = BatchNormalization()(conv4_2)
	activ4_2 = Activation(activ)(bn4_2)
	MP4 = MaxPooling1D(pool_length=3)(activ4_2)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	conv5_2 = Convolution1D(128,3,border_mode='same',init=init)(activ5_1)
	bn5_2 = BatchNormalization()(conv5_2)
	activ5_2 = Activation(activ)(bn5_2)
	MP5 = MaxPooling1D(pool_length=3)(activ5_2)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	conv6_2 = Convolution1D(128,3,border_mode='same',init=init)(activ6_1)
	bn6_2 = BatchNormalization()(conv6_2)
	activ6_2 = Activation(activ)(bn6_2)
	MP6 = MaxPooling1D(pool_length=3)(activ6_2)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	conv7_2 = Convolution1D(256,3,border_mode='same',init=init)(activ7_1)
	bn7_2 = BatchNormalization()(conv7_2)
	activ7_2 = Activation(activ)(bn7_2)
	MP7 = MaxPooling1D(pool_length=3)(activ7_2)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	conv8_2 = Convolution1D(256,3,border_mode='same',init=init)(activ8_1)
	bn8_2 = BatchNormalization()(conv8_2)
	activ8_2 = Activation(activ)(bn8_2)
	MP8 = MaxPooling1D(pool_length=3)(activ8_2)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	conv9_2 = Convolution1D(512,3,border_mode='same',init=init)(activ9_1)
	bn9_2 = BatchNormalization()(conv9_2)
	activ9_2 = Activation(activ)(bn9_2)
	MP9 = MaxPooling1D(pool_length=3)(activ9_2)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model19683frames_3s_ETE_FilterDoubleLayerLessFirstFilter(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(8,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(32,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(32,3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	MP1 = MaxPooling1D(pool_length=3)(activ1_1)

	conv2 = Convolution1D(32,3,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	conv2_1 = Convolution1D(32,3,border_mode='same',init=init)(activ2)
	bn2_1 = BatchNormalization()(conv2_1)
	activ2_1 = Activation(activ)(bn2_1)
	MP2 = MaxPooling1D(pool_length=3)(activ2_1)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	MP3 = MaxPooling1D(pool_length=3)(activ3_1)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	MP4 = MaxPooling1D(pool_length=3)(activ4_1)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	MP5 = MaxPooling1D(pool_length=3)(activ5_1)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	MP6 = MaxPooling1D(pool_length=3)(activ6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	MP7 = MaxPooling1D(pool_length=3)(activ7_1)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	MP8 = MaxPooling1D(pool_length=3)(activ8_1)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	MP9 = MaxPooling1D(pool_length=3)(activ9_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model19683frames_3s_ETE_FilterDoubleLayerStride(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(16,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(32,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(32,3,subsample_length=3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)

	conv2 = Convolution1D(32,3,border_mode='same',init=init)(activ1_1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	conv2_1 = Convolution1D(32,3,subsample_length=3,border_mode='same',init=init)(activ2)
	bn2_1 = BatchNormalization()(conv2_1)
	activ2_1 = Activation(activ)(bn2_1)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(activ2_1)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,subsample_length=3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(activ3_1)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,subsample_length=3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(activ4_1)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,subsample_length=3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(activ5_1)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,subsample_length=3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(activ6_1)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,subsample_length=3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(activ7_1)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,subsample_length=3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(activ8_1)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,subsample_length=3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(activ9_1)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model



def build_model243frames_3s_ETE_FilterDoubleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(32,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(activ0)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	MP4 = MaxPooling1D(pool_length=3)(activ4_1)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	MP6 = MaxPooling1D(pool_length=3)(activ6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	MP7 = MaxPooling1D(pool_length=3)(activ7_1)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	MP8 = MaxPooling1D(pool_length=3)(activ8_1)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	MP9 = MaxPooling1D(pool_length=3)(activ9_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model

def build_model729frames_3s_ETE_FilterDoubleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(32,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(activ0)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	MP4 = MaxPooling1D(pool_length=3)(activ4_1)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	MP5 = MaxPooling1D(pool_length=3)(activ5_1)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	MP6 = MaxPooling1D(pool_length=3)(activ6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	MP7 = MaxPooling1D(pool_length=3)(activ7_1)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	MP8 = MaxPooling1D(pool_length=3)(activ8_1)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	MP9 = MaxPooling1D(pool_length=3)(activ9_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model2187frames_3s_ETE_FilterDoubleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(32,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(activ0)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	MP3 = MaxPooling1D(pool_length=3)(activ3_1)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	MP4 = MaxPooling1D(pool_length=3)(activ4_1)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	MP5 = MaxPooling1D(pool_length=3)(activ5_1)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	MP6 = MaxPooling1D(pool_length=3)(activ6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	MP7 = MaxPooling1D(pool_length=3)(activ7_1)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	MP8 = MaxPooling1D(pool_length=3)(activ8_1)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	MP9 = MaxPooling1D(pool_length=3)(activ9_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model6561frames_3s_ETE_FilterDoubleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(32,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(32,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(32,3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	MP1 = MaxPooling1D(pool_length=3)(activ1_1)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(MP1)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	MP3 = MaxPooling1D(pool_length=3)(activ3_1)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	MP4 = MaxPooling1D(pool_length=3)(activ4_1)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	MP5 = MaxPooling1D(pool_length=3)(activ5_1)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	MP6 = MaxPooling1D(pool_length=3)(activ6_1)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	MP7 = MaxPooling1D(pool_length=3)(activ7_1)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	MP8 = MaxPooling1D(pool_length=3)(activ8_1)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	MP9 = MaxPooling1D(pool_length=3)(activ9_1)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model19683frames_3s_ETE_FilterTripleLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(16,conv_window_size,subsample_length=hop_size,border_mode='valid',init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(32,3,border_mode='same',init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	conv1_1 = Convolution1D(32,3,border_mode='same',init=init)(activ1)
	bn1_1 = BatchNormalization()(conv1_1)
	activ1_1 = Activation(activ)(bn1_1)
	conv1_2 = Convolution1D(32,3,border_mode='same',init=init)(activ1_1)
	bn1_2 = BatchNormalization()(conv1_2)
	activ1_2 = Activation(activ)(bn1_2)
	MP1 = MaxPooling1D(pool_length=3)(activ1_2)

	conv2 = Convolution1D(32,3,border_mode='same',init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	conv2_1 = Convolution1D(32,3,border_mode='same',init=init)(activ2)
	bn2_1 = BatchNormalization()(conv2_1)
	activ2_1 = Activation(activ)(bn2_1)
	conv2_2 = Convolution1D(32,3,border_mode='same',init=init)(activ2_1)
	bn2_2 = BatchNormalization()(conv2_2)
	activ2_2 = Activation(activ)(bn2_2)
	MP2 = MaxPooling1D(pool_length=3)(activ2_2)

	conv3 = Convolution1D(64,3,border_mode='same',init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	conv3_1 = Convolution1D(64,3,border_mode='same',init=init)(activ3)
	bn3_1 = BatchNormalization()(conv3_1)
	activ3_1 = Activation(activ)(bn3_1)
	conv3_2 = Convolution1D(64,3,border_mode='same',init=init)(activ3_1)
	bn3_2 = BatchNormalization()(conv3_2)
	activ3_2 = Activation(activ)(bn3_2)
	MP3 = MaxPooling1D(pool_length=3)(activ3_2)

	conv4 = Convolution1D(64,3,border_mode='same',init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	conv4_1 = Convolution1D(64,3,border_mode='same',init=init)(activ4)
	bn4_1 = BatchNormalization()(conv4_1)
	activ4_1 = Activation(activ)(bn4_1)
	conv4_2 = Convolution1D(64,3,border_mode='same',init=init)(activ4_1)
	bn4_2 = BatchNormalization()(conv4_2)
	activ4_2 = Activation(activ)(bn4_2)
	MP4 = MaxPooling1D(pool_length=3)(activ4_2)

	conv5 = Convolution1D(128,3,border_mode='same',init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	conv5_1 = Convolution1D(128,3,border_mode='same',init=init)(activ5)
	bn5_1 = BatchNormalization()(conv5_1)
	activ5_1 = Activation(activ)(bn5_1)
	conv5_2 = Convolution1D(128,3,border_mode='same',init=init)(activ5_1)
	bn5_2 = BatchNormalization()(conv5_2)
	activ5_2 = Activation(activ)(bn5_2)
	MP5 = MaxPooling1D(pool_length=3)(activ5_2)

	conv6 = Convolution1D(128,3,border_mode='same',init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	conv6_1 = Convolution1D(128,3,border_mode='same',init=init)(activ6)
	bn6_1 = BatchNormalization()(conv6_1)
	activ6_1 = Activation(activ)(bn6_1)
	conv6_2 = Convolution1D(128,3,border_mode='same',init=init)(activ6_1)
	bn6_2 = BatchNormalization()(conv6_2)
	activ6_2 = Activation(activ)(bn6_2)
	MP6 = MaxPooling1D(pool_length=3)(activ6_2)

	conv7 = Convolution1D(256,3,border_mode='same',init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	conv7_1 = Convolution1D(256,3,border_mode='same',init=init)(activ7)
	bn7_1 = BatchNormalization()(conv7_1)
	activ7_1 = Activation(activ)(bn7_1)
	conv7_2 = Convolution1D(256,3,border_mode='same',init=init)(activ7_1)
	bn7_2 = BatchNormalization()(conv7_2)
	activ7_2 = Activation(activ)(bn7_2)
	MP7 = MaxPooling1D(pool_length=3)(activ7_2)

	conv8 = Convolution1D(256,3,border_mode='same',init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	conv8_1 = Convolution1D(256,3,border_mode='same',init=init)(activ8)
	bn8_1 = BatchNormalization()(conv8_1)
	activ8_1 = Activation(activ)(bn8_1)
	conv8_2 = Convolution1D(256,3,border_mode='same',init=init)(activ8_1)
	bn8_2 = BatchNormalization()(conv8_2)
	activ8_2 = Activation(activ)(bn8_2)
	MP8 = MaxPooling1D(pool_length=3)(activ8_2)

	conv9 = Convolution1D(512,3,border_mode='same',init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	conv9_1 = Convolution1D(512,3,border_mode='same',init=init)(activ9)
	bn9_1 = BatchNormalization()(conv9_1)
	activ9_1 = Activation(activ)(bn9_1)
	conv9_2 = Convolution1D(512,3,border_mode='same',init=init)(activ9_1)
	bn9_2 = BatchNormalization()(conv9_2)
	activ9_2 = Activation(activ)(bn9_2)
	MP9 = MaxPooling1D(pool_length=3)(activ9_2)

	conv10 = Convolution1D(512,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model6561frames_3s_nopooling9999_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,subsample_length=9,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)

	conv2 = Convolution1D(128,3,subsample_length=9,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)

	conv3 = Convolution1D(256,3,subsample_length=9,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)

	conv4 = Convolution1D(256,3,subsample_length=9,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)

	conv10 = Convolution1D(256,1,border_mode='same',init=init)(activ4)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


		
def build_model6561frames_3s_nopooling33333333_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)

	conv2 = Convolution1D(128,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)

	conv3 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)

	conv4 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)

	conv5 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)

	conv6 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)

	conv7 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)

	conv8 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)

	conv10 = Convolution1D(256,1,border_mode='same',init=init)(activ8)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


	
def build_model6561frames_3s_3x3MP_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	zero0 = ZeroPadding1D(padding=4)(pool_input)	
	conv0 = Convolution1D(128,conv_window_size,subsample_length=3,border_mode='valid',W_regularizer=l2(1e-7),init=init)(zero0)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)
	MP0 = MaxPooling1D(pool_length=3)(activ0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv7 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=3)(activ7)

	conv8 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=3)(activ8)

	conv10 = Convolution1D(256,1,border_mode='same',init=init)(MP8)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model81frames_3s_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv10 = Convolution1D(256,1,border_mode='same',init=init)(MP4)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model



def build_model243frames_3s_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv10 = Convolution1D(256,1,border_mode='same',init=init)(MP5)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model




def build_model729frames_3s_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv10 = Convolution1D(256,1,border_mode='same',init=init)(MP6)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model




def build_model2187frames_3s_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv7 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=3)(activ7)

	conv10 = Convolution1D(256,1,border_mode='same',init=init)(MP7)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model



def build_model6561frames_3s_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv7 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=3)(activ7)

	conv8 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=3)(activ8)

	conv10 = Convolution1D(256,1,border_mode='same',init=init)(MP8)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model19683frames_3s_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv7 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=3)(activ7)

	conv8 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=3)(activ8)

	conv9 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=3)(activ9)

	conv10 = Convolution1D(256,1,border_mode='same',init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	dropout1 = Dropout(0.5)(activ10)

	Flattened = Flatten()(dropout1)

	output = Dense(options.num_tags,activation='sigmoid')(Flattened)
	model = Model(input=pool_input,output=output)

	return model


def build_model27frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	Flattend = Flatten()(MP3)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


	
def build_model81frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	Flattend = Flatten()(MP4)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model243frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	Flattend = Flatten()(MP5)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


	
def build_model729frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	Flattend = Flatten()(MP6)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model

	
def build_model2187frames_ETE_3x3MP(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	zero0 = ZeroPadding1D(padding=4)(pool_input)	
	conv0 = Convolution1D(128,conv_window_size,subsample_length=3,border_mode='valid',W_regularizer=l2(1e-7),init=init)(zero0)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)
	MP0 = MaxPooling1D(pool_length=3)(activ0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv7 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=3)(activ7)

	Flattend = Flatten()(MP7)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model

	
def build_model2187frames_ETE_nopool9993(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,9,subsample_length=9,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)

	conv2 = Convolution1D(128,9,subsample_length=9,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)

	conv3 = Convolution1D(256,9,subsample_length=9,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)

	conv4 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)

	Flattend = Flatten()(activ4)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model2187frames_ETE_nopool3333333(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)

	conv2 = Convolution1D(128,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)

	conv3 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)

	conv4 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)

	conv5 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)

	conv6 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)

	conv7 = Convolution1D(256,3,subsample_length=3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)

	Flattend = Flatten()(activ7)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model



def build_model2187frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv7 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=3)(activ7)

	Flattend = Flatten()(MP7)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


	
def build_model6561frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=3)(activ4)

	conv5 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=3)(activ5)

	conv6 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=3)(activ6)

	conv7 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=3)(activ7)

	conv8 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=3)(activ8)

	Flattend = Flatten()(MP8)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model864frames_ETE_192(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(192,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=regul,init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(192,3,border_mode='same',W_regularizer=regul,init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(192,3,border_mode='same',W_regularizer=regul,init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=regul,init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=regul,init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=regul,init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=regul,init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=regul,init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=regul,init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	Flattend = Flatten()(MP8)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model864frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	Flattend = Flatten()(MP8)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model

def build_model1728frames_ETE_4x2MP(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,4,subsample_length=4,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)
	MP0 = MaxPooling1D(pool_length=2)(activ0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	Flattend = Flatten()(MP9)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model1728frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	Flattend = Flatten()(MP9)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model

def build_model3456frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	Flattend = Flatten()(MP10)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model

def build_model6912frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(128,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,3,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=3)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv11 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	MP11 = MaxPooling1D(pool_length=2)(activ11)

	Flattend = Flatten()(MP11)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model

def build_model64frames_ETE_2layer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,8,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=8)(activ1)

	conv2 = Convolution1D(128,8,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=8)(activ2)

	Flattend = Flatten()(MP2)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


	
def build_model64frames_ETE_3layer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,4,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=4)(activ1)

	conv2 = Convolution1D(128,4,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=4)(activ2)

	conv3 = Convolution1D(256,4,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=4)(activ3)

	Flattend = Flatten()(MP3)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model64frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	Flattend = Flatten()(MP6)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


	
def build_model128frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	Flattend = Flatten()(MP7)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


	
	
def build_model256frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	Flattend = Flatten()(MP8)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


	
def build_model512frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	Flattend = Flatten()(MP9)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model1024frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	Flattend = Flatten()(MP10)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model

	
def build_model1024frames_ETE_nopooling_dilation(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,16,subsample_length=16,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)

	conv2 = Convolution1D(128,8,subsample_length=8,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)

	conv3 = Convolution1D(256,4,subsample_length=4,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)

	conv4 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)

	Flattend = Flatten()(activ4)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


	
def build_model2048frames_ETE_nopooling8s_4(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,8,subsample_length=8,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)

	conv2 = Convolution1D(128,8,subsample_length=8,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)

	conv3 = Convolution1D(256,8,subsample_length=8,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)

	conv4 = Convolution1D(256,4,subsample_length=4,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)

	Flattend = Flatten()(activ4)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model2048frames_ETE_nopooling8(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,8,subsample_length=8,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)

	conv2 = Convolution1D(128,8,subsample_length=8,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)

	conv3 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)

	conv4 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)

	conv5 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)

	conv6 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)

	conv7 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)

	Flattend = Flatten()(activ7)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model2048frames_ETE_nopooling4(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,4,subsample_length=4,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)

	conv2 = Convolution1D(128,4,subsample_length=4,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)

	conv3 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)

	conv4 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)

	conv5 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)

	conv6 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)

	conv7 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)

	conv8 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)

	conv9 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)

	Flattend = Flatten()(activ9)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model2048frames_ETE_nopooling(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)

	conv2 = Convolution1D(128,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)

	conv3 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)

	conv4 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)

	conv5 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)

	conv6 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)

	conv7 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)

	conv8 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)

	conv9 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)

	conv10 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)

	conv11 = Convolution1D(256,2,subsample_length=2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)

	Flattend = Flatten()(activ11)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model2048frames_ETE_4x2MP(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	zero0 = ZeroPadding1D(padding=4)(pool_input)	
	conv0 = Convolution1D(128,conv_window_size,subsample_length=4,border_mode='valid',W_regularizer=l2(1e-7),init=init)(zero0)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)
	MP0 = MaxPooling1D(pool_length=2)(activ0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv11 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	MP11 = MaxPooling1D(pool_length=2)(activ11)

	Flattend = Flatten()(MP11)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model2048frames_ETE_2x4MP(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	zero0 = ZeroPadding1D(padding=4)(pool_input)
	conv0 = Convolution1D(128,conv_window_size,subsample_length=2,border_mode='valid',W_regularizer=l2(1e-7),init=init)(zero0)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)
	MP0 = MaxPooling1D(pool_length=4)(activ0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv11 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	MP11 = MaxPooling1D(pool_length=2)(activ11)

	Flattend = Flatten()(MP11)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model



def build_model2048frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv11 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	MP11 = MaxPooling1D(pool_length=2)(activ11)

	Flattend = Flatten()(MP11)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model4096frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv11 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	MP11 = MaxPooling1D(pool_length=2)(activ11)

	conv12 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP11)
	bn12 = BatchNormalization()(conv12)
	activ12 = Activation(activ)(bn12)
	MP12 = MaxPooling1D(pool_length=2)(activ12)

	Flattend = Flatten()(MP12)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model

def build_model4096frames_ETE_1stLayer(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)
	MP0 = MaxPooling1D(pool_length=2)(activ0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv11 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	MP11 = MaxPooling1D(pool_length=2)(activ11)

	Flattend = Flatten()(MP11)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model


def build_model8192frames_ETE(sample_length,hop_size,window):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(hop_size,window)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=hop_size,border_mode='valid',W_regularizer=l2(1e-7),init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)

	conv1 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(activ0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=2)(activ1)

	conv2 = Convolution1D(128,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=2)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	conv4 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP3)
	bn4 = BatchNormalization()(conv4)
	activ4 = Activation(activ)(bn4)
	MP4 = MaxPooling1D(pool_length=2)(activ4)

	conv5 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP4)
	bn5 = BatchNormalization()(conv5)
	activ5 = Activation(activ)(bn5)
	MP5 = MaxPooling1D(pool_length=2)(activ5)

	conv6 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP5)
	bn6 = BatchNormalization()(conv6)
	activ6 = Activation(activ)(bn6)
	MP6 = MaxPooling1D(pool_length=2)(activ6)

	conv7 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP6)
	bn7 = BatchNormalization()(conv7)
	activ7 = Activation(activ)(bn7)
	MP7 = MaxPooling1D(pool_length=2)(activ7)

	conv8 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP7)
	bn8 = BatchNormalization()(conv8)
	activ8 = Activation(activ)(bn8)
	MP8 = MaxPooling1D(pool_length=2)(activ8)

	conv9 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP8)
	bn9 = BatchNormalization()(conv9)
	activ9 = Activation(activ)(bn9)
	MP9 = MaxPooling1D(pool_length=2)(activ9)

	conv10 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP9)
	bn10 = BatchNormalization()(conv10)
	activ10 = Activation(activ)(bn10)
	MP10 = MaxPooling1D(pool_length=2)(activ10)

	conv11 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP10)
	bn11 = BatchNormalization()(conv11)
	activ11 = Activation(activ)(bn11)
	MP11 = MaxPooling1D(pool_length=2)(activ11)

	conv12 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP11)
	bn12 = BatchNormalization()(conv12)
	activ12 = Activation(activ)(bn12)
	MP12 = MaxPooling1D(pool_length=2)(activ12)

	conv13 = Convolution1D(256,2,border_mode='same',W_regularizer=l2(1e-7),init=init)(MP12)
	bn13 = BatchNormalization()(conv13)
	activ13 = Activation(activ)(bn13)
	MP13 = MaxPooling1D(pool_length=2)(activ13)

	Flattend = Flatten()(MP13)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model



def build_model54frames_ETE_integrated(sample_length):

	pool_input = Input(shape=(sample_length,1))
	conv_window_size = calculate_window_size(options.hop_size,options.conv_window_size)

	conv0 = Convolution1D(128,conv_window_size,subsample_length=options.hop_size,border_mode='valid',W_regularizer=regul,init=init)(pool_input)
	bn0 = BatchNormalization()(conv0)
	activ0 = Activation(activ)(bn0)
	MP0 = MaxPooling1D(pool_length=3)(activ0)

	conv1 = Convolution1D(128,3,border_mode='same',W_regularizer=regul,init=init)(MP0)
	bn1 = BatchNormalization()(conv1)
	activ1 = Activation(activ)(bn1)
	MP1 = MaxPooling1D(pool_length=3)(activ1)

	conv2 = Convolution1D(256,3,border_mode='same',W_regularizer=regul,init=init)(MP1)
	bn2 = BatchNormalization()(conv2)
	activ2 = Activation(activ)(bn2)
	MP2 = MaxPooling1D(pool_length=3)(activ2)

	conv3 = Convolution1D(256,2,border_mode='same',W_regularizer=regul,init=init)(MP2)
	bn3 = BatchNormalization()(conv3)
	activ3 = Activation(activ)(bn3)
	MP3 = MaxPooling1D(pool_length=2)(activ3)

	Flattend = Flatten()(MP3)

	dense1 = Dense(256)(Flattend)
	bnd1 = BatchNormalization()(dense1)
	activd1 = Activation(activ)(bnd1)
	dropout1 = Dropout(0.5)(activd1)

	output = Dense(options.num_tags,activation='sigmoid')(dropout1)
	model = Model(input=pool_input,output=output)

	return model



def generator_train(train_list,y_train_init,num_frame_input,hop_size,window):
	i = 0
	j = 0

	batch_size = options.batch_size
	leng = len(train_list)
	subset_size = int(leng/options.partition)
	# example, total 201680, partition=40, subset_size=5042, batch_size=50
	conv_window_size = calculate_window_size(hop_size,window)
	sample_length = calculate_sample_length(hop_size,num_frame_input)
	num_segment = calculate_num_segment(hop_size,num_frame_input)
	# example, num_frame_input=27, hop_size=512, sample_length=15360, conv_window_size=2048

	while 1:
		# load subset
		x_train_sub = np.zeros((subset_size*num_segment,sample_length,1))
		y_train_sub = np.zeros((subset_size,options.num_tags))

		for iter in range(0,subset_size):
			'''	
			# for debugging
			if iter == 0:
				print '\n'+str(iter*options.partition+i)
			else:
				print iter*options.partition+i,iter,i
			'''

			# load x_train
			tmp = np.load(feature_path + 
					train_list[iter*options.partition+i])
		
			for iter2 in range(0,num_segment):

				x_train_sub[num_segment*iter+iter2,:,0] = tmp[hop_size*(iter2*num_frame_input):hop_size*(iter2*num_frame_input+num_frame_input)]
				
			y_train_sub[iter] = y_train_init[iter*options.partition+i,:]
			
		# duplication
		y_train_sub = np.repeat(y_train_sub,num_segment,axis=0)
		# print 'sub train set loaded!' + str(i)

		# segments randomization
		tmp_train = np.arange(num_segment*subset_size)
		np.random.shuffle(tmp_train)
		x_train_sub = x_train_sub[tmp_train]
		y_train_sub = y_train_sub[tmp_train]

		# segment flatten
		x_train_sub_batch = np.zeros((batch_size,sample_length,1))
		y_train_sub_batch = np.zeros((batch_size,options.num_tags))

		for iter2 in range(0,subset_size*num_segment/batch_size):

			# batch set
			for iter3 in range(0,batch_size):

				x_train_sub_batch[iter3] = x_train_sub[iter3*subset_size*num_segment/batch_size+j,:]
				y_train_sub_batch[iter3] = y_train_sub[iter3*subset_size*num_segment/batch_size+j,:]
				'''	
				# for debugging
				if iter3 == 0:
					print '\n'+str(iter3*subset_size*num_segment/batch_size+j)
				else:
					print iter3*subset_size*num_segment/batch_size+j
				'''

			j = j + 1
			yield (x_train_sub_batch,y_train_sub_batch)

		if j == subset_size*num_segment/batch_size:
			j = 0
		i = i + 1
		if i == options.partition:
			i = 0

def generator_valid(valid_list,y_valid_init,num_frame_input,hop_size,window):
	i = 0

	leng = len(valid_list)
	batch_size = int(leng/options.partition_valid)
	# example, total 201680, partition=40, subset_size=5042, batch_size=50
	conv_window_size = calculate_window_size(hop_size,window)
	sample_length = calculate_sample_length(hop_size,num_frame_input)
	num_segment = calculate_num_segment(hop_size,num_frame_input)
	# example, num_frame_input=27, hop_size=512, sample_length=15360, conv_window_size=2048

	while 1:
		# load subset
		x_valid_batch = np.zeros((batch_size*num_segment,sample_length,1))
		y_valid_batch = np.zeros((batch_size,options.num_tags))

		for iter in range(0,batch_size):
			'''
			# for debugging
			if iter == 0:
				print '\n'+str(iter*options.partition+i)
			else:
				print iter*options.partition+i
			'''

			# load x_train
			tmp = np.load(feature_path + 
					valid_list[iter*options.partition_valid+i])
			for iter2 in range(0,num_segment):

				x_valid_batch[num_segment*iter+iter2,:,0] = tmp[hop_size*(iter2*num_frame_input):hop_size*(iter2*num_frame_input+num_frame_input)]

			y_valid_batch[iter] = y_valid_init[iter*options.partition_valid+i,:]

		# duplication
		y_valid_batch = np.repeat(y_valid_batch,num_segment,axis=0)

		yield (x_valid_batch,y_valid_batch)

		i = i + 1
		if i == options.partition_valid:
			i = 0


class SGDLearningRateTracker(Callback):
	def on_epoch_end(self,epoch,logs={}):
		optimizer = self.model.optimizer

		# lr printer
		lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
		print('\nEpoch%d lr: %.6f' % (epoch+1,lr))


def rerun(model_name,num_frame_input,lr,lr_prev,hop_size,window):

	# load data
	train_list = cP.load(open(label_path + 'train_list_pub.cP','r'))
	valid_list = cP.load(open(label_path + 'valid_list_pub.cP','r'))
	test_list = cP.load(open(label_path + 'test_list_pub.cP','r'))
	print len(train_list),len(valid_list),len(test_list)
	
	y_train_init = np.load(label_path + 'y_train_pub.npy')
	y_valid_init = np.load(label_path + 'y_valid_pub.npy')
	y_test = np.load(label_path + 'y_test_pub.npy')
	print 'data loaded!!!'

	# parameters
	batch_size = options.batch_size
	nb_epoch = options.nb_epoch
	lrdecay = options.lrdecay
	num_frames_per_song = options.num_frames_per_song
	conv_window_size = calculate_window_size(hop_size,window)
	
	# load model
	sample_length = calculate_sample_length(hop_size,num_frame_input)
	architecture_name = 'model_architecture_%s_%d_%d_%.6f.json' % (model_name,conv_window_size,hop_size,lr_prev)

	json_file = open(architecture_name,'r')
	loaded_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_json)

	load_weight = 'best_weights_%s_%d_%d_%.6f.hdf5' % (model_name,conv_window_size,hop_size,lr_prev)
	model.load_weights(load_weight)
	print 'model loaded!!!!'

	# compile & optimizer
	sgd = SGD(lr=lr,decay=lrdecay,momentum=0.9,nesterov=True)
	model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

	# print model summary
	print model_name
	print 'lr: ' + str(lr)
	print model.summary()

	# train / valid / test
	train_size = options.train_size
	valid_size = options.valid_size
	test_size = options.test_size

	train_list = train_list[0:train_size]
	valid_list = valid_list[0:valid_size]
	test_list = test_list[0:test_size]
	
	y_train_init = y_train_init[:train_size,:]
	y_valid_init = y_valid_init[:valid_size,:]
	y_test = y_test[:test_size,:]

	# counting song segments
	num_segment = calculate_num_segment(hop_size,num_frame_input)
	print 'Number of segments per song: ' + str(num_segment)	
	print 'Conv window size: ' + str(conv_window_size)
	print 'Hop size: ' + str(hop_size)

	# load valid set
	x_valid = np.zeros((valid_size*num_segment,sample_length,1))
	y_valid = np.repeat(y_valid_init,num_segment,axis=0)
	print x_valid.shape,y_valid.shape

	for iter in range(0,valid_size):
		tmp = np.load(feature_path + valid_list[iter])
		
		# segmentation
		for iter2 in range(0,num_segment):
			x_valid[num_segment*iter+iter2,:,0] = tmp[hop_size*(iter2*num_frame_input):hop_size*(iter2*num_frame_input+num_frame_input)]

		if np.remainder(iter,500) == 0:
			print iter
	print iter+1

	print x_valid.shape

	# Callbacks
	weight_name = 'best_weights_%s_%d_%d_%.6f.hdf5' % (model_name,conv_window_size,hop_size,lr)
	checkpointer = ModelCheckpoint(weight_name,monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
	earlyStopping = EarlyStopping(monitor='val_loss',patience=options.patience,verbose=0,mode='auto')
	lr_tracker = SGDLearningRateTracker()

	# fit generator!!!
	hist = model.fit_generator(generator_train(train_list,y_train_init,num_frame_input,hop_size,window),
			callbacks=[earlyStopping,checkpointer,lr_tracker],max_q_size=options.max_q_size,
			samples_per_epoch=train_size*num_segment,nb_epoch=nb_epoch,verbose=1,
			validation_data=(x_valid,y_valid))

	# save model architecture
	json_string = model.to_json()
	json_name = 'model_architecture_%s_%d_%d_%.6f.json' % (model_name,conv_window_size,hop_size,lr)
	open(json_name,'w').write(json_string)

	# load best model weights for testing
	model.load_weights(weight_name)
	print 'best model loaded for testing'

	#------------------test set!!------------------------------------------
	# load test set
	x_test_tmp = np.zeros((num_segment,sample_length,1))
	predx_test = np.zeros((test_size,options.num_tags))
	test_split = 12
	for iter2 in range(0,test_split):
		for iter in range(iter2*int(test_size/test_split),(iter2+1)*int(test_size/test_split)):
			file_name = feature_path + test_list[iter]
			tmp = np.load(file_name)

			for iter3 in range(0,num_segment):
				x_test_tmp[iter3,:,0] = tmp[hop_size*(iter3*num_frame_input):hop_size*(iter3*num_frame_input+num_frame_input)]
			
			# prediction each segments & Average them
			predx_test[iter] = np.mean(model.predict(x_test_tmp),axis=0)

			if np.remainder(iter,1000) == 0:
				print iter
		print iter+1
	
	print 'predx_test shape: ' + str(predx_test.shape)
	print 'y_test.shape: ' + str(y_test.shape)

	test_aroc, test_map_tmp = eval_retrieval(predx_test,y_test)
	print model_name
	print ('total_test_aroc: %.4f' % test_aroc)


	save_dir = '/home/richter/pubtagatune1DCNN_beat_endToEnd/result_dir_end/'
	save_name = model_name + '_' + 'conv_window:' + str(conv_window_size) + '_' + 'hop_size:' + str(hop_size) + '_' + 'lr:' + str(lr) + '_' + str(test_aroc) + '.pkl'
	save_list = []
	save_list.append('test_aroc: %.4f' % (test_aroc))

	cP.dump(save_list,open(save_dir+save_name,'w'))
	print 'result save done!!!'




def main(model_name,num_frame_input,window,hop_size):

	# load data
	train_list = cP.load(open(label_path + 'train_list_pub.cP','r'))
	valid_list = cP.load(open(label_path + 'valid_list_pub.cP','r'))
	test_list = cP.load(open(label_path + 'test_list_pub.cP','r'))
	print len(train_list),len(valid_list),len(test_list)
	
	y_train_init = np.load(label_path + 'y_train_pub.npy')
	y_valid_init = np.load(label_path + 'y_valid_pub.npy')
	y_test = np.load(label_path + 'y_test_pub.npy')
	print 'data loaded!!!'

	# parameters
	batch_size = options.batch_size
	nb_epoch = options.nb_epoch
	lr_list = options.lr
	lr = lr_list[0]
	lrdecay = options.lrdecay
	num_frames_per_song = options.num_frames_per_song
	conv_window_size = calculate_window_size(hop_size,window)
	
	# build model
	sample_length = calculate_sample_length(hop_size,num_frame_input)
	model_config = ('build_model%s(sample_length,hop_size,window)' % model_name)
	print model_config
	model = eval(model_config)

	# compile & optimizer
	sgd = SGD(lr=lr,decay=lrdecay,momentum=0.9,nesterov=True)
	model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

	# print model summary
	print model.summary()

	# train / valid / test
	train_size = options.train_size
	valid_size = options.valid_size
	test_size = options.test_size

	train_list = train_list[0:train_size]
	valid_list = valid_list[0:valid_size]
	test_list = test_list[0:test_size]
	
	y_train_init = y_train_init[:train_size,:]
	y_valid_init = y_valid_init[:valid_size,:]
	y_test = y_test[:test_size,:]

	# counting song segments
	num_segment = calculate_num_segment(hop_size,num_frame_input)
	print 'Number of segments per song: ' + str(num_segment)	
	print 'Conv window size: ' + str(conv_window_size)
	print 'Hop size: ' + str(hop_size)

	# load valid set
	x_valid = np.zeros((valid_size*num_segment,sample_length,1))
	y_valid = np.repeat(y_valid_init,num_segment,axis=0)
	print x_valid.shape,y_valid.shape

	for iter in range(0,valid_size):
		tmp = np.load(feature_path + valid_list[iter])
		
		# segmentation
		for iter2 in range(0,num_segment):
			x_valid[num_segment*iter+iter2,:,0] = tmp[hop_size*(iter2*num_frame_input):hop_size*(iter2*num_frame_input+num_frame_input)]

		if np.remainder(iter,500) == 0:
			print iter
	print iter+1

	print x_valid.shape

	# Callbacks
	weight_name = 'best_weights_%s_%d_%d_%.6f.hdf5' % (model_name,conv_window_size,hop_size,lr)
	checkpointer = ModelCheckpoint(weight_name,monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
	earlyStopping = EarlyStopping(monitor='val_loss',patience=options.patience,verbose=0,mode='auto')
	lr_tracker = SGDLearningRateTracker()

	# fit generator!!!
	hist = model.fit_generator(generator_train(train_list,y_train_init,num_frame_input,hop_size,window),
			callbacks=[earlyStopping,checkpointer,lr_tracker],max_q_size=options.max_q_size,
			samples_per_epoch=train_size*num_segment,nb_epoch=nb_epoch,verbose=1,
			validation_data=(x_valid,y_valid))

	# save model architecture
	json_string = model.to_json()
	json_name = 'model_architecture_%s_%d_%d_%.6f.json' % (model_name,conv_window_size,hop_size,lr)
	open(json_name,'w').write(json_string)

	# load best model weights for testing
	model.load_weights(weight_name)
	print 'best model loaded for testing'

	#------------------test set!!------------------------------------------
	# load test set
	x_test_tmp = np.zeros((num_segment,sample_length,1))
	predx_test = np.zeros((test_size,options.num_tags))
	test_split = 12
	for iter2 in range(0,test_split):
		for iter in range(iter2*int(test_size/test_split),(iter2+1)*int(test_size/test_split)):
			file_name = feature_path + test_list[iter]
			tmp = np.load(file_name)

			for iter3 in range(0,num_segment):
				x_test_tmp[iter3,:,0] = tmp[hop_size*(iter3*num_frame_input):hop_size*(iter3*num_frame_input+num_frame_input)]
			
			# prediction each segments & Average them
			predx_test[iter] = np.mean(model.predict(x_test_tmp),axis=0)

			if np.remainder(iter,1000) == 0:
				print iter
		print iter+1
	
	print 'predx_test shape: ' + str(predx_test.shape)
	print 'y_test.shape: ' + str(y_test.shape)

	test_aroc, test_map_tmp = eval_retrieval(predx_test,y_test)
	print model_config
	print ('total_test_aroc: %.4f' % test_aroc)


	save_dir = '/home/richter/pubtagatune1DCNN_beat_endToEnd/result_dir_end/'
	save_name = model_name + '_' + 'conv_window:' + str(conv_window_size) + '_' + 'hop_size:' + str(hop_size) + '_' + 'lr:' + str(lr) + '_' + str(test_aroc) + '.pkl'
	save_list = []
	save_list.append('test_aroc: %.4f' % (test_aroc))

	cP.dump(save_list,open(save_dir+save_name,'w'))
	print 'result save done!!!'

	

if __name__ == '__main__':
	
	model_list = options.model_list
	window_size = options.conv_window_size
	hop_size = options.hop_size
	lr_list = options.lr

	for window in range(0,len(options.conv_window_size)):
		for hop in range(0,len(options.hop_size)):
			for iter in range(len(model_list)):
				num_list = re.findall(r'\d+', model_list[iter])
				frame_input = int(num_list[0])

				conv_window_size = calculate_window_size(hop_size[hop],window_size[window])			
				json_name = 'model_architecture_%s_%d_%d_%.6f.json' % (model_list[iter],conv_window_size,hop_size[hop],lr_list[0])
				if os.path.isfile(json_name) == 1:
					print "already calculated"
					continue
				main(model_list[iter],frame_input,window_size[window],hop_size[hop])

	for window in range(0,len(options.conv_window_size)):
		for hop in range(0,len(options.hop_size)):
			for iter in range(len(model_list)):
				for lr_idx in range(1,len(lr_list)):
					num_list = re.findall(r'\d+', model_list[iter])
					frame_input = int(num_list[0])

					conv_window_size = calculate_window_size(hop_size[hop],window_size[window])
					json_name = 'model_architecture_%s_%d_%d_%.6f.json' % (model_list[iter],conv_window_size,hop_size[hop],lr_list[lr_idx])
					if os.path.isfile(json_name) == 1:
						print 'already calculated'
						continue

					rerun(model_list[iter],frame_input,lr_list[lr_idx],lr_list[lr_idx-1],hop_size[hop],window_size[window])







































