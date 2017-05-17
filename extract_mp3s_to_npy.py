import numpy as np
import librosa
import os.path
import cPickle as cP
import sys
import glob

fs = 22050 # 16000,22050
nth = 4
max_length = 640512 # 465600(29.1),640512(29.05)

label_path = '../pubtagatune1DCNN_beat/'
feature_path = '/media/ssd2/pubMagnatagatune_mp3s_to_npy/' # pubMagnatagatune_mp3s_to_npy_16k, pubMagnatagatune_mp3s_to_npy
data_path = '/media/bach1/dataset/MagnatagatunePub/mp3/'

train_list = cP.load(open(label_path + 'train_list_pub.cP','r'))
valid_list = cP.load(open(label_path + 'valid_list_pub.cP','r'))
test_list = cP.load(open(label_path + 'test_list_pub.cP','r'))

all_list = train_list + valid_list + test_list
length_id = len(all_list)
print length_id

def main(nth):
	print int(nth*length_id/5),int((nth+1)*length_id/5)

	for iter in range(int(nth*length_id/5),int((nth+1)*length_id/5)):
		save_name = feature_path + all_list[iter]
		file_name = data_path + all_list[iter].replace('.npy','.mp3')

		if not os.path.exists(os.path.dirname(save_name)):
			os.makedirs(os.path.dirname(save_name))

		if os.path.isfile(save_name) == 1:
			print iter, save_name + '_file_exist!!!!!!!!!!!!!!!'
			continue

		y,sr = librosa.load(file_name,sr=fs)
		y = y.astype(np.float32)
		
		if len(y) > max_length:
			y = y[0:max_length]
		print iter,len(y),save_name
		np.save(save_name,y)


if __name__ == '__main__':

	main(nth)













