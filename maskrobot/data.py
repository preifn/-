#coding:utf-8

import os
from PIL import Image
import numpy as np

#彩色圖片輸入,將channel 1 改成 3，data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data():
	imgs_1 = os.listdir("./trainImg")
	num_1 = len(imgs_1)
	imgs_2 = os.listdir("./testImg")
	num_2 = len(imgs_2)
	data_train = np.empty((num_1,3,100,100),dtype="uint8") # for train
	label_train = np.empty((num_1,),dtype="uint8")
	data_test = np.empty((num_2,3,100,100),dtype="uint8") # for test
	label_test = np.empty((num_2,),dtype="uint8")
	
	
	
	for i in range(num_1):
		img_1 = Image.open("./trainImg/"+imgs_1[i])
		arr_1 = np.array(img_1)
		data_train[i,:,:,:] = [arr_1[:,:,0],arr_1[:,:,1],arr_1[:,:,2]]
		label_train[i] = int(imgs_1[i].split('.')[0])
		

	for i in range(num_2):
		img_2 = Image.open("./testImg/"+imgs_2[i])
		arr_2 = np.array(img_2)
		data_test[i,:,:,:] = [arr_2[:,:,0],arr_2[:,:,1],arr_2[:,:,2]]
		label_test[i] = int(imgs_2[i].split('.')[0])

	return (data_train,label_train), (data_test,label_test)
# def main():
# 	load_data()
# main()

