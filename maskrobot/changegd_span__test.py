import os
import cv2
import numpy as np
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
opfl="./gd"
def main(): 
    file_list = os.listdir(opfl)
    count = 0
    for name in file_list:
        image = cv2.imread(opfl+r"/"+name)
        img = cv2.resize(image, (100, 100))#正方
        #img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) #黑白
        data = img_to_array(img)
        samples = expand_dims(data,0)
        datagen = ImageDataGenerator(rotation_range=120)
        it = datagen.flow(samples, batch_size=1)
        for i in range(1):#bad*250 gd*50
            batch= it.next()
            img = batch[0].astype('uint8')

            data2 = img_to_array(img)#
            samples2 = expand_dims(data2,0)
            datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
            it2 = datagen.flow(samples2, batch_size=1)
            for j in range(5):
                batch = it2.next()
                img = batch[0].astype('uint8')  
                cv2.imwrite("./testImg"+r"/1." + str(count) + ".JPG", img)
                count+=1
main()


