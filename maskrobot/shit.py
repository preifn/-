# coding: utf-8

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from data import load_data
import numpy as np
import random

np.random.seed(10)


(x_train,y_train),(x_test,y_test)=load_data()

x_train = x_train.transpose(0, 2, 3, 1)
x_test = x_test.transpose(0, 2, 3, 1)

index_1 = [i for i in range(len(x_train))]
random.shuffle(index_1)
x_train = x_train[index_1]
y_train = y_train[index_1]

index_2 = [i for i in range(len(x_test))]
random.shuffle(index_2)
x_test = x_test[index_2]
y_test = y_test[index_2]

print("train data:",'images:',x_train.shape," labels:",y_train.shape) 
print("test data:",'images:',x_test.shape," labels:",y_test.shape) 

x_train_normalize = x_train.astype('float32') / 255.0
x_test_normalize = x_test.astype('float32') / 255.0

y_train_OneHot = keras.utils.to_categorical(y_train)
y_test_OneHot = keras.utils.to_categorical(y_test)

print(y_train_OneHot.shape)
print(y_test_OneHot.shape)



model = Sequential()


model.add(Conv2D(filters=32,kernel_size=(7,7),
                 input_shape=(100, 100,3), 
                 activation='relu', 
                 padding='same'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dropout(rate=0.25))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(2, activation='softmax'))

print(model.summary())



try:
    model.load_weights("./cifarCnnModel.h5")
    print("載入模型成功!繼續訓練模型")
except :    
    print("載入模型失敗!開始訓練一個新模型")


# 訓練模型

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])
train_history=model.fit(x_train_normalize, y_train_OneHot,
                        validation_split=0.2,
                        epochs=20, batch_size=256, verbose=1)          

import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')



scores = model.evaluate(x_test_normalize,y_test_OneHot,verbose=1)
print(scores[:10])


# 進行預測
prediction = np.argmax(model.predict(x_test_normalize), axis=-1)
prediction[:10]

# 查看預測結果

label_dict={0:"bad",1:"good"}
			
print(label_dict)		

import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(i)+','+label_dict[labels[i]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

plot_images_labels_prediction(x_test_normalize,y_test,prediction,0,10)

# 查看預測機率

Predicted_Probability=model.predict(x_test_normalize)

def show_Predicted_Probability(y,prediction,x_img,Predicted_Probability,i):
    print('label:',label_dict[y[i]],
          'predict:',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_test[i],(100,100,3)))
    plt.show()
    for j in range(2):
        print(label_dict[j]+ ' Probability:%1.9f'%(Predicted_Probability[i][j]))

show_Predicted_Probability(y_test,prediction,x_test_normalize,Predicted_Probability,0)
show_Predicted_Probability(y_test,prediction,x_test_normalize,Predicted_Probability,3)

# Step 8. Save Weight to h5 

#model.save_weights("./cifarCnnModel.h5")
print("Saved model to disk")