import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, Lambda
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

#load data
(X_train,y_train),(X_test,y_test)=mnist.load_data()
X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
print(X_train.shape)
print(X_test.shape)

#visualize data
X_train_ = X_train.reshape(X_train.shape[0], 28, 28)
for i in range(0, 3):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train_[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);

#one-hot encoding
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
num_classes=y_test.shape[1]

#augment data
gen=image.ImageDataGenerator()
batches=gen.flow(X_train,y_train,batch_size=64)

#normalize data
mean=np.mean(X_train)
std=np.std(X_train)
def standardize(x):
    return (x-mean)/std

#define model
def model():
    model=Sequential()
    model.add(Lambda(standardize,input_shape=(28,28,1)))
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(Conv2D(64,(3,3),activation="relu"))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),activation="relu"))
    model.add(Conv2D(128,(3,3),activation="relu"))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512,activation="relu"))
    model.add(Dense(10,activation="softmax"))
    
    model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=["accuracy"])
    model.fit_generator(generator=batches,steps_per_epoch=batches.n,epochs=3,validation_data = (X_test,y_test))
    return model

#train model
model=model()

#evaluate model
score=model.evaluate(X_test,y_test,verbose=0)
print("CNN Error:%.2f%%" %(100-score[1]*100))

#predict output
X_test=pd.read_csv('../input/test.csv')
X_test=X_test.values.reshape(X_test.shape[0],28,28,1)
preds=model.predict_classes(X_test,verbose=1)
model.save('digit_recognizer.h5')

#define export method
def write_preds(preds,fname):
    pd.DataFrame({"ImageId":list(range(1,len(preds)+1)),"Label":preds}).to_csv(fname,index=False,header=True)

#export output
write_preds(preds,"cnn-test.csv")
