import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, Lambda
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist


#Load data from MNIST database
(train_x,train_y),(test_x,) = mnist.load_data()
train_x = train_x.reshape(train_x.shape[0],28,28,1).astype("float32")
test_x = test_x.reshape(test_x.shape[0],28,28,1).astype("float32")

#one-hot encoding
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
num_classes = test_y.shape[1]

#augment data
newData = image.ImageDataGenerator()
batches = newData.flow(train_x, train_y, batch_size = 64)

#normalize data
m = np.mean(train_x)
s = np.std(train_x)

#define standardize
def standardize(x):
    return (x-m)/s

#define model
def model():
    model = Sequential()
    model.add(Lambda(standardize,input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation = "relu"))
    model.add(Conv2D(128, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation = "relu"))
    model.add(Dense(10, activation = "softmax"))
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.fit_generator(generator = batches, steps_per_epoch = batches.n, epochs = 3, validation_data = (test_x, test_y))
    return model

#train model
model = model()

#evaluate model
score = model.evaluate(test_x, test_y, verbose=0)
print("CNN Error:%.2f%%" %(100 - score[1] * 100))

#predict output
test_x = pd.read_csv('../input/test.csv')
test_x = test_x.values.reshape(test_x.shape[0],28,28,1)
preds=model.predict_classes(test_x,verbose=1)
model.save('ConvNet_model.h5')

#export output
pd.DataFrame({"Id": list(range(1, len(preds) + 1)), "Label": preds}).to_csv("test_result.csv", index = False, header = True)
