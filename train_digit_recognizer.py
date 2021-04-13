
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
import random
import numpy as np
from emnist import list_datasets,extract_training_samples,extract_test_samples


#labels: 0-9 -> 0-9, A-Z -> 10-35, a-z -> 36-61 
x_train, y_train = extract_training_samples('byclass')
# print(x_train.shape)
# print(y_train[0:10])
# print(max(y_train),min(y_train))

# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(5,2, figsize=(28,28))

# for i,ax in enumerate(axes.flat):
#     ax.imshow(x_train[i])
    

x_test, y_test = extract_test_samples('byclass')
# print(x_test.shape)
# print(y_test.shape)


x_train = np.array(x_train) / 255.0
print(type(x_train[0]))
y_train = np.array(y_train)
x_test = np.array(x_test) / 255.0
y_test = np.array(y_test)


print("Before: ", x_train.shape)
# # 60000 rows of 28*28 matrix
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print("After: ", x_train.shape)

input_shape = (28, 28, 1)

# # convert class vectors to binary class matrices
# # we apply to_categorical() function since we have multiclass results(0–9: 10 output)
y_train = keras.utils.to_categorical(y_train, 62)
y_test = keras.utils.to_categorical(y_test, 62)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(62, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # loss function
              metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=200,
                 epochs=3, validation_data=(x_test, y_test))
print("The model has successfully trained")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('mnist.h5')
print("Saving the model as emnist.h5")