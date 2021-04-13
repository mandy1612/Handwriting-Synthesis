import cv2
import numpy as np
from keras.models import load_model

model = load_model('cnn.h5')
x = cv2.imread('35_18.jpeg',0)
cv2.imshow("x",x)
# x=~x
#compute a bit-wise inversion so black becomes white and vice versa
#make it the right size
x=cv2.resize(x,(28,28))
#convert to a 4D tensor to feed into our model
x = x.reshape(1,28,28,1)
x = x.astype('float32')
x /= 255
#perform the prediction
out = model.predict(x)
print(np.argmax(out))