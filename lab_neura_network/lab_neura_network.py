import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import  Sequential
from keras.layers import Dense, Dropout, Flatten

(X_train,y_train),(X_test,y_test) = mnist.load_data()

print("X_train shape : ", X_train.shape)
print("y_train shape : ",y_train.shape)
print("X_test shape : ",X_test.shape)
print("y_test shape ",y_test.shape)

print("X_train type :",X_train.dtype)
print("y_train type : ",y_train.dtype)
print("X_test type : ",X_test.dtype)
print("y_test type : ",y_test.dtype)

img_index = 999

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

input_shape = (28,28,1)

X_train = X_train[:2000, :, :, :].astype('float32')
X_test = X_test[:400, :, :, :].astype('float32')

y_train = y_train[:2000]
y_test = y_test[:400]

X_train /=255
X_test /=255

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:',y_test.shape)

model = Sequential()
model.add(Flatten())
model.add(Dense(100, activation='relu', input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train,epochs=20,batch_size=32,validation_split=0.2,verbose=1)

image_index = 26
plt.imshow(X_test[image_index].reshape(28,28), cmap='Greys')

plt.figure(figsize=(9,6))
plt.plot(model.history.history['accuracy'],label='Train accuracy')
plt.plot(model.history.history['val_accuracy'],label='Validation accuracy')
plt.ylabel('Value')
plt.xlabel('No. epoch')

score = model.evaluate(X_test, y_test, verbose=0)
print("loss : ",score[0])
print("Accuracy : ",score[1])

pred = model.predict(X_test[image_index].reshape(1,28,28,1))
print("Prediction probability array is : ")

count = 0
for i in pred.squeeze():
    print(count,":",i)
    count+=1

print("From which the max choice is ",pred.argmax())
plt.show()