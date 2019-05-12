import sys
import os
import numpy as np
from PIL import Image
from operator import itemgetter


path = "train1"
folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
#print(folder_names)

# Each row is an image
img = np.zeros([94257, 45,45], dtype = float)
labels = np.zeros([94257])

j = 0
for i in range(10):
	for image in os.listdir(path + "/" + folder_names[i]):
		train_image = path + "/" + folder_names[i] + "/" + image
		img[j] = np.asarray(Image.open(train_image).convert('L').resize((45,45), Image.ANTIALIAS))
		img[j] = img[j] / 255.0
		labels[j] = folder_names[i]
		j += 1
y=labels
X=img

from numpy import reshape
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils

"""
yo bhatia this ones for you remove this line and see the git diff 
blahhh blaaah blaaah
qwertyu

path = "train"
folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
print(folder_names)

# Each row is an image
img = np.zeros([160, 2025], dtype = float)
labels = np.zeros([160])

j = 0
for i in range(10):
	for image in os.listdir(path + "/" + folder_names[i]):
		train_image = path + "/" + folder_names[i] + "/" + image
		img[j] = np.asarray(Image.open(train_image).convert('L').resize((45,45), Image.ANTIALIAS)).flatten()
		img[j] = img[j] / 255.0
		labels[j] = folder_names[i]
		j += 1
"""

from sklearn.model_selection import train_test_split
X, valData, y, valLabels = train_test_split(X, y, test_size=0.25, random_state=42)
X = X.reshape(X.shape[0], 45, 45,1)
valData = valData.reshape(valData.shape[0], 45, 45,1)
#valData=img
#valLabels=labels
y = np_utils.to_categorical(y)
valLabels = np_utils.to_categorical(valLabels)
num_classes = valLabels.shape[1]

for k in range(1):
	model=Sequential()
	model.add(Conv2D(128, 5, 5, activation='relu', input_shape=(45,45,1)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Conv2D(128, (5, 5), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(250,input_dim=128, kernel_initializer='normal', activation='relu'))
	model.add(Dropout(0.14))
	#model.add(Dense(200, activation='relu'))
	#model.add(Dropout(0.14))
	model.add(Dense(num_classes,kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X, y, validation_data=(valData, valLabels), epochs=10, batch_size=200, verbose=2)
	scores = model.evaluate(valData, valLabels, verbose=0)
	print("Baseline Error: %.2f%%" % (100-scores[1]*100))
	#sprint("droput value to be:%f"%(0.1+0.01*k))
