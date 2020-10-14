from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Flatten, Dense, Dropout
from tensorflow.python.keras import layers

class vgg():
	@staticmethod
	def vgg16(num_classes=1000):
		model = Sequential()
		model.add(layers.ZeroPadding2D((1,1),input_shape=(384, 384, 3)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(128, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(128, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(256, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(256, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(256, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

		model.add(layers.Flatten())
		model.add(layers.Dense(4096, activation='relu'))
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(1024, activation='relu'))
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(num_classes, activation='softmax'))

		return model

	@staticmethod
	def vgg19(num_classes=1000):
		model = Sequential()
		model.add(layers.ZeroPadding2D((1,1),input_shape=(384, 384, 3)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(128, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(128, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(256, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(256, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(256, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(256, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.ZeroPadding2D((1,1)))
		model.add(layers.Conv2D(512, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

		model.add(layers.Flatten())
		model.add(layers.Dense(4096, activation='relu'))
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(1024, activation='relu'))
		model.add(layers.Dropout(0.5))
		model.add(layers.Dense(num_classes, activation='softmax'))

		return model