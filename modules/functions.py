import numpy
import h5py
import cv2
import random
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold, train_test_split

class supportFn():
	@staticmethod
	def shuffle_data(imgs, lbs, random_state=1):
		images = []; labels = []
		for i in range(imgs.shape[0]):
			Data = []
			for c in range(imgs.shape[1]):
				Data.append([imgs[i][c], lbs[i][c]])
			
			for _ in range(random_state):
				random.shuffle(Data)

			_images = []; _labels = []
			for image, label in Data:
				_images.append(image)
				_labels.append(label)
			images.append(_images)
			labels.append(_labels)

		images = numpy.array(images, dtype='uint8')
		labels = numpy.array(labels, dtype='uint8')

		return images, labels

	@staticmethod
	def load_data(FileName):
		# Loading image data...

		h5f = h5py.File(FileName, 'r')

		images = h5f['images'][:]
		labels = h5f['labels'][:]

		h5f.close()

		return images, labels

	@staticmethod
	def split_types(images, labels):
		NR = []; Traumatic_preforation = []; AOM=[];COM=[];Congenital_cholesteatoma=[];OME=[]
		index = 0
		for label in labels:
			if label == 0:
				NR.append(images[index])
			elif label == 1:
				Traumatic_preforation.append(images[index])
			elif label == 2:
				AOM.append(images[index])
			elif label == 3:
				COM.append(images[index])
			elif label == 4:
				Congenital_cholesteatoma.append(images[index])
			elif label == 5:
				OME.append(images[index])
			index += 1
		
		for _ in range(3):
			random.shuffle(NR)
			random.shuffle(Traumatic_preforation)
			random.shuffle(AOM)
			random.shuffle(COM)
			random.shuffle(Congenital_cholesteatoma)
			random.shuffle(OME)
		
		NR = numpy.array(NR, dtype='uint8')
		Traumatic_preforation = numpy.array(Traumatic_preforation, dtype='uint8')
		AOM = numpy.array(AOM, dtype='uint8')
		COM = numpy.array(COM, dtype='uint8')
		Congenital_cholesteatoma = numpy.array(Congenital_cholesteatoma, dtype='uint8')
		OME = numpy.array(OME, dtype='uint8')
		
		return NR, Traumatic_preforation, AOM, COM, Congenital_cholesteatoma, OME

	@staticmethod
	def enlarge_augmentation(Type, times=1):
		dataGenerator = ImageDataGenerator(rotation_range=8.0,
									width_shift_range=0,
									brightness_range=(1.0, 1.25),
									shear_range=0,
									zoom_range=[0.85, 1],
									fill_mode='wrap',
									height_shift_range=0,
									horizontal_flip=True,
									vertical_flip=False)
		
		_Type_Aug = []
		for i in range(Type.shape[0]):
			_img = numpy.expand_dims(Type[i], axis=0)
			Gen = dataGenerator.flow(_img)
			for _ in range(times):
				_Type_Aug.append(next(Gen)[0].astype('uint8'))

		Type_Aug = numpy.array(_Type_Aug, dtype='uint8')
		#print(Type_Aug.shape)
		return Type_Aug

	def enlarge_data(self, NR, Traumatic_preforation,AOM,COM,Congenital_cholesteatoma,OME, times=1):
		max_size = 0
		if (NR.shape[0] > max_size): max_size=NR.shape[0]
		if (Traumatic_preforation.shape[0] > max_size): max_size=Traumatic_preforation.shape[0]
		if (AOM.shape[0] > max_size): max_size=AOM.shape[0]
		if (COM.shape[0] > max_size): max_size=COM.shape[0]
		if (Congenital_cholesteatoma.shape[0] > max_size): max_size=Congenital_cholesteatoma.shape[0]
		if (OME.shape[0] > max_size): max_size=OME.shape[0]

		NR_times = round(times*(max_size/NR.shape[0]))
		Traumatic_preforation_times = round(times*(max_size/Traumatic_preforation.shape[0]))
		AOM_times = round(times*(max_size/AOM.shape[0]))
		COM_times = round(times*(max_size/COM.shape[0]))
		Congenital_cholesteatoma_times = round(times*(max_size/Congenital_cholesteatoma.shape[0]))
		OME_times = round(times*(max_size/OME.shape[0]))
		#print(NR_times, DR_times, GLC_times, AMD_times)

		NR_Aug = self.enlarge_augmentation(NR, times=NR_times)
		Traumatic_preforation_Aug = self.enlarge_augmentation(Traumatic_preforation, times=Traumatic_preforation_times)
		AOM_Aug = self.enlarge_augmentation(AOM, times=AOM_times)
		COM_Aug = self.enlarge_augmentation(COM, times=COM_times)
		Congenital_cholesteatoma_Aug = self.enlarge_augmentation(Congenital_cholesteatoma, times=Congenital_cholesteatoma_times)
		OME_Aug = self.enlarge_augmentation(OME, times=OME_times)

		return NR_Aug, Traumatic_preforation_Aug, AOM_Aug,COM_Aug,Congenital_cholesteatoma_Aug,OME_Aug

	@staticmethod
	def data_ratio_equalization(NR, Traumatic_preforation,AOM,COM,Congenital_cholesteatoma,OME, n_splits=10):
		min_size = NR.shape[0]
		if (Traumatic_preforation.shape[0] < min_size): min_size=Traumatic_preforation.shape[0]
		if (AOM.shape[0] < min_size): min_size=AOM.shape[0]
		if (COM.shape[0] < min_size): min_size=COM.shape[0]
		if (Congenital_cholesteatoma.shape[0] < min_size): min_size=Congenital_cholesteatoma.shape[0]
		if (OME.shape[0] < min_size): min_size=OME.shape[0]
		#print(min_size)
		size_derivation = (min_size//n_splits)*n_splits

		NR_images = []; Traumatic_preforation_images = []; AOM_images=[]; COM_images=[];Congenital_cholesteatoma_images=[];OME_images=[]
		for i in range(size_derivation):
			NR_images.append(NR[i])
			Traumatic_preforation_images.append(Traumatic_preforation[i])
			AOM_images.append(AOM[i])
			COM_images.append(COM[i])
			Congenital_cholesteatoma_images.append(Congenital_cholesteatoma[i])
			OME_images.append(OME[i])
		NR_images = numpy.array(NR_images, dtype='uint8')
		Traumatic_preforation_images = numpy.array(Traumatic_preforation_images, dtype='uint8')
		AOM_images = numpy.array(AOM_images, dtype='uint8')
		COM_images = numpy.array(COM_images, dtype='uint8')
		Congenital_cholesteatoma_images = numpy.array(Congenital_cholesteatoma_images, dtype='uint8')
		OME_images = numpy.array(OME_images, dtype='uint8')
		return NR_images, Traumatic_preforation_images, AOM_images,COM_images,Congenital_cholesteatoma_images,OME_images

	@staticmethod
	def kfold_split(batch_image, n_splits=10):
		train_test_kf = KFold(n_splits=n_splits, shuffle=False, random_state=None)
		times = len(batch_image)//n_splits
		Type = []
		for i in range(times*n_splits):
			Type.append(batch_image[i])
		Type = numpy.array(Type, dtype='uint8')

		train_validation = []; test = []
		for train_validation_idx, test_idx in train_test_kf.split(Type):
			_train_validation, _test = Type[train_validation_idx], Type[test_idx]
			train_validation.append(_train_validation)
			test.append(_test)

		train_validation = numpy.array(train_validation, dtype='uint8')
		train = []; validation = []
		for i in range(len(train_validation)):
			_train, _validation = train_test_split(train_validation[i], test_size=1/(n_splits-1), 
													shuffle=False, random_state=None)
			train.append(_train)
			validation.append(_validation)

		train = numpy.asarray(train)
		validation = numpy.asarray(validation)
		test = numpy.asarray(test)

		return train, validation, test

	@staticmethod
	def combine_images(NR_images, Traumatic_preforation_images, AOM_images,COM_images,Congenital_cholesteatoma_images,OME_images, n_splits=10):
		images = []; labels = []
		for i in range(0, n_splits):
			_images = []; _labels = []
			for c in range(0, NR_images.shape[1]):
				_images.append(NR_images[i][c])
				_labels.append(0)
			for c in range(0, Traumatic_preforation_images.shape[1]):
				_images.append(Traumatic_preforation_images[i][c])
				_labels.append(1)
			for c in range(0, AOM_images.shape[1]):
				_images.append(AOM_images[i][c])
				_labels.append(2)
			for c in range(0, COM_images.shape[1]):
				_images.append(COM_images[i][c])
				_labels.append(3)
			for c in range(0, Congenital_cholesteatoma_images.shape[1]):
				_images.append(Congenital_cholesteatoma_images[i][c])
				_labels.append(4)
			for c in range(0, OME_images.shape[1]):
				_images.append(OME_images[i][c])
				_labels.append(5)
			images.append(_images)
			labels.append(_labels)

		images = numpy.array(images, dtype='uint8')
		labels = numpy.array(labels, dtype='uint8')

		return images, labels

	@staticmethod
	def save_kfold_data(train_images, train_labels,
						validation_images, validation_labels,
						test_images, test_labels, FileName='KFold_data.h5'):
		h5f = h5py.File(FileName, 'a')

		h5f.create_dataset('train_images', data=train_images)
		h5f.create_dataset('train_labels', data=train_labels)
		h5f.create_dataset('validation_images', data=validation_images)
		h5f.create_dataset('validation_labels', data=validation_labels)
		h5f.create_dataset('test_images', data=test_images)
		h5f.create_dataset('test_labels', data=test_labels)

		h5f.close()

	@staticmethod
	def display_accuracy_rate(pLabel, rLabel):
		assert (len(pLabel) == len(rLabel))
		print("")
		count = 0
		for i in range(len(pLabel)):
			if (pLabel[i] == rLabel[i]):
				count += 1
		Accuracy = count*100/len(pLabel)

		return Accuracy