import sys, os
import numpy
import h5py
from PIL import Image
from modules import Function, Parameters, Display
from keras.applications.inception_v3 import InceptionV3
#from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from modules.load_image import DataGenerator
from keras import optimizers, losses, metrics
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plots

def warn(*args, **kwargs):
	pass

def build_model(nb_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(nb_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = True
    Adadelta0 = optimizers.Adadelta(lr=0.13)
    # compile the model (should be done *after* setting layers to non-trainable)
    print("starting model compile")
    #compile(model)
    model.compile(optimizer=Adadelta0,loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
    print("model compile done")
    return model

def main():
	# Get Hyperparameters
	batch_size16, batch_size32, epochs = Parameters.train_parameters()
	image_path = Parameters.get_image_path()

	train_test_set_name = 'train_test_sets.h5'

	if not os.path.isfile(train_test_set_name):
		ImgGen = DataGenerator(image_path)
		images, labels = ImgGen.__getimage__()				# Load images
		ImgGen.__save__(images, labels, FileName=train_test_set_name)
	
	KFold_train_test_set_name = 'KFold_image_data.h5'

	if not os.path.isfile(KFold_train_test_set_name):
		# Read all images by theirs labels
		images, labels = Function.load_data(train_test_set_name)
		print(images.shape)
		# Split to disease or normal types
		NR, Traumatic_preforation, AOM, COM, Congenital_cholesteatoma, OME = Function.split_types(images, labels)
		print(NR.shape, Traumatic_preforation.shape,AOM.shape,COM.shape,Congenital_cholesteatoma.shape,OME.shape)
		NR, Traumatic_preforation, AOM, COM, Congenital_cholesteatoma, OME = Function.enlarge_data(NR, Traumatic_preforation, AOM, COM, Congenital_cholesteatoma, OME, times=3)
		n_splits = 5

		NR, Traumatic_preforation, AOM, COM, Congenital_cholesteatoma, OME = Function.data_ratio_equalization(NR, Traumatic_preforation, AOM, COM, Congenital_cholesteatoma, OME, n_splits=n_splits)
		print(NR.shape, Traumatic_preforation.shape,AOM.shape,COM.shape,Congenital_cholesteatoma.shape)

		NR_train, NR_validation, NR_test = Function.kfold_split(NR, n_splits=n_splits)
		Traumatic_preforation_train, Traumatic_preforation_validation, Traumatic_preforation_test = Function.kfold_split(Traumatic_preforation, n_splits=n_splits)
		AOM_train, AOM_validation, AOM_test = Function.kfold_split(AOM, n_splits=n_splits)
		COM_train, COM_validation, COM_test = Function.kfold_split(COM, n_splits=n_splits)
		Congenital_cholesteatoma_train, Congenital_cholesteatoma_validation, Congenital_cholesteatoma_test = Function.kfold_split(Congenital_cholesteatoma, n_splits=n_splits)
		OME_train, OME_validation, OME_test = Function.kfold_split(OME, n_splits=n_splits)

		train_images, train_labels = Function.combine_images(NR_train, Traumatic_preforation_train,AOM_train,COM_train,Congenital_cholesteatoma_train,OME_train, n_splits=n_splits)
		validation_images, validation_labels = Function.combine_images(NR_validation, Traumatic_preforation_validation,AOM_validation,COM_validation,Congenital_cholesteatoma_validation,OME_validation, n_splits=n_splits)
		test_images, test_labels = Function.combine_images(NR_test, Traumatic_preforation_test,AOM_test,COM_test,Congenital_cholesteatoma_test,OME_test, n_splits=n_splits)

		Function.save_kfold_data(train_images, train_labels,
								validation_images, validation_labels,
								test_images, test_labels, FileName=KFold_train_test_set_name)

	shuffled_train_validation_set_name = 'shuffled_train_validation_set.h5'

	if not os.path.isfile(shuffled_train_validation_set_name):
		# Load train images and labels
		h5f = h5py.File(KFold_train_test_set_name, 'r')

		train_images = h5f['train_images'][:]
		train_labels = h5f['train_labels'][:]
		validation_images = h5f['validation_images'][:]
		validation_labels = h5f['validation_labels'][:]

		h5f.close()

		shuffled_train_images, shuffled_train_labels = Function.shuffle_data(train_images, train_labels, random_state=5)
		shuffled_validation_images, shuffled_validation_labels = Function.shuffle_data(validation_images, validation_labels, random_state=5)

		# Save train images and labels
		h5f = h5py.File(shuffled_train_validation_set_name, 'a')

		h5f.create_dataset('shuffled_train_images', data=shuffled_train_images)
		h5f.create_dataset('shuffled_train_labels', data=shuffled_train_labels)
		h5f.create_dataset('shuffled_validation_images', data=shuffled_validation_images)
		h5f.create_dataset('shuffled_validation_labels', data=shuffled_validation_labels)

		h5f.close()

	# Load train images and labels
	h5f = h5py.File(shuffled_train_validation_set_name, 'r')

	shuffled_train_images = h5f['shuffled_train_images'][:]
	shuffled_train_labels = h5f['shuffled_train_labels'][:]
	shuffled_validation_images = h5f['shuffled_validation_images'][:]
	shuffled_validation_labels = h5f['shuffled_validation_labels'][:]

	h5f.close()

	for i in range(shuffled_train_images.shape[0]):
		#input_tensor = Input(shape=(384, 384, 3))
		#384, 384, 3)
		#model=InceptionV3(include_top=False,input_tensor=None,input_shape=None, weights=None, include_top=True, classes=6)
		#model=InceptionV3(include_top=False,input_tensor=None,input_shape=None, weights=None, include_top=True, classes=6)
		model=build_model(6)

		Adadelta0 = optimizers.Adadelta(lr=0.13)

		monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=30, verbose=1, 
								mode='auto', restore_best_weights=True)

		model.compile(	optimizer=Adadelta0,
						loss='sparse_categorical_crossentropy',
						metrics=['sparse_categorical_accuracy'])

		history0 = model.fit(shuffled_train_images[i], shuffled_train_labels[i],
							validation_data=(shuffled_validation_images[i], shuffled_validation_labels[i]), 
							callbacks=[monitor], batch_size=batch_size16, epochs=epochs, verbose=1)

		model.save_weights('Weight/BS32_AUG_KF5_proeardrum_inceptionv3_013/KF'+str(i)+'Weight000120_Aug_inceptionV3_ji.h5')
		'''
		model.compile(	optimizer=Adadelta1,
						loss='sparse_categorical_crossentropy', 
						metrics=['sparse_categorical_accuracy'])

		history1 = model.fit(shuffled_train_images[i], shuffled_train_labels[i],
							validation_data=(shuffled_validation_images[i], shuffled_validation_labels[i]), 
							callbacks=[monitor], batch_size=batch_size, epochs=epochs, verbose=1)

		model.save_weights('Weight/HE_patience20_BS64_20190829/KF'+str(i)+'Weight1e1.h5')

		model.compile(	optimizer=Adadelta2,
						loss='sparse_categorical_crossentropy', 
						metrics=['sparse_categorical_accuracy'])

		history2 = model.fit(shuffled_train_images[i], shuffled_train_labels[i],
							validation_data=(shuffled_validation_images[i], shuffled_validation_labels[i]), 
							callbacks=[monitor], batch_size=batch_size, epochs=epochs, verbose=1)

		model.save_weights('Weight/HE_patience20_BS64_20190829/KF'+str(i)+'Weight1e2.h5')

		model.compile(	optimizer=Adadelta3,
						loss='sparse_categorical_crossentropy', 
						metrics=['sparse_categorical_accuracy'])

		history3 = model.fit(shuffled_train_images[i], shuffled_train_labels[i],
							validation_data=(shuffled_validation_images[i], shuffled_validation_labels[i]), 
							callbacks=[monitor], batch_size=batch_size, epochs=epochs, verbose=1)

		model.save_weights('Weight/HE_patience20_BS64_20190829/KF'+str(i)+'Weight1e3.h5')

		model.compile(	optimizer=Adadelta4,
						loss='sparse_categorical_crossentropy', 
						metrics=['sparse_categorical_accuracy'])

		history4 = model.fit(shuffled_train_images[i], shuffled_train_labels[i],
							validation_data=(shuffled_validation_images[i], shuffled_validation_labels[i]), 
							callbacks=[monitor], batch_size=batch_size, epochs=epochs, verbose=1)

		model.save_weights('Weight/HE_patience20_BS64_20190829/KF'+str(i)+'Weight1e4.h5')
		'''
		# Save cross validation train history
		history_location = 'history/BS32_AUG_KF5_proeardrum_inceptionv3_013/KF'+str(i)+'_cv_history000120_train50_Aug_inceptionV3_ji.h5'
		#print(history_location)
		h5f = h5py.File(history_location, 'w')

		h5f.create_dataset('history0_train_accuracy', data=history0.history['sparse_categorical_accuracy'])
		#h5f.create_dataset('history1_train_accuracy', data=history1.history['sparse_categorical_accuracy'])
		#h5f.create_dataset('history2_train_accuracy', data=history2.history['sparse_categorical_accuracy'])
		#h5f.create_dataset('history3_train_accuracy', data=history3.history['sparse_categorical_accuracy'])
		#h5f.create_dataset('history4_train_accuracy', data=history4.history['sparse_categorical_accuracy'])

		h5f.create_dataset('history0_train_loss', data=history0.history['loss'])
		#h5f.create_dataset('history1_train_loss', data=history1.history['loss'])
		#h5f.create_dataset('history2_train_loss', data=history2.history['loss'])
		#h5f.create_dataset('history3_train_loss', data=history3.history['loss'])
		#h5f.create_dataset('history4_train_loss', data=history4.history['loss'])

		h5f.create_dataset('history0_val_accuracy', data=history0.history['val_sparse_categorical_accuracy'])
		#h5f.create_dataset('history1_val_accuracy', data=history1.history['val_sparse_categorical_accuracy'])
		#h5f.create_dataset('history2_val_accuracy', data=history2.history['val_sparse_categorical_accuracy'])
		#h5f.create_dataset('history3_val_accuracy', data=history3.history['val_sparse_categorical_accuracy'])
		#h5f.create_dataset('history4_val_accuracy', data=history4.history['val_sparse_categorical_accuracy'])

		h5f.create_dataset('history0_val_loss', data=history0.history['val_loss'])
		#h5f.create_dataset('history1_val_loss', data=history1.history['val_loss'])
		#h5f.create_dataset('history2_val_loss', data=history2.history['val_loss'])
		#h5f.create_dataset('history3_val_loss', data=history3.history['val_loss'])
		#h5f.create_dataset('history4_val_loss', data=history4.history['val_loss'])

		h5f.close()

if __name__ == '__main__':
	try:
		main()

		sys.exit()
	except (EOFError, KeyboardInterrupt) as err:
		print("Keyboard Interupted or Error!!!")