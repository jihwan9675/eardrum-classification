import h5py
import numpy
import sys
from sklearn.metrics import confusion_matrix, classification_report
from modules import resnet, vgg, Function, Display, Parameters
from keras.applications.inception_v3 import InceptionV3
#from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from modules.load_image import DataGenerator
from keras import optimizers, losses, metrics
from keras.callbacks import EarlyStopping

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
	h5f = h5py.File('KFold_image_data.h5', 'r')

	test_images = h5f['test_images'][:]
	test_labels = h5f['test_labels'][:]

	h5f.close()
	print(test_images[0].shape)
	shuffled_images, shuffled_labels = Function.shuffle_data(test_images, test_labels, random_state=2)

	for i in range(shuffled_images.shape[0]):
		#for c in range(5):
			shuffled_images, shuffled_labels = Function.shuffle_data(test_images, test_labels, random_state=2)
			model = build_model(6)
			model2 = resnet.resnet50(6)
			#model.summary()

			weight_location = 'Weight/BS32_AUG_KF5_proeardrum_inceptionv3_013/KF'+str(i)+'Weight000120_Aug_inceptionV3_ji.h5'
			weight_location2 = 'Weight/BS32_AUG_KF5_proeardrum_resnet50_013/KF'+str(i)+'Weight000120_train50_Aug_ji.h5'
			print(weight_location)
			model.load_weights(weight_location)
			model2.load_weights(weight_location2)

			prediction = model.predict(shuffled_images[i], verbose=1)
			prediction2 = model2.predict(shuffled_images[i], verbose=1)
			predict_labels = []

			for j in range(prediction.shape[0]):
				Max = 0
				idx = -1
				#print('[{0}] inception : {1} {2} {3} {4} {5}'.format(str(j), str(prediction[j][0]), str(prediction[j][1], str(prediction[j][2]),str(prediction[j][3]), str(prediction[j][4]))))
				#print('[{0}] resnet : {1} {2} {3} {4} {5}'.format(str(j), str(prediction2[j][0]), str(prediction2[j][1], str(prediction2[j][2]),str(prediction2[j][3]), str(prediction2[j][4]))))
				for k in range(prediction.shape[1]):
					#print('inception : {0}  /// : resnet : {1}'.format(str(prediction[j][k]), str(prediction2[j][k])))
					if (k == 0):
						Max = prediction[j][0]+prediction2[j][0]
						idx = 0
					else:
						if (prediction[j][k]+prediction2[j][k] > Max):
							Max = prediction[j][k]+prediction2[j][k]

							idx = k
				predict_labels.append(idx)
				'''
				print('')
				print('Normal, Traumatic_Perforation, AOM, COM, Congenital_cholesteatoma, OME')
				print('Prediction probability: {0}, True Label: {1}'.format(prediction[j], shuffled_labels[i][j]))				#To see probability and real label
				Display.Im3D(shuffled_images[i][j])																	#To see Test Image predicted
				'''
			Accuracy = Function.display_accuracy_rate(predict_labels, shuffled_labels[i])
			print('KF{0}: lr=1e-{1}: Test accuracy is {2}%.'.format(str(i), str(3), Accuracy))
			
			print(confusion_matrix(shuffled_labels[i], predict_labels))
			print(classification_report(shuffled_labels[i], predict_labels, target_names=['Normal', 'Traumatic_Perforation', 'AOM', 'COM', 'Congenital_cholesteatoma', 'OME']))
	'''
	for i in range((len(shuffled_images[0]))//4):
		Display._4img3D(shuffled_images[0][i*4], predict_labels[i*4], shuffled_labels[0][i*4], shuffled_images[0][i*4+1], predict_labels[i*4+1], shuffled_labels[0][i*4+1],
				  shuffled_images[0][i*4+2], predict_labels[i*4+2], shuffled_labels[0][i*4+2], shuffled_images[0][i*4+3], predict_labels[i*4+3], shuffled_labels[0][i*4+3])
	'''
	'''
	h5f = h5py.File('TrueFalse.h5', 'a')

	h5f.create_dataset('predict_labels', data=predict_labels)
	h5f.create_dataset('real_labels', data=shuffled_labels[0])

	h5f.close()
	'''

if __name__ == '__main__':
	try:
		main()
		
		sys.exit()
	except (EOFError, KeyboardInterrupt) as err:
		print("Keyboard Interrupted or Error!!!")