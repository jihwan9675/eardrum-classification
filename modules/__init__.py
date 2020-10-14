import os
from models.resnet_model import resnet
#from models.inception_v3 import InceptionV3
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from models.vgg_model import vgg
from modules.functions import supportFn
from modules.display import Display
from modules.get_hyperparameters import GetHyperparameters

resnet = resnet()

#input_tensor = Input(shape=(384, 384, 3))
#inceptionv3=InceptionV3(input_tensor=input_tensor,weights=None,include_top=True,classes=6)
InceptionV3=InceptionV3()
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
Function = supportFn()
Display = Display()
Parameters = GetHyperparameters()