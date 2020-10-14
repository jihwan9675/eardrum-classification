import os
import cv2
import numpy
import h5py
from modules import Display

class DataGenerator():
    def __init__(self, path):
        self.path = path
        self.on_epoch_end()

    @staticmethod
    def contrast_limited_adaptive_HE(channel_img):
        assert(len(channel_img.shape)==2)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))      # Create CLAHE Object
        clahe_image = numpy.empty(channel_img.shape, dtype='uint8')
        clahe_image = clahe.apply(numpy.array(channel_img, dtype='uint8'))

        return clahe_image

    def image_preprocessing(self, img):
        #Display.Im3D(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img)

        #Display.Im2D(l)
        #Display.Hist2D(l, 'r')
        l = self.contrast_limited_adaptive_HE(l)
        #Display.Hist2D(l, 'r')
        #Display.Im2D(l)

        processed_image = cv2.merge((l, a, b))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_LAB2RGB)
        #Display.Im3D(processed_image)
        
        '''
        Display.Im3D(img)
        r, g, b = cv2.split(img)

        #Display.Im2D(r)
        r = self.contrast_limited_adaptive_HE(r)
        #Display.Im2D(r)

        #Display.Im2D(g)
        g = self.contrast_limited_adaptive_HE(g)
        #Display.Im2D(g)

        #Display.Im2D(b)
        b = self.contrast_limited_adaptive_HE(b)
        #Display.Im2D(b)

        processed_image = cv2.merge((r, g, b))
        Display.Im3D(processed_image)
        '''
        
        return processed_image

    def __getimage__(self):
        images = []; labels = []

        # Get all images in Folders
        _types = next(os.walk(self.path))[1]
        print(_types)

        for _type in _types:
            if _type == 'normal':
                _label = 0
            elif _type == 'Traumatic_preforation':
                _label = 1
            elif _type == 'AOM':
                _label = 2
            elif _type == 'COM':
                _label = 3
            elif _type == 'Congenital_cholesteatoma':
                _label = 4
            elif _type == 'OME':
                _label = 5
                print("suc")
            Mask = numpy.zeros((600, 600), dtype='uint8')
            Mask[Mask.shape[0]//2][Mask.shape[1]//2] = 255
            GLCMask = cv2.dilate(Mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                (Mask.shape[0]-30, Mask.shape[1]-30)), iterations=1)
            _IDs = next(os.walk(self.path+_type+'/'))[2]
            for _ID in _IDs:
                print("Reading: "+ os.path.join(self.path, _type, _ID))
                _img = cv2.imread(os.path.join(self.path, _type, _ID))
                _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
                _img = cv2.resize(_img, (600, 600))
                _img = cv2.bitwise_and(_img, _img, mask=GLCMask)
                _img = cv2.resize(_img, (384, 384))
                # Preprocessing
                _img = self.image_preprocessing(_img)

                images.append(_img)
                labels.append(_label)
        
        images = numpy.array(images, dtype='uint8')
        labels = numpy.array(labels, dtype='uint8')

        return images, labels

    @staticmethod
    def __save__(images, labels, FileName='image_data.h5'):
        # Save images and theirs labels to file
        h5f = h5py.File(FileName, 'a')

        h5f.create_dataset('images', data=images)
        h5f.create_dataset('labels', data=labels)

        h5f.close()

    @staticmethod
    def on_epoch_end():
        pass

    def __len__(self):
        return int(numpy.ceil(len(self.IDs)))