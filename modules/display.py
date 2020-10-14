import numpy
import matplotlib.pyplot as plots

class Display():
    @staticmethod
    def Im2D(img):
        assert (len(img.shape)==2)
        plots.figure()
        plots.imshow(img, cmap='gray')
        plots.grid(False)
        plots.show()

    @staticmethod
    def Im3D(img):
        assert (len(img.shape)==3)
        plots.figure()
        plots.imshow(img)
        plots.grid(False)
        plots.show()

    @staticmethod
    def _4img3D(img1, plb1, rlb1, img2, plb2, rlb2, img3, plb3, rlb3, img4, plb4, rlb4):
        assert (len(img1.shape)==3)
        assert (len(img2.shape)==3)
        assert (len(img3.shape)==3)
        assert (len(img4.shape)==3)
        fig = plots.figure(figsize=(10, 8))
        fig.subplots_adjust(hspace=0.4)
        ax = fig.add_subplot(2,2,1)
        ax.imshow(img1)
        ax.set_title('Image 1')
        ax.set_xlabel('Prediction: {0} Fact: {1}'.format(str(plb1), str(rlb1)))
        ax = fig.add_subplot(2,2,2)
        ax.imshow(img2)
        ax.set_title('Image 2')
        ax.set_xlabel('Prediction: {0} Fact: {1}'.format(str(plb2), str(rlb2)))
        ax = fig.add_subplot(2,2,3)
        ax.imshow(img3)
        ax.set_title('Image 3')
        ax.set_xlabel('Prediction: {0} Fact: {1}'.format(str(plb3), str(rlb3)))
        ax = fig.add_subplot(2,2,4)
        ax.imshow(img4)
        ax.set_title('Image 4')
        ax.set_xlabel('Prediction: {0} Fact: {1}'.format(str(plb4), str(rlb4)))
        plots.show()

    @staticmethod
    def Hist2D(img, color='r'):
        assert (len(img.shape)==2)
        plots.figure()
        plots.hist(img.flatten(), 256, [0, 256], color=color)
        plots.xlim([0, 256])
        plots.legend(('Histogram'), loc='upper left')
        plots.title('Image Histogram')
        plots.xlabel('Pixel Level')
        plots.show()