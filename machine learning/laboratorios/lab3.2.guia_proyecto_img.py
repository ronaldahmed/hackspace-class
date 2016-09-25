"""
Prueba de Proyecto Parcial. Imagenes
"""
  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from scipy import misc
from skimage import color, exposure
#from skimage.filter import gaussian
import warnings
import glob as gb

import ipdb
warnings.filterwarnings("ignore")

###########################################################
def plot_image(img):
	plt.figure()
	plt.imshow(img,cmap='gray')
	plt.axis('off')
	plt.show()

###########################################################

## 1) Leer dataset
x_dataset = []
Y = []
for filename in gb.glob('datasets/imagenes/*.ppm'):
	img = misc.imread(filename)
	x_dataset.append( img )
	Y.append( filename )

## 2) preprocess
X = []
for img in x_dataset:
   # 2.1. Convertir a escala de grises
   gray_img = color.rgb2gray(img)

   # 2.2. Ecualizar imagen
   eq = exposure.equalize_hist(gray_img)
   # 2.3. Algun filtro
   # binarizar imagen
   bin_img = (eq>eq.mean()).astype(int)

   plot_image(bin_img)
   #plot_image(bin_img)
   # 2.4. Aplanar imagen
   X.append( np.reshape(bin_img,[-1]) )



## 3) Dividir en training, test
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=42)
























