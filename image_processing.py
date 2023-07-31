import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage as nd
from skimage.filters import sobel,butterworth
from skimage.filters import apply_hysteresis_threshold
from skimage.filters import butterworth
from skimage.filters import farid
from skimage.filters import laplace
from skimage.filters import unsharp_mask
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.feature import canny
from skimage import io, color, filters



img = cv2.imread('FigureL.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img,'gray')
#------------------------------------
plt.imshow(img,'gray');plt.axis(False)
plt.show()
#------------------------------------
histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
print()
plt.plot(histogram, color='black')
plt.xlabel('Piksel Değeri')
plt.ylabel('Piksel Sayısı')
plt.title('Gri Görüntünün Histogramı')
plt.xlim([0, 256])
plt.show()

#------------------------------------
properties = ['contrast', 'correlation', 'energy', 'dissimilarity', 'homogeneity', 'ASM']
glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
haralick_features = [graycoprops(glcm, prop).ravel()[0] for prop in properties]

print("Haralick Textural Features:")
print("Contrast:", haralick_features[0])
print("Correlation:", haralick_features[1])
print("energy:", haralick_features[2])
print("dissimilarity:", haralick_features[3])
print("homogeneity:", haralick_features[4])
print("ASM:", haralick_features[5])
#------------------------------------

entropy_img = entropy(img, disk(1))
plt.imshow(entropy_img,'gray');plt.axis(False)

unsharp_img = unsharp_mask(img, radius=10, amount=1)
plt.imshow(unsharp_img,'gray');plt.axis(False)

gasussian_img = nd.gaussian_filter(unsharp_img, sigma = 1)
plt.imshow(gasussian_img,'gray');plt.axis(False)

sobel_img = sobel(gasussian_img)
plt.imshow(sobel_img,'gray');plt.axis(False)

print(np.mean(sobel_img))

threshold_img = apply_hysteresis_threshold(sobel_img, 0.2 ,0.03)
plt.imshow(threshold_img,'gray');plt.axis(False)

print(np.mean(threshold_img))

butterworth_img = butterworth(sobel_img, 0.1,False ,0.2, channel_axis=-1)
plt.imshow(butterworth_img,'gray');plt.axis(False)

farid_img = farid(img)
plt.imshow(farid_img,'gray');plt.axis(False)

laplace_img = laplace(img)
plt.imshow(laplace_img,'gray');plt.axis(False)

unsharp_img = unsharp_mask(img, radius=10, amount=1)
plt.imshow(unsharp_img,'gray');plt.axis(False)

farid_img = farid(gasussian_img)
plt.imshow(farid_img,'gray');plt.axis(False)

laplace_img = laplace(farid_img)
plt.imshow(laplace_img,'gray');plt.axis(False)

#------------------------------------

# sadece 3'u fotograf icin
y_shape = int(gasussian_img.shape[0]/2)
x_shape = int(gasussian_img.shape[1]/2)
canny_img = canny(gasussian_img[y_shape-10:y_shape+30, x_shape-40:x_shape+40], sigma=1)
print(np.mean(canny_img))
plt.imshow(canny_img,'gray');plt.axis(False)
plt.show()

#------------------------------------


# Gabor filtreleri oluşturma
ksize = 27  # Filtre boyutu
theta = np.pi  # Açı (radyan cinsinden)
sigma = 4.0  # Standart sapma kaa-baz
lambda_ = 15.0  # Dalga boyu
gamma = 0.01  # Gama değeri kaz-baa

gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma)

# Gabor filtresini görüntüye uygulama
filtered_image = cv2.filter2D(img, cv2.CV_64F, gabor_filter)

print(np.mean(filtered_image))

plt.imshow(filtered_image,'gray')
plt.show()

#------------------------------------

# Local Binary Patterns hesaplaması
radius = 1
n_points = 8 * radius
lbp_image = local_binary_pattern(img, n_points, radius, method='uniform')
# LBP özelliklerini histograma dönüştürme
hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
# Histogramı normalize etme
hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)
print(hist[8]) ## bu alinacaktir
plt.bar(range(0, n_points + 2), hist)
plt.title("LBP Histogram")
plt.xlabel("LBP Pattern")
plt.ylabel("Normalized Frequency")
plt.show()

plt.imshow(lbp_image,'gray')
plt.show()

#------------------------------------

#Histogram of Oriented Gradients (HOG)


# HOG özelliklerini hesaplama
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

hog_features, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)
print(np.mean(hog_features))
plt.imshow(hog_image,'gray')
plt.show()

#---------------------------
#kenar yogunlugu

gaussian_image = cv2.GaussianBlur(img, (3, 3), 0)
canny_image = cv2.Canny(gaussian_image, 30 ,60)

print(np.sum(canny_image == 255)/(canny_image.shape[0]*canny_image.shape[1]))

#----------------
#canny

gaussian_image = cv2.GaussianBlur(img, (3, 3), 0)
canny_image = cv2.Canny(gaussian_image, 30 ,60)
plt.imshow(canny_image, 'gray')

# Gabor filtreleri oluşturma
ksize =  3  # Filtre boyutu
theta = np.pi/2  # Açı (radyan cinsinden)
sigma = 2  # Standart sapma kaa-baz
lambda_ = np.pi/2 # Dalga boyu
gamma = 0.01 # Gama değeri kaz-baa

gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma)
plt.imshow(gabor_filter,'gray')
# Gabor filtresini görüntüye uygulama
filtered_image = cv2.filter2D(canny_image, cv2.CV_64F, gabor_filter)
print(1-(np.sum(filtered_image == 0)/(filtered_image.shape[0]*filtered_image.shape[1])))
print(np.mean(filtered_image))
print(np.mean(img))

plt.imshow(filtered_image,'gray')
plt.show()

#------------------

#edge direction
   # Gradientleri hesapla
gradient_x = filters.sobel_h(img)
gradient_y = filters.sobel_v(img)

    # Gradient değerlerini mutlak değerlere dönüştür
gradient_x = np.abs(gradient_x)
gradient_y = np.abs(gradient_y)

    # Yatay ve dikey gradientlerin toplamını hesapla
total_gradient_x = np.sum(gradient_x)
total_gradient_y = np.sum(gradient_y)

    # Kenar yönünü belirle
if total_gradient_x > total_gradient_y:
    print("yatay kenar")
elif total_gradient_y > total_gradient_x:
    print("dikey kenar")

