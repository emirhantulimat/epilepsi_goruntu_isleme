import SimpleITK as sitk
import cv2
import imageio
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage as nd
from skimage import io, color, filters
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

#NIfTI görüntülerini JPEG dilimlerine dönüştürmek için fonksiyon
def nifti_to_jpg(nifti_path, slice_axis=0):
    """
        Verilen NIfTI formatındaki görüntüyü belirtilen eksen (varsayılan olarak 0) boyunca
    dilimlere böler ve her dilimi JPEG formatında kaydediyor.
    
     Dilimleme için kullanılacak eksenin (slice_axis) 0 olarak belirlendi
    """
    max = 0
    index = 0

    # NIfTI görüntüyü yükleniyor
    img = sitk.ReadImage(nifti_path)
    
    # Görüntüyü numpy dizisine dönüştürüyor
    img_array = sitk.GetArrayFromImage(img)
    

    # Görüntü yoğunluk değerlerinin [0, 1] aralığına normalize ediliyor
    img_array = img_array.astype(np.float32)
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    
    # Dilimlenecek toplam dilim sayısının belirleniyor
    num_slices = img_array.shape[slice_axis]
        
    for slice_number in range(num_slices):
        # 2B dilimlerin elde edilmesi
        if slice_axis == 0:
            slice_2d = img_array[slice_number, :, :]
        elif slice_axis == 1:
            slice_2d = img_array[:, slice_number, :]
        else:
            slice_2d = img_array[:, :, slice_number]
        
        # Dilimi çıktı boyutuna yeniden boyutlandırma
        slice_2d = cv2.cvtColor(slice_2d, cv2.COLOR_BGR2RGB)

        # Dilimi JPEG olarak kaydetme
        slice_2d = (slice_2d * 255).astype(np.uint8)

        image = slice_2d.copy()

        # Dilimi ikili görüntüye dönüştürme
        image[image > 10] = 255
        image[image <= 10] = 0

        # Morfolojik işlem uygulama
        kernel = np.ones((5,5),np.uint8)
        eroded_mask = cv2.dilate(image,kernel,iterations = 1)
        eroded_area = np.sum(eroded_mask)

        #alansal olarak en buyuk beyin goruntusunu aliyor
        if slice_number > 10 and eroded_area > max:
            max = eroded_area
            index = slice_number


    image_arr = []
    #dongude, bulunan en buyuk 3 goruntunun 10 aralıkla array e kaydediliyor
    #  
    for i in range(1,4):
        img = img_array[index + (i*10), :, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img * 255).astype(np.uint8)
        image_arr.append(img)

    for i in range(3):
        image1_copy = image_arr[i].copy()
        image_gray = cv2.cvtColor(image1_copy,cv2.COLOR_BGR2GRAY)
        image_gray[image_gray > 10] = 255
        image_gray[image_gray <= 10] = 0
        # goruntuyu kirpmak icin en buyuk contour bulunuyor
        contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        #  bounding rectangle cizililiyor 
        x, y, w, h = cv2.boundingRect(sorted_contours[0])
        image_arr[i] = image1_copy[y:y+h,x:x+w]
        plt.subplot(1,3,i+1)
        plt.imshow(image_arr[i]);plt.axis("off")
        plt.title(nifti_path[40:-1])
        plt.show()

    image_arr_RL = []
    for img in image_arr:
        y = int(img.shape[0]/2)
        x = int(img.shape[1]/2)
        image_arr_RL.append(img[0:y,0:x])
        plt.imshow(img[0:y,0:x])
        plt.show()
        image_arr_RL.append(img[0:y,x:int(img.shape[1])])
        plt.imshow(img[0:y,x:int(img.shape[1])])
        plt.show()

    # 6 ROI bolgesi seciliyor ve bir arrayle donduruluyor
    return image_arr_RL

# Görüntüleri kırpma işlemi için fonksiyon
def crop(image_arr):
    arr = [[[], []] for _ in range(3)]
    for i in range(3):
        shape = image_arr[i].shape
        arr[i][0] = image_arr[i][0:int(shape[0]/2), 0:int(shape[1]/2)]
        arr[i][1] = image_arr[i][0:int(shape[0]/2), int(shape[1]/2):int(shape[1])]
        plt.subplot(1,2,1)
        # plt.title(largest_three_ero[i*10])
        plt.imshow(arr[i][0]);plt.axis("off")
        plt.subplot(1,2,2)
        # plt.title(largest_three_ero[i*10])
        plt.imshow(arr[i][1]);plt.axis("off")

    return arr

# Canny kenar tespiti uygulamak için fonksiyon
def canny1(image):
    gaussian_image = cv2.GaussianBlur(image, (3, 3), 0)
    canny_image = cv2.Canny(gaussian_image, 30 ,60)

    return np.sum(canny_image == 255)/(canny_image.shape[0]*canny_image.shape[1])

# En yoğun piksellerin sayısını bulmak için fonksiyon
def number_of_densest_pixels(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    index = np.argmax(histogram)
    value = np.max(histogram)

    return index,value

# Kenarların yönünü belirlemek için fonksiyon
def direction_of_edges(image):

    gray_img_array = image[:, :, 0]
    gradient_x = filters.sobel_h(gray_img_array)
    gradient_y = filters.sobel_v(gray_img_array)

        # Gradient değerlerini mutlak değerlere dönüştür
    gradient_x = np.abs(gradient_x)
    gradient_y = np.abs(gradient_y)

        # Yatay ve dikey gradientlerin toplamını hesapla
    total_gradient_x = np.sum(gradient_x)
    total_gradient_y = np.sum(gradient_y)

        # Kenar yönünü belirle
    if total_gradient_x > total_gradient_y:
        #yatay
        return 0
    elif total_gradient_y > total_gradient_x:
                #dikey
        return 1

# Gabor filtresi uygulamak için fonksiyon
def gabor(image):
    # Gabor filtreleri oluşturma
    ksize = 27  # Filtre boyutu
    theta = np.pi  # Açı (radyan cinsinden)
    sigma = 4.0  # Standart sapma kaa-baz
    lambda_ = 15.0  # Dalga boyu
    gamma = 0.01  # Gama değeri kaz-baa
    
    gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambda_, gamma)
    
    # Gabor filtresini görüntüye uygulama
    filtered_image = cv2.filter2D(image, cv2.CV_64F, gabor_filter)

    return np.mean(filtered_image)

# Piksel ortalama değerini hesaplamak için fonksiyon
def pixel_mean(image):
    

    return np.sum(image)/(image.shape[0]*image.shape[1]*255)

# GLCM özelliklerini çıkarmak için fonksiyon
def GLCM_feature(image):
    gray_img_array = image[:, :, 0]

    properties = ['contrast', 'correlation', 'energy', 'dissimilarity', 'homogeneity', 'ASM']
    glcm = graycomatrix(gray_img_array, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    haralick_features = [graycoprops(glcm, prop).ravel()[0] for prop in properties]

    return haralick_features


# Lokal Binari Desen (LBP) özelliğini hesaplamak için fonksiyon

def LBP(image):
    gray_img_array = image[:, :, 0]

    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_img_array, n_points, radius, method='uniform')
    # LBP özelliklerini histograma dönüştürme
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # Histogramı normalize etme
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    lbp = hist[8]
    return lbp

# Merkezi bölgede Canny kenar yoğunluğunu hesaplamak için fonksiyon
def center_ROI_canny(image):
    # sadece 3'u fotograf icin
    gray_img_array = image[:, :, 0]

    unsharp_img = unsharp_mask(gray_img_array, radius=10, amount=1)
    gasussian_img = nd.gaussian_filter(unsharp_img, sigma = 1)
    y_shape = int(gasussian_img.shape[0]/2)
    x_shape = int(gasussian_img.shape[1]/2)
    canny_img = canny(gasussian_img[y_shape-10:y_shape+30, x_shape-40:x_shape+40], sigma = 1)
    return np.mean(canny_img)

# Histograma yönlü gradyan (HOG) özelliğini hesaplamak için fonksiyon
def HOG(image):
    gray_img_array = image[:, :, 0]

    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    hog_features, hog_image = hog(gray_img_array, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)
    return np.mean(hog_features)
        

            
if __name__ == "__main__":
    data_path = os.listdir(r'\dataset')
    features = []
    data = {}
    with open(r'features.txt', 'r') as file:
        for line in file:
            features.append(line.strip())
    for i in range(6):
        for feature in features:
            if feature != '':
                data[f'{feature}{i}'] = []

# Veri setini dolaşarak özellikleri çıkartın
    for index, path in enumerate(data_path):
        image_arr = nifti_to_jpg(rf'dataset\{path}')

        for i, img in enumerate(image_arr):
            pixel_density_, pixel_density_value_ = number_of_densest_pixels(img)
            edge_density_ = canny1(img)
            edge_direction_ = direction_of_edges(img)
            mean_pixel_density_ = pixel_mean(img)
            GLCM = GLCM_feature(img)
            contrast_ = GLCM[0]
            correlation_ = GLCM[1]
            energy_ = GLCM[2]
            dissimilarity_ = GLCM[3]
            homogeneity_ = GLCM[4]
            ASM_ = GLCM[5]
            mean_gabor_value_ = gabor(img)
            LBP_ = LBP(img)
            canny_mean_ = center_ROI_canny(img)
            HOG_ = HOG(img)

            data[f'{features[0]}{i}'].append(pixel_density_)
            data[f'{features[1]}{i}'].append(pixel_density_value_)
            data[f'{features[2]}{i}'].append(edge_density_)
            data[f'{features[3]}{i}'].append(edge_direction_)
            data[f'{features[4]}{i}'].append(mean_pixel_density_)
            data[f'{features[5]}{i}'].append(contrast_)
            data[f'{features[6]}{i}'].append(correlation_)
            data[f'{features[7]}{i}'].append(energy_)
            data[f'{features[8]}{i}'].append(dissimilarity_)
            data[f'{features[9]}{i}'].append(homogeneity_)
            data[f'{features[10]}{i}'].append(ASM_)
            data[f'{features[11]}{i}'].append(mean_gabor_value_)
            data[f'{features[12]}{i}'].append(LBP_)
            data[f'{features[13]}{i}'].append(canny_mean_)
            data[f'{features[14]}{i}'].append(HOG_)

    # Veri çerçevesi oluşturun ve CSV dosyası olarak kaydedin
    df = pd.DataFrame(data)
    df.to_csv("features.csv")