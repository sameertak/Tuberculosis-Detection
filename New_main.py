import scipy.stats
import skimage.measure
import PIL.Image
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import io


class_arr, ent_arr, skew_arr, kur_arr, cont_arr, ener_arr, asm_arr, homo_arr, diss_arr, corr_arr = [], [], [], [], [], [], [], [], [], []

folder_dir = "Tuberculosis"
for images in os.listdir(folder_dir):
    img = PIL.Image.open(f'Tuberculosis/{images}').convert('L')
    img.save(f'TuberculosisCase/{images}')

folder_dir = "TuberculosisCase"
for images in os.listdir(folder_dir):
    if images.endswith(".png"):
        image = io.imread(f"TuberculosisCase/{images}")
        an_image = PIL.Image.open(f"TuberculosisCase/{images}")
        image_sequence = an_image.getdata()
        image_array = np.array(image_sequence)

        glcm = graycomatrix(image, [1], [np.pi / 2])

        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        entropy = skimage.measure.shannon_entropy(image_array)
        skewness = scipy.stats.skew(image_array)
        kurtosis = scipy.stats.skew(image_array)

        ener_arr.append(energy)
        homo_arr.append(homogeneity)
        diss_arr.append(dissimilarity)
        corr_arr.append(correlation)
        asm_arr.append(asm)
        cont_arr.append(contrast)
        ent_arr.append(entropy)
        skew_arr.append(skewness)
        kur_arr.append(kurtosis)
        class_arr.append(1)


folder_dir = "Normal"
for images in os.listdir(folder_dir):
    img = PIL.Image.open(f'Normal/{images}').convert('L')
    img.save(f'NormalCase/{images}')

folder_dir = "NormalCase"

for images in os.listdir(folder_dir):
    if images.endswith(".png"):
        image = io.imread(f"NormalCase/{images}")
        an_image = PIL.Image.open(f"NormalCase/{images}")
        image_sequence = an_image.getdata()
        image_array = np.array(image_sequence)

        glcm = graycomatrix(image, [1], [np.pi / 2])

        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        entropy = skimage.measure.shannon_entropy(image_array)
        skewness = scipy.stats.skew(image_array)
        kurtosis = scipy.stats.kurtosis(image_array)

        ener_arr.append(energy)
        homo_arr.append(homogeneity)
        diss_arr.append(dissimilarity)
        corr_arr.append(correlation)
        asm_arr.append(asm)
        cont_arr.append(contrast)
        ent_arr.append(entropy)
        skew_arr.append(skewness)
        kur_arr.append(kurtosis)
        class_arr.append(0)

dct = {'Class': class_arr, 'Entropy': ent_arr, 'Skewness': skew_arr, 'Kurtosis': kur_arr,
       'Contrast': cont_arr, 'Energy': ener_arr, 'Homogeneity': homo_arr,
       'Dissimilarity': diss_arr, 'Correlation': corr_arr}
df = pd.DataFrame(dct)
df.to_csv('Dataset.csv')
