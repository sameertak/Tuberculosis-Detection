import streamlit as st
import pickle
import os
from skimage import io
import scipy.stats
import skimage.measure
import PIL.Image
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import sklearn
print(sklearn.__version__)
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


def show_predict_page():
    st.title("Detect Tuberculosis")
    st.write("### Upload X-Ray Image")
    image_file = st.file_uploader("", type=["png"])

    if image_file is not None:

        with open(os.path.join("uploads", image_file.name), "wb") as f:
            f.write((image_file).getbuffer())

        st.success("File Uploaded")

    click = st.button('Detect')

    if click:

        folder_dir = "uploads"
        for images in os.listdir(folder_dir):
            img = PIL.Image.open(f'uploads/{images}').convert('L')
            img.save(f'upload-bw/{images}')

        folder_dir = "upload-bw"
        for images in os.listdir(folder_dir):
            if images.endswith(".png"):
                image = io.imread(f"upload-bw/{images}")
                an_image = PIL.Image.open(f"upload-bw/{images}")
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


        dct = np.array([[entropy, skewness, kurtosis, contrast, energy, homogeneity, dissimilarity, correlation, asm]])

        data = load_model()
        reg = data['model']
        case = reg.predict(dct)

        for images in os.listdir("uploads"):
            os.remove(f'uploads/{images}')
            os.remove(f'upload-bw/{images}')

        if case[0] == 1:
            st.subheader(f'The X-ray image has symptoms of Tuberculosis')
        if case[0] == 0:
            st.subheader(f'The X-ray image does not have symptoms of Tuberculosis')

    # country = st.selectbox("Country", countries)
    # education = st.selectbox("Education Level", educations)
    #
    # experience = st.slider("Years of Experience", 0,50,3)
    # click = st.button('Calculate Salary')
    #
    # if click:
    #     X =np.array([[country, education, experience]])
    #     X[:,0] = le_country.transform(X[:,0])
    #     X[:,1] = le_education.transform(X[:,1])
    #     X = X.astype(float)
    #
    #     salary = reg.predict(X)
    #
    #     st.subheader(f'The Estimated Salary is ${salary[0]:.2f}')
