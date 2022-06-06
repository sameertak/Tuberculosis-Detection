# # importing the required libraries
# import os
# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename, redirect
# import time
#
# # initialising the flask app
# app = Flask(__name__)
#
# # Creating the upload folder
# upload_folder = "uploads/"
# if not os.path.exists(upload_folder):
#     os.mkdir(upload_folder)
#
# # Max size of the file
# app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
#
# # Configuring the upload folder
# app.config['UPLOAD_FOLDER'] = upload_folder
#
# # configuring the allowed extensions
# allowed_extensions = ['jpg', 'png']
#
#
# def check_file_extension(filename):
#     return filename.split('.')[-1] in allowed_extensions
#
#
# # The path for uploading the file
# @app.route('/')
# def upload_file():
#     return render_template('upload.html')
#
#
# @app.route('/upload', methods=['GET', 'POST'])
# def uploadfile():
#     if request.method == 'POST':  # check if the method is post
#         f = request.files['file']  # get the file from the files object
#         # Saving the file in the required destination
#         if check_file_extension(f.filename):
#             f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))  # this will secure the file
#             return 'file uploaded successfully'  # Display this message after uploading
#         else:
#             return redirect(upload_file())
import PIL.Image
import flask
import pickle
import pandas as pd
from flask import request
import os
from skimage.feature import graycomatrix, graycoprops
from skimage import io
from werkzeug.utils import secure_filename
import numpy as np
import scipy.stats
import skimage.measure

with open(f'tb_dataset.pkl', 'rb') as f:
    model = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')

upload_folder = "uploads/"

if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

allowed_extensions = ['jpg', 'png']


def check_file_extension(filename):
    return filename.split('.')[-1] in allowed_extensions

@app.route('/upload', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')
    if flask.request.method == 'POST':
        f = request.files['file']  # get the file from the files object
                # Saving the file in the required destination
        if check_file_extension(f.filename):
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))  # this will secure the file

            folder_dir = "uploads"
            for images in os.listdir(folder_dir):
                img = PIL.Image.open(f'uploads/{images}').convert('L')
                img.save(f'uploads_bw/{images}')

            folder_dir = "uploads_bw"
            for images in os.listdir(folder_dir):
                if images.endswith(".png"):
                    image = io.imread(f"uploads_bw/{images}")
                    an_image = PIL.Image.open(f"uploads_bw/{images}")
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

            input_variables = pd.DataFrame([[energy, homogeneity, dissimilarity, correlation, asm, contrast, entropy, skewness, kurtosis]],
                                           columns=['energy', 'homogeneity', 'dissimilarity', 'correlation', 'asm', 'contrast', 'entropy', 'skewness', 'kurtosis'],
                                           dtype=float)
            prediction = model.predict(input_variables)[0]
            return flask.render_template('main.html',
                                         original_input={'energy': energy,
                                                         'homogeneity': homogeneity,
                                                         'dissimilarity': dissimilarity,
                                                         'correlation': correlation,
                                                         'asm': asm,
                                                         'contrast': contrast,
                                                         'entropy': entropy,
                                                         'skewness': skewness,
                                                         'kurtosis': kurtosis},
                                         result=prediction,
                                         )

        else:
            return 'Error'