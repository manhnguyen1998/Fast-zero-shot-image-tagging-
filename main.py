#from utils import generate_random_start, generate_from_seed
import os
from app import app
import urllib3.request
from utils import get_tag, load_keras_model
from keras.models import load_model
import tensorflow as tf
from flask import Flask, render_template, request, redirect, flash
from wtforms import Form, TextField, validators, SubmitField, DecimalField, IntegerField
from werkzeug.utils import secure_filename
from keras import backend as K
# import model
import h5py,numpy as np

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/images')
# MODEL_DIR = os.path.join(BASE_DIR, 'model')
# model = app.config['MODEL']
# encoder= app.config['ENCODER']

# Create app

semantic_mat_r=h5py.File('291labels.mat','r')
semantic_mat=semantic_mat_r.get('semantic_mat')
semantic_mat=np.array(semantic_mat).astype("float32")

embedding_matrix_norm = np.load('embedding_matrix_norm.npy')
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route("/", methods=['GET', 'POST'])
def home():
    K.clear_session()
    # global model, encoder
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            flash('File successfully uploaded')
            load_keras_model()
            global model, encoder
            global semantic_mat,embedding_matrix_norm
            label_iaprtc12 = get_tag(filepath)
            # print(kq)

            response = {}
            response['path'] = 'static/images/'+filename
            response['text'] = label_iaprtc12
            return render_template('index.html', response=response)

        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    #load_keras_model()
    # Run app
    app.run(host="0.0.0.0", port=8081, debug=False )