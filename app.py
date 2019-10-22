from flask import Flask
from utils import load_keras_model

# UPLOAD_FOLDER = '/home/duc_mnsd/Desktop/anhduc/static/images'

app = Flask(__name__)
app.secret_key = "secret key"
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# model, encoder =load_keras_model()

# app.config['MODEL'] =model 
# app.config['ENCODER']= encoder

