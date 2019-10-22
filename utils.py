import tensorflow as tf
import os
from PIL import Image
import numpy as np
from keras.models import load_model
# import cv2
from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
#                          Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
# from keras.optimizers import Adam, RMSprop
# from keras.layers.wrappers import Bidirectional
# from keras.layers.merge import add
# from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, load_model
# from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

BASE_DIR = os.path.dirname(__file__)
#UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static/images')
MODEL_DIR = os.path.join(BASE_DIR, 'model/best_model.h5')
MODEL_DIR_encode = os.path.join(BASE_DIR, 'model/VGG_encoder.h5')

# max_length= 34
# vocab_size= 1652
# embedding_dim=200


# def preprocess(image_path):
#     # Convert all the images to size 299x299 as expected by the inception v3 model
#     img = image.load_img(image_path, target_size=(299, 299))
#     print("naruto naruto")
#     # Convert PIL image to numpy array of 3-dimensions
#     x = image.img_to_array(img)
#     # Add one more dimension
#     x = np.expand_dims(x, axis=0)
#     # preprocess the images using preprocess_input() from inception module
#     x = preprocess_input(x)
#     return x
def preprocess(image_path):
    # Convert all the images to size 224x224 as expected by the VGG19 model
    img = image.load_img(image_path, target_size=(224, 224))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = VGG_encoder.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def load_keras_model():
    from keras.models import model_from_json
    global VGG_encoder,model_iaprtc12,model_nuswide
    # json_file_iaprtc12 = open('./model/model_im_iaprtc12.json','r')
    # load_model_json_iaprtc12= json_file_iaprtc12.read()
    # json_file_iaprtc12.close()
    # model_iaprtc12 = model_from_json(load_model_json_iaprtc12,custom_objects={'tf': tf})
    # model_iaprtc12.load_weights("./model/best_model.h5")

    json_file_nuswide = open('./model/model_im_nuswide.json','r')
    load_model_json_nuswide= json_file_nuswide.read()
    json_file_nuswide.close()
    model_nuswide = model_from_json(load_model_json_nuswide,custom_objects={'tf': tf})
    model_nuswide.load_weights('./model/best_model.h5')

    VGG_encoder = load_model("./model/VGG_encoder.h5")

 
def get_tag(img):
  feature = encode(img)
  feature = feature.reshape(1,4096)
  # tags = model_iaprtc12.predict(feature)
  tags2 = model_nuswide.predict(feature)
  # top_label = np.argsort(tags)[:,286:]
  top_label2 = np.argsort(tags2)[:,76:]
  # print(top_label2.shape)
  label_nuswide = []
  i = 0
  with open("Concepts81.txt") as f:
      for line in f:
        for j in range(5):
          if i == top_label2[0][j]:
            label_nuswide.append(line)
        i = i+1
  f.close()
  
  # label_iaprtc12 = []
  # i = 0
  # with open("iaprtc12_dictionary.txt") as f:
  #   for line in f:
  #     for j in range(5):
  #       if i == top_label[0][j]:
  #         label_iaprtc12.append(line)
  #     i = i+1
  # f.close()
  return label_nuswide
