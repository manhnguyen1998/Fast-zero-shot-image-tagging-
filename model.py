# Thêm thư viện
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, GRU, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
# from keras.layers.wrappers import Bidirectional
# from keras.layers.merge import add
# from keras.applications.inception_v3 import InceptionV3
# from keras.preprocessing import image
# from keras.models import Model
# from keras import Input, layers
# from keras import optimizers
# from keras.applications.inception_v3 import preprocess_input
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical

# max_length= 34

import h5py,numpy as np
import scipy.io
import os
import keras
# import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
import keras.layers as L
from keras.callbacks import ModelCheckpoint
from keras.models import Model,load_model

from keras.preprocessing import image
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Lambda

# import pickle
from keras.applications.vgg19 import VGG19, preprocess_input
# import sklearn

semantic_mat_r=h5py.File('291labels.mat','r')
semantic_mat=semantic_mat_r.get('semantic_mat')
semantic_mat=np.array(semantic_mat).astype("float32")

embedding_matrix_norm = np.load('embedding_matrix_norm.npy')
"""
rank_layer: get label confidence by multiply ingsemtantic embeddings with principle direction generated by previous layers
"""

# global semantic_mat,embedding_matrix_norm
def rank_layer(input_tensor):
    import h5py,numpy as np
    import tensorflow as tf
    semantic_mat_r=h5py.File('291labels.mat','r')
    semantic_mat=semantic_mat_r.get('semantic_mat')
    semantic_mat=np.array(semantic_mat).astype("float32")
    label_predict=tf.tensordot(input_tensor,semantic_mat,axes=1)
    return label_predict

def rank_layer2(input_tensor):
    import h5py,numpy as np
    import tensorflow as tf
    embedding_matrix_norm = np.load('embedding_matrix_norm.npy')
    label_predict=tf.tensordot(input_tensor,embedding_matrix_norm,axes=1)
    return label_predict
  

# Tạo model
def create_model():
  import tensorflow as tf
  model = Sequential()

  model.add(Dense(input_dim=4096, output_dim=8192, init="uniform"))
  model.add(Activation("relu"))
  model.add(Dropout(0.5))
  model.add(Dense(input_dim=8192, output_dim=2048, init="uniform"))
  model.add(Activation("relu"))
  model.add(Dense(input_dim=2048, output_dim=300, init="uniform"))
  model.add(Activation("relu"))
  model.add(Dense(input_dim=300, output_dim=300, init="uniform"))
  model.add(Activation("linear"))
# model.add(Lambda(rank_layer,output_shape=[training_data_target.shape[1]]))
  lambda_layer = Lambda(rank_layer,output_shape=[291])
  model.add(lambda_layer)
  return model


def create_second_model():
  import tensorflow as tf
  model = Sequential()

  model.add(Dense(input_dim=4096, output_dim=8192, init="uniform"))
  model.add(Activation("relu"))
  model.add(Dropout(0.5))
  model.add(Dense(input_dim=8192, output_dim=2048, init="uniform"))
  model.add(Activation("relu"))
  model.add(Dense(input_dim=2048, output_dim=300, init="uniform"))
  model.add(Activation("relu"))
  model.add(Dense(input_dim=300, output_dim=300, init="uniform"))
  model.add(Activation("linear"))
# model.add(Lambda(rank_layer,output_shape=[training_data_target.shape[1]]))
  lambda_layer = Lambda(rank_layer2,output_shape=[embedding_matrix_norm.shape[1]])
  model.add(lambda_layer)
  return model



# model2.load_weights('/content/drive/My Drive/Colab Notebooks/dataset/best_model.h5')

model = create_model()
model2 = create_second_model()
# model.load_weights('/content/drive/My Drive/Colab Notebooks/dataset/best_model.h5')

model_json_iaprtc12= model.to_json()
with open("./model/model_im_iaprtc12.json", "w") as json_file:
    json_file.write(model_json_iaprtc12)

model_json_nuswide= model2.to_json()
with open("./model/model_im_nuswide.json", "w") as json_file:
    json_file.write(model_json_nuswide)
