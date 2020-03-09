#!/usr/bin/env python

import os
import h5py
from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from tensorflow.keras import layers
from resnet import ResNet

def shadow_layers_weights_distr(model):
  ret = {}

  l1 = model.get_layer('dense_11')
  l2 = model.get_layer('dense_12')
  l3 = model.get_layer('dense_13')
  l4 = model.get_layer('predict')

  ret['dense_11'] = {'4d1': l1.count_params()}
  ret['dense_12'] = {'4d3': l2.count_params()}
  ret['dense_13'] = {'4a': l3.count_params() }

  l4_params = int(l4.count_params()/4)
  ret['predict'] = {'4b': l4_params, '4c': l4_params, '4e': l4_params, '4f': l4_params}

  return ret


def get_shadow_params_from_res(res_model, shadow_model, shadow_layers_dict, res_layers_dict):
  weights = {} 
  shadow_layer_names = ['dense_11', 'dense_12', 'dense_13', 'predict']

  for shadow_layer_name in shadow_layer_names:
    shadow_layer = shadow_model.get_layer(shadow_layer_name)
    shadow_weight_shape = (shadow_layer.trainable_weights)[0].shape
    shadow_weight_length = np.prod(shadow_weight_shape)
    shadow_bias_shape = (shadow_layer.trainable_weights)[1].shape
    shadow_bias_length = np.prod(shadow_bias_shape)
    shadow_layer_dict = shadow_layers_dict[shadow_layer_name]
    shadow_layer_weights_flatten = []
    for dict_key in shadow_layer_dict:
      res_layer_name = res_layers_dict[dict_key]
      res_layer = res_model.get_layer(res_layer_name)
      res_layer_weight = (res_layer.trainable_weights)[0].numpy().flatten()
      res_lw_len = len(res_layer_weight)
      extract_len = int(shadow_layer_dict[dict_key]/2)
      for weight in res_layer_weight[:extract_len]:
        shadow_layer_weights_flatten.append(weight)
      for weight in res_layer_weight[res_lw_len-extract_len:]:
        shadow_layer_weights_flatten.append(weight)

    shadow_lw_p1 = np.asarray(shadow_layer_weights_flatten[:shadow_weight_length]).reshape(shadow_weight_shape)
    shadow_lw_p2 = np.asarray(shadow_layer_weights_flatten[shadow_weight_length:]).reshape(shadow_bias_shape)
    weights[shadow_layer_name] = [shadow_lw_p1, shadow_lw_p2]
    #weights[shadow_layer_name] = [tf.convert_to_tensor(shadow_lw_p1), tf.convert_to_tensor(shadow_lw_p2]
   
  return weights

def gray(images):
  return np.dot(images[..., :3], [0.299, 0.587, 0.114])

def shadow_img_fn(index, shadow_result):
  return (shadow_result[index]*255).numpy()

def train_img_fn(index, x_train):
  return x_train[index].reshape(1024)

def MAPE_index(index, shadow_result, x_train):
  img1 = shadow_img_fn(index, shadow_result)
  img2 = train_img_fn(index, x_train)
  return np.mean(abs(img1-img2))

def MAPE(train_size, shadow_result, x_train):
  mean_sum = 0.0
  for i in range(train_size):
    mean_sum += MAPE_index(i, shadow_result, x_train)
  return mean_sum/train_size

model_save_dir = os.path.join(os.getcwd(), 'saved_models')
shadow_model_save_name = 'shadow_cifar10.hdf5'
shadow_model_save_path = os.path.join(model_save_dir, shadow_model_save_name)
res_model_save_name = 'resnet_cifar10.hdf5'
res_model_save_path = os.path.join(model_save_dir, res_model_save_name)

if __name__ == '__main__':
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

  def num_of_classes(train, test):
    l = []
    for label in train:
      if label not in l:
        l.append(label)
    for label in test:
      if label not in l:
        l.append(label)
    return len(l)
  num_of_classes_val = num_of_classes(y_train, y_test)
  res_model = ResNet(
    input_shape = x_train.shape[1:],
    classes = num_of_classes_val)

  x_train = gray(x_train)

  shadow_input_dim = 200
  # previous model 128 * 3
  shadow_layer_names = ['dense_11', 'dense_12', 'dense_13', 'predict']
  shadow_inputs = keras.Input(shape=(shadow_input_dim,), name='shadow_input')
  shadow_res = layers.Dense(128, activation='relu', name='dense_11')(shadow_inputs)
  shadow_res = layers.Dense(256, activation='relu', name='dense_12')(shadow_res)
  shadow_res = layers.Dense(256, activation='relu', name='dense_13')(shadow_res)
  shadow_outputs = layers.Dense(1024, activation='relu', name='predict')(shadow_res)
  shadow_model = keras.Model(inputs=shadow_inputs, outputs=shadow_outputs, name='shadow')
  shadow_optimizer = keras.optimizers.Adam(learning_rate=1e-3)
  shadow_lossfn = keras.losses.MeanSquaredError()
  shadow_model.summary()  
  shadow_layers_dict = shadow_layers_weights_distr(shadow_model)

  #shadow_model.load_weights(shadow_model_save_path)

  shadow_train_size = 800
  shadow_img_at_same_unit = 4
  shadow_x_train = np.zeros((shadow_train_size, shadow_input_dim), dtype='float32')
  for i in range(shadow_x_train.shape[0]):
    shadow_x_train[i][int(i/shadow_img_at_same_unit)] = (i%shadow_img_at_same_unit)*10+1

  #shadow_res = shadow_model(shadow_x_train) 
  #mape = MAPE(shadow_train_size, shadow_res, x_train)
  #print("MAPE: ", mape)

  #plt.figure(figsize=(1,1))
  #plt.imshow(shadow_img_fn(1, shadow_res).reshape(32, 32), cmap='gray')
  #plt.show()
     

  # Get params from res model
  
  res_optimizer = keras.optimizers.Adam(learning_rate=1e-3)
  res_lossfn = keras.losses.CategoricalCrossentropy()
  res_acc = keras.metrics.CategoricalAccuracy()
  res_layers_dict = {
    '4d1': 'res_stage_4d_id_block_1_conv2d',
    '4d3': 'res_stage_4d_id_block_3_conv2d', 
    '4a': 'res_stage_4a_conv_block_2_conv2d',
    '4b': 'res_stage_4b_id_block_2_conv2d', 
    '4c': 'res_stage_4c_id_block_2_conv2d',  
    '4e': 'res_stage_4e_id_block_2_conv2d',  
    '4f': 'res_stage_4f_id_block_2_conv2d'  
  }
  res_model.load_weights(res_model_save_path)

  new_shadow_weights = get_shadow_params_from_res(res_model, shadow_model, shadow_layers_dict, res_layers_dict)
  for shadow_layer_name in shadow_layer_names:
    shadow_model.get_layer(shadow_layer_name).set_weights(new_shadow_weights[shadow_layer_name])

  shadow_res = shadow_model(shadow_x_train) 
  mape = MAPE(shadow_train_size, shadow_res, x_train)
  print("MAPE: ", mape)

  #plt.figure(figsize=(1,1))
  #plt.imshow(shadow_img_fn(1, shadow_res).reshape(32, 32), cmap='gray')
  #plt.show()

  row = 5
  col = 5

  f = plt.figure()
  for i in range(row):
    for j in range(col):
      index = i*col+j
      cur_img = f.add_subplot(row, col, index+1)
      plt.imshow(shadow_img_fn(index, shadow_res).reshape(32, 32), cmap='gray')

  plt.show()
