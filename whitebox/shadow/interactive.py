#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import resnet
import h5py
import cv2
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Progbar
from PIL import Image
from matplotlib import pyplot as plt

from resnet import ResNet

img_dir = os.path.join(os.getcwd(), 'test_imgs')

model_save_dir = os.path.join(os.getcwd(), 'saved_models')
res_model_save_name = 'resnet_cifar10.hdf5'
res_model_save_path = os.path.join(model_save_dir, res_model_save_name)
shadow_model_save_name = 'shadow_cifar10.hdf5'
shadow_model_save_path = os.path.join(model_save_dir, shadow_model_save_name)
combined_model_save_name = 'shadow_res_cifar10.hdf5'
combined_model_save_path = os.path.join(model_save_dir, combined_model_save_name)

def normalize(x):
  x_min = np.min(x)
  x_max = np.max(x)
  x = (x - x_min) / (x_max - x_min)
  return x
  
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

def get_res_params_from_shadow(res_model, shadow_model, shadow_layers_dict, res_layers_dict):
  weights = {}
  shadow_layer_names = ['dense_11', 'dense_12', 'dense_13', 'predict']

  for shadow_layer_name in shadow_layer_names:
    shadow_layer = shadow_model.get_layer(shadow_layer_name)
    shadow_layer_weights_flatten = []
    shadow_layer_weights = (shadow_layer.trainable_weights)[0].numpy().flatten()
    shadow_layer_bias = (shadow_layer.trainable_weights)[1].numpy().flatten()
    shadow_weights_length = shadow_layer.count_params()
    for weight in shadow_layer_weights:
      shadow_layer_weights_flatten.append(weight)
    for bias in shadow_layer_bias:
      shadow_layer_weights_flatten.append(bias)
    store_start_from = 0
    for tar_res_layer in shadow_layers_dict[shadow_layer_name]:
      cur_params_len = shadow_layers_dict[shadow_layer_name][tar_res_layer]
      half_params_len = int(cur_params_len/2)
      weights[res_layers_dict[tar_res_layer]] = {
        'len': cur_params_len,
        'p1': shadow_layer_weights_flatten[store_start_from:store_start_from+half_params_len],
        'p2': shadow_layer_weights_flatten[store_start_from+half_params_len:store_start_from+cur_params_len],
      }
      store_start_from += cur_params_len

  return weights
   

if __name__ == '__main__':
  if not os.path.exists(img_dir):
    os.mkdir(img_dir)
  if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  #print(x_test)
  #print(x_test)
  #exit(0)

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
  y_train = keras.utils.to_categorical(y_train, num_classes=num_of_classes_val)
  y_test = keras.utils.to_categorical(y_test, num_classes=num_of_classes_val)

  res_model = ResNet(
    input_shape = x_train.shape[1:],
    classes = num_of_classes_val)
  res_optimizer = keras.optimizers.Adam(learning_rate=1e-3)
  res_lossfn = keras.losses.CategoricalCrossentropy()
  res_acc = keras.metrics.CategoricalAccuracy()
  res_eva_lossfn = keras.losses.CategoricalCrossentropy()
  res_eva_acc = keras.metrics.CategoricalAccuracy()
  res_layers_dict = {
    '4d1': 'res_stage_4d_id_block_1_conv2d',
    '4d3': 'res_stage_4d_id_block_3_conv2d', 
    '4a': 'res_stage_4a_conv_block_2_conv2d',
    '4b': 'res_stage_4b_id_block_2_conv2d', 
    '4c': 'res_stage_4c_id_block_2_conv2d',  
    '4e': 'res_stage_4e_id_block_2_conv2d',  
    '4f': 'res_stage_4f_id_block_2_conv2d'  
  }
  #res_model.load_weights(res_model_save_path)

  # Select parameters from the following layers: 
  # 1. res_stage_4d_id_block_1_conv2d   260K
  # 2. res_stage_4d_id_block_3_conv2d   260K
  # 3. res_stage_4a_conv_block_2_conv2d 590K
  # 4. res_stage_4b_id_block_2_conv2d   590K
  # 5. res_stage_4c_id_block_2_conv2d   590K
  # 6. res_stage_4e_id_block_2_conv2d   590K
  # 7. res_stage_4f_id_block_2_conv2d   590K

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
  # Hide params to 4d1(1), 4d3(1), 4a(1), 4b(1/4)+4c(1/4)+4f(1/4)+4e(1/4)

  #w = get_shadow_params_from_res(res_model, shadow_model, shadow_layers_dict, res_layers_dict)
  #print(w)
  #for shadow_layer_name in shadow_layer_names:
  #  shadow_model.get_layer(shadow_layer_name).set_weights(w[shadow_layer_name])
  #  #print(shadow_model.get_layer(shadow_layer_name).trainable_weights)
  #  print(w[shadow_layer_name][0].shape, w[shadow_layer_name][1].shape)

  #w_r = get_res_params_from_shadow(res_model, shadow_model, shadow_layers_dict, res_layers_dict) 
  #for k in w_r:
  #  print(k)
  #  print(w_r[k]['len'])
  #  print(len(w_r[k]['p1']))
  #  print(len(w_r[k]['p2']))
  #  print('---')
  #tmp_sha_l1_weights = (shadow_model.get_layer('dense_11').trainable_weights)[0]
  #print(len(tmp_sha_l1_weights.numpy().flatten()))
  #print(shadow_model.count_params())

  #new_res_weights = get_res_params_from_shadow(res_model, shadow_model, shadow_layers_dict, res_layers_dict)
  #for tmp1_res_layer_name in new_res_weights:
  #  tmp1_res_layer = res_model.get_layer(tmp1_res_layer_name)
  #  tmp1_new_weights = new_res_weights[tmp1_res_layer_name]
  #  tmp1_weights = (tmp1_res_layer.trainable_weights)[0]
  #  tmp1_weights_shape = tmp1_weights.shape
  #  #eva
  #  stored_weights = tmp1_weights.numpy().flatten()
  #  #print(tmp1_weights[0]) 
  #  #print(tmp1_weights[-1]) 
  #  
  #  tmp1_weights = tmp1_weights.numpy().flatten()
  #  update_part_len = int(tmp1_new_weights['len']/2)
  #  for i in range(update_part_len):
  #    tmp1_weights[i] = tmp1_new_weights['p1'][i]
  #    tmp1_weights[-1-i] = tmp1_new_weights['p2'][-1-i]
  #  tmp1_weights = np.asarray(tmp1_weights).reshape(tmp1_weights_shape)
  #  res_model.get_layer(tmp1_res_layer_name).set_weights(
  #    [tmp1_weights, (tmp1_res_layer.trainable_weights)[1].numpy()]
  #  ) 

  #  tmp1_weights = (tmp1_res_layer.trainable_weights)[0].numpy().flatten()
  #  print('*')
  #  #print(tmp1_weights[0]) 
  #  #print(tmp1_weights[-1]) 
  #  #eva
  #  changed_weights = tmp1_weights.flatten()
  #  difference = changed_weights-stored_weights
  #  print(len(difference) - difference.tolist().count(0))
  #  print('---next layer')
  #exit(0)

  shadow_train_size = 800
  shadow_img_at_same_unit = 4
  shadow_x_train = np.zeros((shadow_train_size, shadow_input_dim), dtype='float32')
  for i in range(shadow_x_train.shape[0]):
    shadow_x_train[i][int(i/shadow_img_at_same_unit)] = (i%shadow_img_at_same_unit)*10+1

  shadow_y_train = gray(x_train[:shadow_train_size])
  shadow_y_train = (shadow_y_train / 255).reshape(shadow_train_size, 1024)

  #res_epochs = 100
  res_epochs = 20
  res_batch_size = 256
  shadow_epochs = 20000
  shadow_epochs_in_res_epoch = math.ceil(shadow_epochs/res_epochs)
  shadow_batch_size = 100

  x_train = tf.cast(x_train / 255, dtype='float32') 
  x_test = tf.cast(x_test / 255, dtype='float32') 
  res_train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  res_test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  batched_res_test_dataset = res_test_dataset.shuffle(buffer_size=1024).batch(1024)
  shadow_train_dataset = tf.data.Dataset.from_tensor_slices((shadow_x_train, shadow_y_train))
  total_shadow_epoch = 0

  #run_shadow = False
  run_shadow = True
  for res_epoch in range(res_epochs):
    print("Epoch: %s/%s" % (res_epoch, res_epochs))
    res_progbar = Progbar(x_train.shape[0])
    res_progbar.update(0, values=[('loss', 0.0), ('acc', 0.0)])
    batched_res_train_dataset = res_train_dataset.shuffle(buffer_size=1024).batch(res_batch_size)

    for step, (x_batch_train, y_batch_train) in enumerate(batched_res_train_dataset):
      #x_batch_train = normalize(x_batch_train)
      #x_batch_train = tf.cast(x_batch_train, dtype='float32') 
      with tf.GradientTape() as tape:
        res_logits = res_model(x_batch_train) 
        res_acc.update_state(y_batch_train, res_logits)
        res_acc_val = res_acc.result().numpy()
        res_loss_val = res_lossfn(y_batch_train, res_logits)
        res_grads = tape.gradient(res_loss_val, res_model.trainable_weights)
        res_optimizer.apply_gradients(zip(res_grads, res_model.trainable_weights))
        # print(res_loss_val, res_acc_val)
        # if step % 10 == 0:
        #   print('Training loss at epoch %s step %s is %s' 
        #     % (res_epoch, step, float(res_loss_val)))
      res_progbar.update((step+1)*res_batch_size, values=[('loss', res_loss_val), ('acc', res_acc_val)])
    #res_model.save(res_model_save_path)
    res_model.save_weights(res_model_save_path)
    print("")

    res_eva_acc_val = 0.0
    res_eva_loss_val = 0.0
    res_eva_steps = 0
    for step, (x_batch_test, y_batch_test) in enumerate(batched_res_test_dataset):
      #x_batch_train = normalize(x_batch_train)
      #x_batch_test = tf.cast(x_batch_test, dtype='float32') 
      res_eva_steps += 1
      res_test_logits = res_model(x_batch_test) 
      res_eva_acc.update_state(y_batch_test, res_test_logits)
      res_eva_acc_val = res_eva_acc.result().numpy()
      res_eva_loss_val = res_eva_lossfn(y_batch_test, res_test_logits)
      # print(res_eva_loss_val, res_eva_acc_val)
      # if step % 10 == 0:
      #   print('Training loss at epoch %s step %s is %s' 
      #     % (res_epoch, step, float(res_loss_val)))
    res_eva_acc_val = res_eva_acc_val / res_eva_steps
    res_eva_loss_val = res_eva_loss_val / res_eva_steps
    print("Res eva in epoch %s, loss %s, acc %s." % (res_epoch, res_eva_acc_val, res_eva_loss_val))

    if run_shadow:
      # Apply res weights to shadow model
      new_shadow_weights = get_shadow_params_from_res(res_model, shadow_model, shadow_layers_dict, res_layers_dict)

      for shadow_layer_name in shadow_layer_names:
        shadow_model.get_layer(shadow_layer_name).set_weights(new_shadow_weights[shadow_layer_name])

      # Pre-evaluation using the new weights
      pre_shadow_loss_val = 0.0
      pre_steps = 0
      batched_shadow_train_dataset = shadow_train_dataset.shuffle(buffer_size=1024).batch(100)
      for step, (x_batch_train, y_batch_train) in enumerate(batched_shadow_train_dataset):
        pre_steps += 1
        pre_shadow_logits = shadow_model(x_batch_train)
        pre_shadow_loss_val += float(shadow_lossfn(y_batch_train, pre_shadow_logits))
        print(pre_shadow_loss_val)
      print("Pre-evaluation for shadow model - loss: %s, MAPE: %s" % (pre_shadow_loss_val/pre_steps, MAPE(shadow_train_size, shadow_model(shadow_x_train), gray(x_train*255))))

      for shadow_epoch in range(shadow_epochs_in_res_epoch):
        total_shadow_epoch += 1
        batched_shadow_train_dataset = shadow_train_dataset.shuffle(buffer_size=1024).batch(shadow_batch_size)
        shadow_loss_val = 0.0

        for step, (x_batch_train, y_batch_train) in enumerate(batched_shadow_train_dataset):
          #print(y_batch_train.shape)
          with tf.GradientTape() as tape:
            shadow_logits = shadow_model(x_batch_train)
            shadow_loss_val = shadow_lossfn(y_batch_train, shadow_logits)
            shadow_grads = tape.gradient(shadow_loss_val, shadow_model.trainable_weights)
            shadow_optimizer.apply_gradients(zip(shadow_grads, shadow_model.trainable_weights))
        shadow_model.save(shadow_model_save_path)
        if (total_shadow_epoch) % 100 == 0:
          print('Shadow training loss at epoch %s is %s' % (shadow_epoch, float(shadow_loss_val)))
      # Apply shadow weights back to the res model
      new_res_weights = get_res_params_from_shadow(res_model, shadow_model, shadow_layers_dict, res_layers_dict)
      for tmp1_res_layer_name in new_res_weights:
        tmp1_res_layer = res_model.get_layer(tmp1_res_layer_name)
        tmp1_new_weights = new_res_weights[tmp1_res_layer_name]
        tmp1_weights = (tmp1_res_layer.trainable_weights)[0]
        tmp1_weights_shape = tmp1_weights.shape
        
        tmp1_weights = tmp1_weights.numpy().flatten()
        update_part_len = int(tmp1_new_weights['len']/2)
        for i in range(update_part_len):
          tmp1_weights[i] = tmp1_new_weights['p1'][i]
          tmp1_weights[-1-i] = tmp1_new_weights['p2'][-1-i]
        tmp1_weights = np.asarray(tmp1_weights).reshape(tmp1_weights_shape)
        res_model.get_layer(tmp1_res_layer_name).set_weights(
          [tmp1_weights, (tmp1_res_layer.trainable_weights)[1].numpy()]
        ) 

      # Save res model
      res_model.save_weights(res_model_save_path)
      # Evaluation on res model with the new weights
      batched_res_train_dataset = res_train_dataset.shuffle(buffer_size=1024).batch(res_batch_size)
      af_res_acc = keras.metrics.CategoricalAccuracy()
      af_res_acc_val = 0.0
      af_res_loss_val = 0.0
      af_res_eva_steps = 0
      for step, (x_batch_train, y_batch_train) in enumerate(batched_res_train_dataset):
        af_res_eva_steps += 1
        with tf.GradientTape() as tape:
          af_res_logits = res_model(x_batch_train) 
          af_res_acc.update_state(y_batch_train, af_res_logits)
          af_res_acc_val += res_acc.result().numpy()
          af_res_loss_val += float(res_lossfn(y_batch_train, af_res_logits))
      print("Double-evaluation on res model - loss: %s, acc: %s" % (af_res_loss_val/af_res_eva_steps, af_res_acc_val/af_res_eva_steps))
