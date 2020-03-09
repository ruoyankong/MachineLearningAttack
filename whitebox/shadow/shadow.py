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

if __name__ == '__main__':
  if not os.path.exists(img_dir):
    os.mkdir(img_dir)
  if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  #print(x_test)
  #print(x_test)
  #exit(0)
  x_train = tf.cast(x_train / 255, dtype='float32') 
  x_test = tf.cast(x_test / 255, dtype='float32') 

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
  res_optimizer = keras.optimizers.Adam(learning_rate=5e-4)
  res_lossfn = keras.losses.CategoricalCrossentropy()
  res_acc = keras.metrics.CategoricalAccuracy()
  res_eva_lossfn = keras.losses.CategoricalCrossentropy()
  res_eva_acc = keras.metrics.CategoricalAccuracy()
  #res_model.load_weights(res_model_save_path)
  res_model.summary()

  shadow_input_dim = 100
  # previous model 128 * 3
  shadow_inputs = keras.Input(shape=(shadow_input_dim,), name='shadow_input')
  shadow_res = layers.Dense(64, activation='relu', name='dense_11')(shadow_inputs)
  #shadow_res = layers.Dense(64, activation='relu', name='dense_12')(shadow_res)
  #shadow_res = layers.Dense(128, activation='relu', name='dense_14')(shadow_res)
  shadow_res = layers.Dense(192, activation='relu', name='dense_13')(shadow_res)
  shadow_outputs = layers.Dense(1024, activation='relu', name='predict')(shadow_res)
  shadow_model = keras.Model(inputs=shadow_inputs, outputs=shadow_outputs, name='shadow')
  shadow_optimizer = keras.optimizers.Adam(learning_rate=1e-3)
  shadow_lossfn = keras.losses.MeanSquaredError()
  #shadow_model.summary()  

  shadow_train_size = 200
  shadow_img_at_same_unit = 2
  shadow_x_train = np.zeros((shadow_train_size, shadow_input_dim), dtype='float32')
  for i in range(shadow_x_train.shape[0]):
    shadow_x_train[i][int(i/shadow_img_at_same_unit)] = (i%shadow_img_at_same_unit)*10+1

  shadow_y_train = gray(x_train[:shadow_train_size])
  shadow_y_train = (shadow_y_train / 255).reshape(shadow_train_size, 1024)

  #res_epochs = 100
  res_epochs = 20
  res_batch_size = 256
  shadow_epochs = 30000
  shadow_epochs_in_res_epoch = math.ceil(shadow_epochs/res_epochs)
  shadow_batch_size = 100

  res_train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  res_test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  batched_res_test_dataset = res_test_dataset.shuffle(buffer_size=1024).batch(1024)
  shadow_train_dataset = tf.data.Dataset.from_tensor_slices((shadow_x_train, shadow_y_train))
  total_shadow_epoch = 0

  run_shadow = False
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
      print(res_eva_loss_val, res_eva_acc_val)
      # if step % 10 == 0:
      #   print('Training loss at epoch %s step %s is %s' 
      #     % (res_epoch, step, float(res_loss_val)))
    #res_eva_acc_val = res_eva_acc_val / res_eva_steps
    #res_eva_loss_val = res_eva_loss_val / res_eva_steps
    #print("Res eva in epoch %s, loss %s, acc %s." % (res_epoch, res_eva_acc_val, res_eva_loss_val))

    if run_shadow:
      for shadow_epoch in range(shadow_epochs_in_res_epoch):
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
        total_shadow_epoch += 1
        if total_shadow_epoch % 1000 == 0:
          print('Shadow training loss at epoch %s is %s' % (shadow_epoch, float(shadow_loss_val)))
