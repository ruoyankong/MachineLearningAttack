#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import resnet
import h5py
import cv2
import numpy as np
import tensorflow.keras as keras

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import ImageOps, Image

from data import load_data
from resnet import ResNet

img_dir = os.path.join(os.getcwd(), 'test_imgs')

model_save_dir = os.path.join(os.getcwd(), 'saved_models')
#res_model_save_name = 'saved_res_corr_model.hdf5'
#res_model_save_name = 'saved_only_corr_model.hdf5'
res_model_save_name = 'no_attack_resnet_cifar10.hdf5'
res_model_save_path = os.path.join(model_save_dir, res_model_save_name)

cor_level = 1
#extract_length = 32 * 32 * 3 * 1
extract_shape = (200, 32, 32, 3)
#extract_length = 32 * 32 * 200

def normalize(x):
  shape = x.shape
  x = x.flatten()
  x_min = np.min(x)
  x_max = np.max(x)
  x = (x - x_min) / (x_max - x_min)
  return x.reshape(shape)
  

def extract_params(extract_length, total_params):
  rest_length = extract_length
  params = K.constant([])
  for layer_weights in total_params:
    layer_weights_flat = K.flatten(layer_weights)

    if layer_weights_flat.shape[0] < rest_length:
      params = K.concatenate((params, layer_weights_flat))
    else:
      if rest_length > 0:
        params = K.concatenate((params, layer_weights_flat[:rest_length]))
      break
    rest_length = rest_length - layer_weights_flat.shape[0]

  return params

class Callbacks(keras.callbacks.Callback):
  def __init__(self, _total_weights):
    self.total_weights = _total_weights 
  def on_batch_begin(self, batch, logs):
    sum_length = 200 * 32 * 32
    K.set_value(self.total_weights, 
            extract_params(sum_length, self.model.get_weights()))
  # def on_batch_end(self, batch, logs):
  #   print(logs['loss'])

save_res_model = ModelCheckpoint(
        res_model_save_path,
        monitor='val_acc',
        verbose=1,
        mode='max')


class CorrRes:
  def __init__(
          self, 
          input_shape, 
          classes, 
          epochs, 
          batch_size,
          extract_length = 32 * 32 * 3 * 1,
          extract_shape = (1, 32, 32, 3),
          optimizer = 'adam',
          dataset = 'cifar'):

    self.input_shape = input_shape
    self.classes = classes
    self.model = self.get_model()
    self.epochs = epochs
    self.batch_size = batch_size
    self.extract_length = extract_length
    self.extract_shape = extract_shape
    self.optimizer = optimizer
    self.dataset = dataset

    self.loss_value = None
    self.acc_value = None

    self.train_data, self.test_data = load_data(
            name = self.dataset, classes = self.classes)

    # Original design
    #self.extracted_data = self.train_data[0].flatten()[:self.extract_length]
    self.extracted_data = self.train_data[0].flatten()[:self.extract_length * 3]
    self.extracted_data = rgb_to_grayscale(self.extracted_data.reshape(self.extract_shape)).flatten()
    #self.total_weights = self.model.get_weights()
    self.total_weights = K.variable(
            extract_params(self.extract_length, self.model.get_weights()))

    self.model = self.compile_model()

  def get_model(self):
    self.model = ResNet(
      input_shape = self.input_shape, 
      classes = self.classes)

    return self.model

  def compile_model(self):
    self.model.compile(optimizer = self.optimizer,
                  #loss = self.loss(),
                  loss = keras.losses.categorical_crossentropy,
                  metrics = ['accuracy'])
    return self.model

  def loss(self):
    def loss_function(y_true, y_pred):

      params = K.cast(self.total_weights, dtype='float32')
      target_data = K.cast(self.extracted_data, dtype='float32')
      params_mean = K.mean(params)
      target_mean = K.mean(target_data)
      params_d = params - params_mean
      target_d = target_data - target_mean

      num = K.sum((params_d) * (target_d))
      den = K.sqrt(K.sum(K.square(params_d)) * K.sum(K.square(target_d)))
      co = num / den
      loss_co = 1 - abs(co)

      # print("")
      # K.print_tensor(loss_co, message="params=")
      # K.print_tensor(loss_co, message="data=")

      loss = keras.losses.categorical_crossentropy(y_true, y_pred)
      #K.print_tensor(loss, "loss=")
      #K.print_tensor(loss_co, "loss_co=")
      # K.print_tensor(loss, message="loss=")
      # return K.mean(K.abs(y_true - y_pred))
      # Used loss - abs(co) before
      # return loss + loss_co * cor_level

      #self.loss_value = loss + loss_co * cor_level
      self.loss_value = loss
      #self.loss_value = loss - loss_co * cor_level
      #self.loss_value = loss - abs(co) * cor_level
      #self.loss_value = loss_co * cor_level
      return self.loss_value
    
    return loss_function
 
  def train(self):
    #K.print_tensor(extracted_data, message="fff=")
    # model.add_loss()

    self.model.fit(
            self.train_data[0], 
            self.train_data[1], 
            epochs = self.epochs, 
            batch_size = self.batch_size, 
            callbacks=[Callbacks(self.total_weights), save_res_model])

  def load(self, model_path):
    self.model.load_weights(model_path)
    self.total_weights = K.variable(
            extract_params(self.extract_length, self.model.get_weights()))

  def evaluate(self):
    result = self.model.evaluate(self.test_data[0], self.test_data[1])
    self.loss_value = result[0]
    self.accuracy = result[1]

  def attack_evaluate(self):
    params = K.cast(self.total_weights, dtype='float32')
    target_data = K.cast(self.extracted_data, dtype='float32')
    params_mean = K.mean(params)
    target_mean = K.mean(target_data)
    params_d = params - params_mean
    target_d = target_data - target_mean

    num = K.sum((params_d) * (target_d))
    den = K.sqrt(K.sum(K.square(params_d)) * K.sum(K.square(target_d)))
    co = num / den
    print(params)
    print("Corr: ", co)
    #loss_co = 1 - abs(co)

    img_name = 'test.png'
    img_path = os.path.join(img_dir, img_name)
    param_img_name = 'param_test.png'
    param_img_path = os.path.join(img_dir, param_img_name)

    data_in_params = K.get_value(self.total_weights)
    img_from_params = normalize(data_in_params)
    #img_from_params = (img_from_params * 255).astype(np.uint8)
    img_from_params = (img_from_params * 255)
    #img_from_params = np.asarray(ImageOps.invert(Image.fromarray(img_from_params.reshape(32, 32, 3)))).flatten()
    #img_from_params = rgb_to_grayscale(img_from_params.reshape(32, 32, 3)).flatten().astype(np.uint8)
    #img_from_params = rgb_to_grayscale(img_from_params.reshape(32, 32, 3)).flatten()

    #self.extracted_data = rgb_to_grayscale(self.extracted_data.reshape(32,32,3)).flatten().astype(np.uint8)
    #self.extracted_data = rgb_to_grayscale(self.extracted_data.reshape(32,32,3)).flatten()

    K.print_tensor(img_from_params, "pa_before=")
    K.print_tensor(self.extracted_data, "data_before=")

    #img_from_params = rgb_to_grayscale(img_from_params.reshape(32, 32, 3)).flatten().astype(np.uint16)
    #self.extracted_data = rgb_to_grayscale(self.extracted_data.reshape(32,32,3)).flatten().astype(np.uint16)
    K.print_tensor(img_from_params, "pa_after=")
    K.print_tensor(self.extracted_data, "data_after=")
    
    difference = self.extracted_data - img_from_params
    print("mean:", np.mean(np.abs(difference)))
    print("var:", np.var(np.abs(difference)))

    print("mean_pa, var_pa", np.mean(img_from_params), np.var(img_from_params))
    print("mean, var", np.mean(self.extracted_data), np.var(self.extracted_data))

    #img_from_params = img_from_params.reshape(32, 32, 3)
    #img_from_dataset = self.extracted_data.reshape(32, 32, 3)
    img_from_params = img_from_params.reshape(32, 32)
    img_from_dataset = self.extracted_data.reshape(32, 32)
    cv2.imwrite(param_img_path, img_from_params)
    cv2.imwrite(img_path, img_from_dataset)

def rgb_to_grayscale(images):
    #return images[..., :3] * [0.299, 0.587, 0.114]
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])

if __name__ == '__main__':
  if not os.path.exists(img_dir):
    os.mkdir(img_dir)

  #corr_res = CorrRes((32, 32, 3), 10, 30, 256, extract_length)
  sum_length = 1
  for dim in extract_shape:
    sum_length *= dim
  sum_length /= 3
  corr_res = CorrRes((32, 32, 3), 10, 20, 256, int(sum_length), extract_shape)
  #corr_res.train()
  #corr_res.evaluate()
  corr_res.load(res_model_save_path) 
  corr_res.train()
  #corr_res.evaluate()
  #corr_res.attack_evaluate()
