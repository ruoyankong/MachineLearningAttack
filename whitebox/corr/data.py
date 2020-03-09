#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def load_data(name = 'cifar', classes = 2): 
  train_data = ()
  test_data = ()

  if name == 'cifar':
    train_data, test_data = cifar10.load_data()
    train_data = (train_data[0],to_categorical(train_data[1], classes))
    test_data = (test_data[0], to_categorical(test_data[1], classes))

  return train_data, test_data
