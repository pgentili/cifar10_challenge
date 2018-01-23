"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
import numpy as np

from model import Model
from pgd_attack import class_attack_path
import cifar10_input

with open('config.json') as config_file:
    config = json.load(config_file)

data_path = config['data_path']

NUM_CLASSES = 10

def to_prop(mat):
  """Converts 2d array of counts to proportions by dividing over row counts."""
  return mat.astype(np.float64) / mat.sum(axis=1)[:, np.newaxis]

def run_attack(checkpoint, x_adv, epsilon):
  cifar = cifar10_input.CIFAR10Data(data_path)

  model = Model(mode='eval')

  saver = tf.train.Saver()

  num_eval_examples = 10000
  eval_batch_size = 100

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0

  x_nat = cifar.eval_data.xs
  l_inf = np.amax(np.abs(x_nat - x_adv))

  if l_inf > epsilon + 0.0001:
    print('maximum perturbation found: {}'.format(l_inf))
    print('maximum perturbation allowed: {}'.format(epsilon))
    return

  y_pred = [] # label accumulator

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)

    # Iterate over the samples batch-by-batch
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = x_adv[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch}
      cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],
                                        feed_dict=dict_adv)

      total_corr += cur_corr
      y_pred.append(y_pred_batch)

  accuracy = total_corr / num_eval_examples

  print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
  y_pred = np.concatenate(y_pred, axis=0)
  np.save('pred.npy', y_pred)
  print('Output saved at pred.npy')

def run_class_attack(checkpoint, x_adv_list, epsilon):
  cifar = cifar10_input.CIFAR10Data(data_path)

  model = Model(mode='eval')

  saver = tf.train.Saver()

  num_eval_examples = 10000
  eval_batch_size = 100

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

  comb_conf_mat = np.zeros([NUM_CLASSES, NUM_CLASSES], dtype=np.float64)
  # Number of predictable classes within epsilon of each eval example
  dist_preds = np.zeros(num_eval_examples, dtype=np.int32)

  x_nat = cifar.eval_data.xs

  for x_adv in x_adv_list:
    l_inf = np.amax(np.abs(x_nat - x_adv))

    if l_inf > epsilon + 0.0001:
      print('maximum perturbation found: {}'.format(l_inf))
      print('maximum perturbation allowed: {}'.format(epsilon))
      return

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)

    # Iterate over each class's attack dataset
    for (i, x_adv) in enumerate(x_adv_list):
      # Start with one confusion matrix for each class
      conf_mat = np.zeros([NUM_CLASSES, NUM_CLASSES], dtype=np.int32)

      # Iterate over the samples batch-by-batch
      for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = x_adv[bstart:bend, :]
        y_batch = cifar.eval_data.ys[bstart:bend]

        dict_adv = {model.x_input: x_batch,
                    model.y_input: y_batch}
        y_pred_batch, conf_mat_batch = sess.run([model.predictions,
                                                 model.conf_mat],
                                                feed_dict=dict_adv)
        conf_mat += conf_mat_batch
        dist_preds[bstart:bend] += (y_pred_batch == i)

      # Divide by counts of each class
      conf_mat = to_prop(conf_mat)

      # Take i-th column of i-th confusion matrix, corresponding to proportion
      # of each class that gets classified as i during the class i attack
      comb_conf_mat[:, i] = conf_mat[:, i]
  
  reachable_mat = np.zeros([NUM_CLASSES, NUM_CLASSES + 1], dtype=np.int32)
  for (i, true_class) in enumerate(cifar.eval_data.ys):
      reachable_mat[true_class, dist_preds[i]] += 1
  reachable_mat = to_prop(reachable_mat)

  return comb_conf_mat, dist_preds, reachable_mat

def run_class_attack_ext(model_dir, adv_path, epsilon):
  """e.g. for use in jupyter notebooks to avoid config file."""
  checkpoint = tf.train.latest_checkpoint(model_dir)
  x_adv_list = []

  for i in range(NUM_CLASSES):
    path = class_attack_path(adv_path, i)
    x_adv = np.load(path)
    x_adv_list.append(x_adv)

  return run_class_attack(checkpoint, x_adv_list, epsilon)


if __name__ == '__main__':
  import json

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_dir = config['model_dir']

  checkpoint = tf.train.latest_checkpoint(model_dir)
  x_adv = np.load(config['store_adv_path'])

  if checkpoint is None:
    print('No checkpoint found')
  elif x_adv.shape != (10000, 32, 32, 3):
    print('Invalid shape: expected (10000, 32, 32, 3), found {}'.format(x_adv.shape))
  elif np.amax(x_adv) > 255.0001 or np.amin(x_adv) < -0.0001:
    print('Invalid pixel range. Expected [0, 255], found [{}, {}]'.format(
                                                              np.amin(x_adv),
                                                              np.amax(x_adv)))
  else:
    run_attack(checkpoint, x_adv, config['epsilon'])
