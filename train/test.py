#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import functools
import os, sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import gmtime, strftime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.configs.config_v1 as cfg
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network

import libs.preprocessings.coco_v1 as coco_preprocess
import libs.nets.pyramid_network as pyramid_network
import libs.nets.resnet_v1 as resnet_v1

from train.train_utils import _configure_learning_rate, _configure_optimizer, \
  _get_variables_to_train, _get_init_fn, get_var_list_to_restore

resnet50 = resnet_v1.resnet_v1_50
FLAGS = tf.app.flags.FLAGS


def restore(sess):
     """choose which param to restore"""
     if FLAGS.restore_previous_if_exists:
        try:
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            restorer = tf.train.Saver()
            restorer.restore(sess, checkpoint_path)
            print ('restored previous model %s from %s'\
                    %(checkpoint_path, FLAGS.train_dir))
            time.sleep(2)
            return
        except:
            print ('--restore_previous_if_exists is set, but failed to restore in %s %s'\
                    % (FLAGS.train_dir, checkpoint_path))
            time.sleep(2)

     if FLAGS.pretrained_model:
        if tf.gfile.IsDirectory(FLAGS.pretrained_model):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model)
        else:
            checkpoint_path = FLAGS.pretrained_model

        if FLAGS.checkpoint_exclude_scopes is None:
            FLAGS.checkpoint_exclude_scopes='pyramid'
        if FLAGS.checkpoint_include_scopes is None:
            FLAGS.checkpoint_include_scopes='resnet_v1_50'

        vars_to_restore = get_var_list_to_restore()
        for var in vars_to_restore:
            print ('restoring ', var.name)

        try:
           restorer = tf.train.Saver(vars_to_restore)
           restorer.restore(sess, checkpoint_path)
           print ('Restored %d(%d) vars from %s' %(
               len(vars_to_restore), len(tf.global_variables()),
               checkpoint_path ))
        except:
           print ('Checking your params %s' %(checkpoint_path))
           raise

def im_detect(im,gt_boxes):
    im = tf.to_float(im)
    im = tf.convert_to_tensor(im)
    im = tf.expand_dims(im, axis = 0)
    im_shape = tf.shape(im)
    ih = im_shape[1]
    iw = im_shape[2]

    logits, end_points,pyramid_maps = network.get_network(FLAGS.network,im)
    outputs = pyramid_network.build(end_points, ih, iw, pyramid_maps,
            num_classes=2,
            base_anchors=9,
            is_training=False,
            gt_boxes= gt_boxes, gt_masks=None,
            gt_body_masks = None)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
            )
    sess.run(init_op)
    restore(sess)

    # pred_masks = tf.nn.sigmoid(outputs['masks']['mask'])
    pred_masks = outputs['masks']['mask']

    # print ('*******************')
    # print (pred_masks)
    return sess.run(pred_masks)

if __name__ == '__main__':
    img_path = '/home/hpc/ssd/lyj/FastMaskRCNN/test_img/2.jpg'
    im = cv2.imread(img_path)

    im = im / 256.0
    im = (im - 0.5) * 2.0

    ih,iw = im.shape[0],im.shape[1]

    gt_boxes  = (10,1,iw-11,ih-11,10)
    pred_masks = im_detect(im,gt_boxes)
    output = np.squeeze(pred_masks)

    # output1 = np.squeeze(pred_masks['mask_1'])
    # output2 = np.squeeze(pred_masks['mask_2'])

    print (output.shape)

    # hp = output[:,:,0]
    # hp2
    #
    # max_value = np.max(hp1)
    # i,j  = np.unravel_index(hp1.argmax(),hp1.shape)
    # print ('i: ',i,'j: ',j)
    # # print(hp1)
    # assert max_value == hp1[i,j]
    # print('max_value: ',max_value)


    for k in range (14):
        print ( 'this is %d joints'%k)
        hp = output[:,:,k]
        max_v = np.max(hp)
        i,j = np.unravel_index(hp.argmax(),hp.shape)
        print ('i: ',i,'j: ',j)
        assert (max_v == hp[i,j])
        print ('max_value: ',max_v)
