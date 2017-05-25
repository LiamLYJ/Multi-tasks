#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.logs.log import LOG
import libs.configs.config_v1 as cfg
import libs.nets.resnet_v1 as resnet_v1
import libs.datasets.dataset_factory as dataset_factory
import libs.datasets.mpii as mpii
import libs.preprocessings.mpii_v1 as preprocess_mpii
from libs.layers import ROIAlign

resnet50 = resnet_v1.resnet_v1_50
FLAGS = tf.app.flags.FLAGS

with tf.Graph().as_default():
  image,ih,iw,gt_boxes,gt_masks,gt_body_masks,num_masks,image_id,locref_map,locref_mask = \
    mpii.read_refine('/home/hpc/ssd/lyj/FastMaskRCNN/data/lsp/records/lsp_lsp_train_00000-of-00004.tfrecord')

  # image, gt_boxes, gt_masks, gt_body_masks,locref_map,locref_mask = \
  #   preprocess_mpii.preprocess_image(image, gt_boxes, gt_masks, gt_body_masks,locref_map=locref_map,locref_mask=locref_mask)

  sess = tf.Session()
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())


  # boxes = [[100, 100, 200, 200],
  #          [50, 50, 100, 100],
  #          [100, 100, 750, 750],
  #          [50, 50, 60, 60]]
  #
  # boxes = tf.constant(boxes, tf.float32)
  # feat = ROIAlign(image, boxes, False, 16, 7, 7)
  sess.run(init_op)

  tf.train.start_queue_runners(sess=sess)
  with sess.as_default():
      for i in range(10):
        # image_np, ih_np, iw_np, gt_boxes_np, gt_masks_np, num_instances_np, img_id_np, \
        # feat_np = \
        #     sess.run([image, ih, iw, gt_boxes, gt_masks, num_instances, img_id,
        #         feat])
        # print (image_np.shape, gt_boxes_np.shape, gt_masks_np.shape)
        #
        # if i % 100 == 0:
        #     print ('%d, image_id: %s, instances: %d'%  (i, str(img_id_np), num_instances_np))
        #     image_np = 256 * (image_np * 0.5 + 0.5)
        #     image_np = image_np.astype(np.uint8)
        #     image_np = np.squeeze(image_np)
        #     print (image_np.shape, ih_np, iw_np)
        #     print (feat_np.shape)
        #     im = Image.fromarray(image_np)
        #     imd = ImageDraw.Draw(im)
        #     for i in range(gt_boxes_np.shape[0]):
        #         imd.rectangle(gt_boxes_np[i, :])
        #     im.save(str(img_id_np) + '.png')
        #
        image,ih,iw,gt_boxes,gt_masks,gt_body_masks,num_masks,image_id,locref_map,locref_mask = \
                sess.run([image,ih,iw,gt_boxes,gt_masks,gt_body_masks,num_masks,image_id,locref_map,locref_mask])

        print ('ih: ',ih)
        print ('iw: ',iw)
        print('img shape is :',image.shape)
        print ('gt_boxes: ',gt_boxes)
        print ('gt_mask shape is ',gt_masks.shape)
        print ('gt_body_masks shape is ',gt_body_masks.shape)
        print ('num_masks ', num_masks)
        print ('locref_map shape is: ',locref_map.shape)
        print ('locref_mask shape is: ',locref_mask.shape)

        body = gt_body_masks[0,:,:].astype(float)
        # keep = np.where(body == 0)
        # print ('howmany is 0 : ',len(keep[0]))
        # raise
        # cv2.imshow('body_mask',body)
        # cv2.imshow('image',image)

        for i in range(14):
            j = gt_masks[:,:,i].astype(float)
            cv2.imshow('j%d'%i,j)
        # for i in range(28):
        #     locref = locref_mask[:,:,i].astype(float)
        #     cv2.imshow('locref%d'%i,locref)
        # for i in range(28):
        #     loc_map = locref_map[:,:,i]
        #     cv2.imshow('loc_map%d'%i,loc_map)
        # cv2.waitKey(0)


        cv2.imshow('body_mask',body)
        cv2.imshow('image',image)
        cv2.waitKey(0)
        raise
  sess.close()
