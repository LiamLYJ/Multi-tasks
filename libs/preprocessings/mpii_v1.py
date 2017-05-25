#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import libs.configs.config_v1 as cfg
from . import utils as preprocess_utils

FLAGS = tf.app.flags.FLAGS

def preprocess_image(image, gt_boxes, gt_masks, gt_body_masks,is_training=False,locref_map=None,locref_mask=None):
    """preprocess image for coco
    1. random flipping
    2. min size resizing
    3. zero mean
    4. ...
    """
    if not FLAGS.use_refine:
        if is_training:
            return preprocess_for_training(image, gt_boxes, gt_masks,gt_body_masks)
        else:
            return preprocess_for_test(image, gt_boxes, gt_masks,gt_body_masks)
    else:
        if is_training:
            return preprocess_for_training_refine(image,gt_boxes,gt_masks,gt_body_masks,locref_map,locref_mask)
        else:
            return preprocess_for_test_refine(image,gt_boxes,gt_masks,gt_body_masks,locref_map,locref_mask)

def preprocess_for_training(image, gt_boxes, gt_masks,gt_body_masks):

    ih, iw = tf.shape(image)[0], tf.shape(image)[1]
    ## random flipping
    coin = tf.to_float(tf.random_uniform([1]))[0]
    image, gt_boxes, gt_masks,gt_body_masks= \
            tf.cond(tf.greater_equal(coin, 0.5),
                    lambda: (preprocess_utils.flip_image(image),
                            preprocess_utils.flip_gt_boxes(gt_boxes, ih, iw),
                            preprocess_utils.flip_gt_masks_mpii(gt_masks),
                            preprocess_utils.flip_gt_masks(gt_body_masks)
                            ),
                    lambda: (image, gt_boxes, gt_masks,gt_body_masks))

    ## min size resizing
    # new_ih, new_iw = preprocess_utils._smallest_size_at_least(ih, iw, cfg.FLAGS.image_min_size)
    # image = tf.expand_dims(image, 0)
    # image = tf.image.resize_bilinear(image, [new_ih, new_iw], align_corners=False)
    # image = tf.squeeze(image, axis=[0])

    gt_body_masks = tf.expand_dims(gt_body_masks,-1)
    gt_body_masks = tf.cast(gt_body_masks,tf.float32)
    gt_body_masks = tf.image.resize_nearest_neighbor(gt_body_masks,[ih,iw],align_corners = False)
    gt_body_masks = tf.cast(gt_body_masks,tf.int32)
    gt_body_masks = tf.squeeze(gt_body_masks,axis = [-1])

    gt_masks = tf.expand_dims(gt_masks,0)
    gt_masks = tf.cast(gt_masks,tf.float32)

    gt_masks = tf.image.resize_nearest_neighbor(gt_masks,[tf.div(ih,8),tf.div(iw,8)],align_corners = False)
    gt_masks = tf.cast(gt_masks,tf.int32)
    # gt_masks = tf.squeeze(gt_masks,axis = [0])


    # scale_ratio = tf.to_float(new_ih) / tf.to_float(ih)
    # gt_boxes = preprocess_utils.resize_gt_boxes(gt_boxes, scale_ratio)

    # gt_masks = preprocess_utils.resize_gt_mask_mpii(gt_masks,scale_ratio)

    ## zero mean image
    image = tf.cast(image, tf.float32)
    image = image / 256.0
    image = (image - 0.5) * 2.0
    image = tf.expand_dims(image, axis=0)

    ## rgb to bgr
    image = tf.reverse(image, axis=[-1])

    return image, gt_boxes, gt_masks, gt_body_masks


def preprocess_for_training_refine(image, gt_boxes, gt_masks,gt_body_masks,locref_map,locref_mask):

    ih, iw = tf.shape(image)[0], tf.shape(image)[1]
    ## random flipping
    coin = tf.to_float(tf.random_uniform([1]))[0]
    image, gt_boxes, gt_masks,gt_body_masks,locref_map,locref_mask= \
            tf.cond(tf.greater_equal(coin, 0.5),
                    lambda: (preprocess_utils.flip_image(image),
                            preprocess_utils.flip_gt_boxes(gt_boxes, ih, iw),
                            preprocess_utils.flip_gt_masks_mpii(gt_masks),
                            preprocess_utils.flip_gt_masks(gt_body_masks),
                            preprocess_utils.flip_gt_masks(locref_map),
                            preprocess_utils.flip_gt_masks(locref_mask)
                            ),
                    lambda: (image, gt_boxes, gt_masks,gt_body_masks,locref_map,locref_mask))

    ## min size resizing
    # new_ih, new_iw = preprocess_utils._smallest_size_at_least(ih, iw, cfg.FLAGS.image_min_size)
    # image = tf.expand_dims(image, 0)
    # image = tf.image.resize_bilinear(image, [new_ih, new_iw], align_corners=False)
    # image = tf.squeeze(image, axis=[0])

    gt_body_masks = tf.expand_dims(gt_body_masks,-1)
    gt_body_masks = tf.cast(gt_body_masks,tf.float32)
    gt_body_masks = tf.image.resize_nearest_neighbor(gt_body_masks,[ih,iw],align_corners = False)
    gt_body_masks = tf.cast(gt_body_masks,tf.int32)
    gt_body_masks = tf.squeeze(gt_body_masks,axis = [-1])

    gt_masks = tf.expand_dims(gt_masks,0)
    gt_masks = tf.cast(gt_masks,tf.float32)

    gt_masks = tf.image.resize_nearest_neighbor(gt_masks,[tf.div(ih,8),tf.div(iw,8)],align_corners = False)
    gt_masks = tf.cast(gt_masks,tf.int32)
    # gt_masks = tf.squeeze(gt_masks,axis = [0])

    locref_map = tf.expand_dims(locref_map,0)
    locref_map = tf.cast(locref_map,tf.float32)

    locref_map = tf.image.resize_nearest_neighbor(locref_map,[tf.div(ih,8),tf.div(iw,8)],align_corners = False)
    locref_map = tf.cast(locref_map,tf.int32)

    locref_mask = tf.expand_dims(locref_mask,0)
    locref_mask = tf.cast(locref_mask,tf.float32)

    locref_mask = tf.image.resize_nearest_neighbor(locref_mask,[tf.div(ih,8),tf.div(iw,8)],align_corners = False)
    locref_mask = tf.cast(locref_mask,tf.int32)

    # scale_ratio = tf.to_float(new_ih) / tf.to_float(ih)
    # gt_boxes = preprocess_utils.resize_gt_boxes(gt_boxes, scale_ratio)

    # gt_masks = preprocess_utils.resize_gt_mask_mpii(gt_masks,scale_ratio)

    ## zero mean image
    image = tf.cast(image, tf.float32)
    image = image / 256.0
    image = (image - 0.5) * 2.0
    image = tf.expand_dims(image, axis=0)

    ## rgb to bgr
    image = tf.reverse(image, axis=[-1])

    return image, gt_boxes, gt_masks, gt_body_masks,locref_map,locref_mask
