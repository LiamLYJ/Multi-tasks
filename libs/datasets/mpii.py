from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

FLAGS = tf.app.flags.FLAGS

def read(tfrecords_filename):
    if not isinstance(tfrecords_filename,list):
        tfrecords_filename = [tfrecords_filename]
    filename_queue = tf.train.string_input_producer(tfrecords_filename,num_epochs = 100)

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features ={
            'image/img_id': tf.FixedLenFeature([],tf.int64),
            'image/encoded': tf.FixedLenFeature([],tf.string),
            'image/height': tf.FixedLenFeature([],tf.int64),
            'image/width': tf.FixedLenFeature([],tf.int64),
            'label/gt_boxes': tf.FixedLenFeature([],tf.string),  # of shape (1, 5), (x1, y1, x2, y2,1)
            'label/gt_masks': tf.FixedLenFeature([],tf.string),  # of shape (hh,hw,14 )
            'label/num_masks':tf.FixedLenFeature([],tf.int64), # num_of valid joints
            'label/gt_body_masks': tf.FixedLenFeature([],tf.string)  #  of (1,ih,iw)
        }
    )
    image_id = tf.cast(features['image/img_id'],tf.int32)
    ih = tf.cast(features['image/height'],tf.int32)
    iw = tf.cast(features['image/width'],tf.int32)
    num_masks = tf.cast(features['label/num_masks'],tf.int32)

    image = tf.decode_raw(features['image/encoded'],tf.uint8)
    image = tf.reshape(image,(ih,iw,3))

    gt_boxes = tf.decode_raw(features['label/gt_boxes'],tf.float32)
    gt_boxes = tf.reshape(gt_boxes,(1,5))
    gt_masks = tf.decode_raw(features['label/gt_masks'],tf.uint8)
    gt_masks = tf.reshape(gt_masks, (ih,iw,FLAGS.num_joints))
    gt_body_masks = tf.decode_raw(features['label/gt_body_masks'],tf.uint8)
    gt_body_masks = tf.cast(gt_body_masks,tf.int32)
    gt_body_masks = tf.reshape(gt_body_masks,(1,ih,iw))

    return image,ih,iw,gt_boxes,gt_masks,gt_body_masks,num_masks,image_id

def read_refine(tfrecords_filename):
    if not isinstance(tfrecords_filename,list):
        tfrecords_filename = [tfrecords_filename]
    filename_queue = tf.train.string_input_producer(tfrecords_filename,num_epochs = 100)

    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features ={
            'image/img_id': tf.FixedLenFeature([],tf.int64),
            'image/encoded': tf.FixedLenFeature([],tf.string),
            'image/height': tf.FixedLenFeature([],tf.int64),
            'image/width': tf.FixedLenFeature([],tf.int64),
            'label/gt_boxes': tf.FixedLenFeature([],tf.string),  # of shape (1, 5), (x1, y1, x2, y2,1)
            'label/gt_masks': tf.FixedLenFeature([],tf.string),  # of shape (hh,hw,14 )
            'label/locref_map': tf.FixedLenFeature([],tf.string),  # of shape (hh,hw,28 )
            'label/locref_mask': tf.FixedLenFeature([],tf.string),  # of shape (hh,hw,28 )
            'label/num_masks':tf.FixedLenFeature([],tf.int64), # num_of valid joints
            'label/gt_body_masks': tf.FixedLenFeature([],tf.string)  #  of (1,ih,iw)
        }
    )
    image_id = tf.cast(features['image/img_id'],tf.int32)
    ih = tf.cast(features['image/height'],tf.int32)
    iw = tf.cast(features['image/width'],tf.int32)
    num_masks = tf.cast(features['label/num_masks'],tf.int32)

    image = tf.decode_raw(features['image/encoded'],tf.uint8)
    image = tf.reshape(image,(ih,iw,3))

    gt_boxes = tf.decode_raw(features['label/gt_boxes'],tf.float32)
    gt_boxes = tf.reshape(gt_boxes,(1,5))
    gt_masks = tf.decode_raw(features['label/gt_masks'],tf.uint8)
    gt_masks = tf.reshape(gt_masks, (ih,iw,FLAGS.num_joints))
    locref_map = tf.decode_raw(features['label/locref_map'],tf.uint8)
    locref_map = tf.reshape(locref_map,(ih,iw,FLAGS.num_joints*2))
    locref_mask = tf.decode_raw(features['label/locref_mask'],tf.uint8)
    locref_mask = tf.reshape(locref_mask,(ih,iw,FLAGS.num_joints*2))
    gt_body_masks = tf.decode_raw(features['label/gt_body_masks'],tf.uint8)
    gt_body_masks = tf.cast(gt_body_masks,tf.int32)
    gt_body_masks = tf.reshape(gt_body_masks,(1,ih,iw))

    return image,ih,iw,gt_boxes,gt_masks,gt_body_masks,num_masks,image_id,locref_map,locref_mask
