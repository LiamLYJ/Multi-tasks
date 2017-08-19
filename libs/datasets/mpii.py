from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

FLAGS = tf.app.flags.FLAGS

def read(tfrecords_filename):
    num_joints = 14
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
            'label/gt_boxes': tf.FixedLenFeature([],tf.string),  # of shape (N, 5), (x1, y1, x2, y2,1)
            'label/gt_masks': tf.FixedLenFeature([],tf.string),  # of shape (14,hh,hw )
            'label/num_instances': tf.FixedLenFeature([],tf.int64)
        }
    )
    image_id = tf.cast(features['image/img_id'],tf.int32)
    ih = tf.cast(features['image/height'],tf.int32)
    iw = tf.cast(features['image/width'],tf.int32)
    num_instances = tf.cast(features['label/num_instances'],tf.int32)

    image = tf.decode_raw(features['image/encoded'],tf.uint8)
    image = tf.reshape(image,(ih,iw,3))

    gt_boxes = tf.decode_raw(features['label/gt_boxes'],tf.float32)
    gt_boxes = tf.reshape(gt_boxes,[num_instances,5])
    gt_masks = tf.decode_raw(features['label/gt_masks'],tf.uint8)
    gt_masks = tf.reshape(gt_masks, [num_joints,ih,iw])

    return image,ih,iw,gt_boxes,gt_masks,num_instances,image_id

if __name__ == '__main__':
    with tf.Graph().as_default():
        image, ih, iw, gt_boxes, gt_masks,num_instances,image_id = \
            read('/home/hpc/ssd/lyj/Multi-tasks/data/coco/kps_records/train2014_000th_shard.tfrecord')
        image_tmp = tf.cast(image, tf.float32)
        img_4d = tf.expand_dims(image_tmp,0)
        sess = tf.Session()
        init_op = (tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ))
        sess.run(init_op)
        tf.train.start_queue_runners(sess = sess)

        image_sum_sample_shape = tf.shape(img_4d)[1:]
        gt_x_min = gt_boxes[:, 0] / tf.cast(image_sum_sample_shape[1], tf.float32)
        gt_y_min = gt_boxes[:, 1] / tf.cast(image_sum_sample_shape[0], tf.float32)
        gt_x_max = gt_boxes[:, 2] / tf.cast(image_sum_sample_shape[1], tf.float32)
        gt_y_max = gt_boxes[:, 3] / tf.cast(image_sum_sample_shape[0], tf.float32)
        bb = tf.stack([gt_y_min, gt_x_min, gt_y_max, gt_x_max], axis=1)

        tensor_img_with_gt = tf.image.draw_bounding_boxes(img_4d, tf.expand_dims(bb, 0))

        tmp = gt_boxes[:,0]
        with sess.as_default():
            for i in range(2):
                image_, ih_, iw_, gt_boxes_, gt_masks_,image_id_,num_,bb_,results_ = \
                    sess.run([image, ih, iw, gt_boxes, gt_masks, image_id,num_instances,bb,tensor_img_with_gt])
                print ('num:', num_)
                print ('ih: ', ih_)
                print ('iw: ', iw_)
                print ('image_id: ', image_id_)
                print ('gt_boxes :', gt_boxes_)
                print ('gt_boxes_shape:',gt_boxes_.shape)
                print ('gt_masks:', gt_masks_.shape)
                print ('gt_bb:',bb_)
                import cv2
                # for j in range(gt_boxes.shape[0]):
                #     x1 = gt_boxes[j][0]
                #     y1 = gt_boxes[j][1]
                #     x2 = gt_boxes[j][2]
                #     y2 = gt_boxes[j][3]
                #     cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0))
                # cv2.imshow('img',image_)
                cv2.imshow('img',results_[0])
                cv2.waitKey(0)
                # import numpy as np
                # for j in range(14):
                #     print ('this is %d_th:'%j)
                #     h,w = np.where(gt_masks[j] == 1)
                #     for num in range(h.shape[0]):
                #         print ('%d one is :'%num)
                #         print (h[num],w[num])
