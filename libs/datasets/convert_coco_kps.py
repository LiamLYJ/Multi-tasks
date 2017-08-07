from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import numpy as np
import tensorflow as tf
import scipy.io as sio
import cv2
from PIL import Image
from libs.datasets.pycocotools.coco import COCO

from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

FLAGS = tf.app.flags.FLAGS

class ImageReader(object):
  def __init__(self):
    self._decode_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_data, channels=3)
    self._decode_png = tf.image.decode_png(self._decode_data)

  def read_jpeg_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape

  def read_png_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image


def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _to_tfexample_raw(image_id, image_data,
                           height, width,
                           gt_boxes, gt_masks,num_instances):
  """ just write a raw input"""
  return tf.train.Example(features=tf.train.Features(feature={
    'image/img_id': _int64_feature(image_id),
    'image/encoded': _bytes_feature(image_data),
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'label/gt_boxes': _bytes_feature(gt_boxes),  # of shape (N, 5), (x1, y1, x2, y2,1)
    'label/gt_masks': _bytes_feature(gt_masks),  # of shape (14,h,w)
    'label/num_instances': _int64_feature(num_instances)
  }))

def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = '%s_%03dth_shard.tfrecord' % (
      split_name, shard_id)
  return os.path.join(dataset_dir, output_filename)


def _add_to_tfrecord_mat(record_dir,dataset_dir,annotation_path,dataset_split_name,npz_path):
    assert dataset_split_name in ['train2014','val2014']
    annFile = os.path.join(annotation_path,'person_keypoints_train2014.json')
    npz_file = np.load(npz_path)
    coco = COCO(annFile)
    print ('start to convert data')
    num_joints = 14
    num_per_shard = 2500
    num_whole = len(npz_file['id'])
    shard_id = 0
    file_head = '/home/hpc/ssd/lyj/Multi-tasks/data/coco/train2014/'
    table = { 0:15, 1:13, 2:11, 3:12, 4:14, 5:16, 6:9, 7:7 , 8:5, 9:6 , 10:8, 11:10, 12:18 ,13:17 }

    num_shard = np.ceil(num_whole_id / num_per_shard)
    for shard_id in range(0,int(num_shard)):
        record_filename = _get_dataset_filename(record_dir,dataset_split_name,shard_id)
        # options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
            for id in range( num_per_shard*shard_id, min(num_per_shard*shard_id + num_per_shard, num_whole)):
                if id % 10 ==0:
                    sys.stdout.write('\r>> Converting data id: %d shard %d\n' %(id+1,shard_id))
                    sys.stdout.flush()
                imgId = npz_file['id'][id]
                img_path = file_head + coco.loadImgs(imgId)[0]['file_name']
                img = np.array(Image.open(img_path))
                height = int(img.shape[0])
                width = int(img.shape[1])
                img = img.astype(np.uint8)
                img_raw = img.tostring()
                masks = np.zeros((num_joints,height,width))
                num_instances = len(npz_file['kp'][id])
                for person in range(num_instances):
                    kp = npz_file['kp'][id][person]
                    for joint_id in range(num_joints):
                        joint_id_coco = table[joint_id]
                        if  kp[joint_id_coco*3 +2] == 0 :
                            # this joint is not labeled
                            continue
                        masks[joint_id,kp[joint_id_coco*3],kp[joint_id_coco*3 +1]] = 1
                boxes = npz_file['box'][id]
                boxes = boxes.astype(np.float32)
                boxes_raw = boxes.tostring()
                masks = masks.astype(np.uint8)
                masks_raw = masks.tostring()

                example = _to_tfexample_raw(
                    id+ shard_id*num_per_shard,img_raw, height, width, boxes_raw,masks_raw,num_instances
                )
                tfrecord_writer.write(example.SerializeToString())
                # print ('******************')

            tfrecord_writer.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

def run(dataset_dir,dataset_split_name='train2014',npz_path = None):
    # write coco_kps to tf_record
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    record_dir = os.path.join(dataset_dir,'kps_records')
    annotation_path = os.path.join(dataset_dir,'annotations_2')
    npz_path =
    if not tf.gfile.Exists(record_dir):
        tf.gfile.MakeDirs(record_dir)

    if dataset_split_name in ['train2014','val2014']:
        _add_to_tfrecord_mat(record_dir,dataset_dir,annotation_path,dataset_split_name,npz_path)

    print ('\nFinished write mpii dataset to tf_record')

if __name__ == '__main__':
    run('/home/hpc/ssd/lyj/mpii_data','mpii_human_pose_v1_u12_2')