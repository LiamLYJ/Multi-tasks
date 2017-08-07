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


def _add_to_tfrecord_mat(record_dir,dataset_dir,annotation_path,dataset_split_name):
    assert dataset_split_name in ['mpii_human_pose_v1_u12_2']

    mat_file = sio.loadmat(annotation_path)['RELEASE'][0,0]
    print ('start to convert data')
    num_joints = 14
    num_per_shard = 2500
    num_whole = 24987
    shard_id = 0
    file_head = '/home/hpc/ssd/lyj/mpii_data/images/'
    table = {0:5 , 1:4, 2:3 , 3:2 , 4:1 , 5:0 ,8:12 , 9:13 ,10:11, 11:10 ,12:9 ,13:8 , 14:7, 15:6 }
    num_shard = np.ceil(num_whole / num_per_shard)
    for shard_id in range(0,int(num_shard)):
        record_filename = _get_dataset_filename(record_dir,dataset_split_name,shard_id)
        # options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
            for id in range( num_per_shard*shard_id, min(num_per_shard*shard_id + num_per_shard, num_whole)):
                try:
                    if id % 10 ==0:
                        sys.stdout.write('\r>> Converting data id: %d shard %d\n' %(id+1,shard_id))
                        sys.stdout.flush()
                    annolist = mat_file['annolist'][0,id]

                    img_name = annolist['image'][0,0]['name'][0]
                    img_path = file_head + img_name
                    img = np.array(Image.open(img_path))
                    height = int(img.shape[0])
                    width = int(img.shape[1])
                    img = img.astype(np.uint8)
                    img_raw = img.tostring()
                    masks = np.zeros((num_joints,height,width))
                    num_instances = len(annolist['annorect'][0])
                    boxes = np.zeros((num_instances,5))
                    for person in range(num_instances):
                        x1,y1,x2,y2 = width -1 ,height -1,0,0
                        annotation = annolist['annorect'][0,person]
                        annopoints = annotation['annopoints'][0,0]['point']
                        for joint in range(len(annopoints[0])):
                            try :
                                joint_id = table[annopoints[0,joint]['id'][0,0]]
                            except:
                                continue
                            y = int(annopoints[0,joint]['y'][0,0])
                            x = int(annopoints[0,joint]['x'][0,0])
                            x1 = min(x1,x)
                            y1 = min(y1,y)
                            x2 = max(x2,x)
                            y2 = max(y2,y)
                            masks[joint_id,y,x] = 1
                        dist = 0.18 * math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        x1 = max(0, x1 - dist)
                        y1 = max(0, y1 - dist)
                        x2 = min(width -1 , x2+dist)
                        y2 = min(height-1, y2+dist)
                        boxes[person] = np.array([x1,y1,x2,y2,1])
                    boxes = boxes.astype(np.float32)
                    boxes_raw = boxes.tostring()
                    masks = masks.astype(np.uint8)
                    masks_raw = masks.tostring()

                    example = _to_tfexample_raw(
                        id+ shard_id*num_per_shard,img_raw, height, width, boxes_raw,masks_raw,num_instances
                    )
                    tfrecord_writer.write(example.SerializeToString())
                    # print ('******************')
                except:
                    print ('id : %d is for testing, does not have groundtruth'%id)
                    continue
            tfrecord_writer.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

def run(dataset_dir,dataset_split_name):
    # write mpii dataset to tf_record
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    record_dir = os.path.join(dataset_dir,'records')
    annotation_path = os.path.join(dataset_dir,dataset_split_name,'mpii_human_pose_v1_u12_1.mat')
    # annotation_path = os.path.join(dataset_dir,dataset_split_name,'test.mat')

    if not tf.gfile.Exists(record_dir):
        tf.gfile.MakeDirs(record_dir)

    _add_to_tfrecord_mat(record_dir,dataset_dir,annotation_path,dataset_split_name)

    print ('\nFinished write mpii dataset to tf_record')

if __name__ == '__main__':
    # tmp_mat = sio.loadmat('/home/hpc/ssd/lyj/mpii_data/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')
    # image_path = tmp_mat['RELEASE'][0,0]['annolist'][0,0]['image'][0,0]['name'][0]
    # file_head = '/home/hpc/ssd/lyj/mpii_data/images/'
    # img = cv2.imread(file_head + image_path)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # print (len(tmp_mat['RELEASE'][0,0]['annolist'][0,4]['annorect'][0]))
    # raise
    run('/home/hpc/ssd/lyj/mpii_data','mpii_human_pose_v1_u12_2')
