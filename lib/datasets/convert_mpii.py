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
import cv2

from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

FLAGS = tf.app.flags.FLAGS

def _mask_encode(masks,ih,iw,hp=False):

    mask_targets = np.zeros((int(ih),int(iw),FLAGS.num_joints),np.uint8)
    num_valid = masks.shape[0]
    dump = False
    if hp:
        stride = FLAGS.heatmap_stride
        dist_h = round(float(ih)/stride)
        dist_w = round(float(iw)/stride)

        for k in range(num_valid):
            j_id = masks[k,0]
            j_x = masks[k,1]
            j_y = masks[k,2]

            min_x = round(max(1,j_x - dist_w -1))
            max_x = round(min(iw-1,j_x + dist_w + 1))
            min_y = round(max(1,j_y - dist_h -1))
            max_y = round(min(ih-1,j_y + dist_h +1))

            mask_targets[int(min_y):int(max_y),int(min_x):int(max_x),int(j_id)] = 1
    else:
        for k in range(num_valid):
            j_id = masks[k,0]
            j_x = masks[k,1]
            j_y = masks[k,2]
            try :
                mask_targets[int(j_y),int(j_x),int(j_id)] =1
            except:
                # print ('y:',j_y)
                # print ('x:',j_x)
                # print ('ih:',ih)
                # print ('iw:',iw)
                # print ('j_id:',j_id)
                dump = True
    return mask_targets,dump

def _make_mpii_to_dilate(masks,num_joints = None):
    if num_joints is None:
        num_joints = FLAGS.num_joints
    num_valid = masks.shape[0]
    mpii_viss = np.zeros(num_joints)
    mpii_masks = np.zeros((num_joints,2))
    for k in range(num_valid):
        mpii_viss[int(masks[k,0])] = 1
        mpii_masks[int(masks[k,0]),:] = masks[k,1:]

    return mpii_masks,mpii_viss

def _get_distance(p1,p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def _get_center(p1,p2):
    return np.round((p1+p2)*0.5)

def _get_angle(p1,p2):
    a = _get_distance(p1,p2) + 0.01 # incase get NAN for sin\theta
    h = abs(p1[1] - p2[1])
    angle = math.asin(h/a)*180/math.pi
    # print (angle,a)
    return angle

def _draw_seg(img,joints,viss,stride):
    # draw body part seg with rec or ellipse
    ih,iw = img.shape[0],img.shape[1]
    dist_w = round(iw / stride)
    dist_h = round(ih / stride)

    if joints.shape[1] != 2:
        raise ValueError('the joints are invalid')
    if (joints.shape[0] == 4 and viss.shape[0] ==4):
        if len(np.where(viss>0)[0]) == 0:
            pass
        elif len(np.where(viss>0)[0]) == 1:
            index = np.where(viss>0)[0]
            # print ('**************')
            # print (index)
            # print (joints[index,0])
            # raise
            x1 = max(1,joints[index,0] - dist_w)
            y1 = max(1,joints[index,1] - dist_h)
            x2 = min(iw-1,joints[index,0] + dist_w)
            y2 = min(ih-1,joints[index,1] + dist_h)
            img[int(y1):int(y2),int(x1):int(x2)] = 1
        elif len(np.where(viss>0)[0]) == 2:
            indexes = np.where(viss>0)[0]
            center = _get_center(joints[indexes[0],:],joints[indexes[1],:])
            center = np.array(center,np.int32)
            angle = _get_angle(joints[indexes[0],:],joints[indexes[1],:])
            cv2.ellipse(img,(center[0],center[1]),(int(0.5*_get_distance(joints[indexes[0],:],
                                            joints[indexes[1],:])),int(min(dist_h,dist_w))),
                                        int(angle),0,360,1,-1)
        else :
            indexes = np.where(viss>0)[0]
            if len(indexes) == 3:
                center1 = _get_center(joints[indexes[0],:],joints[indexes[1],:])
                center2 = _get_center(joints[indexes[1],:],joints[indexes[2],:])
            else:
                center1 = _get_center(joints[indexes[0],:],joints[indexes[1],:])
                center2 = _get_center(joints[indexes[2],:],joints[indexes[3],:])
            center = _get_center(center1,center2)
            dist = []
            for i in range(len(indexes)):
                dist.append(_get_distance(center,joints[indexes[i],:]))

            # draw a rectangle around the center (deprecated )
            # thresh = min(dist)
            # x1 = max(1,center[0] - thresh)
            # y1 = max(1,center[1] - thresh)
            # x2 = min(iw-1,center[0] + thresh)
            # y2 = min(ih-1,center[1] + thresh)
            # img[int(y1):int(y2),int(x1):int(x2)] = 1

            # draw ellipse
            center = np.array(center,np.int32)
            cv2.ellipse(img,(center[0],center[1]),(int(max(dist)*0.9),int(min(dist)*0.8)),
                                        0,0,360,1,-1)

    elif (joints.shape[0] == 2 and viss.shape[0] ==2 ):
        if viss[0] == 0 and viss[1] == 0:
            pass
        elif viss[0] == 1 and viss[1] == 1:
            center = _get_center(joints[0,:],joints[1,:])
            angle = _get_angle(joints[0,:],joints[1,:])
            center = np.array(center,np.int32)
            cv2.ellipse(img,(center[0],center[1]),(int(0.5*_get_distance(joints[0,:],joints[1,:])),int(min(dist_h,dist_w))),
                                        int(angle),0,360,1,-1)
        else:
            index = np.where(viss > 0)[0]
            x1 = max(1,joints[index,0] - dist_w)
            y1 = max(1,joints[index,1] - dist_h)
            x2 = min(iw-1,joints[index,0] + dist_w)
            y2 = min(ih-1,joints[index,1] + dist_h)
            img[int(y1):int(y2),int(x1):int(x2)] = 1
    else :
        raise ValueError('the joints and viss are invalid')

def _get_part_strucute(masks,viss,ih,iw,stride):
    img = np.zeros((ih,iw,1))

    # head; (left and right) up(bottom) arm, up(bottom)leg,uppart body -> 10 parts
    label =[]
    label.append([13,12])
    label.append([8,7])
    label.append([7,6])
    label.append([9,10])
    label.append([10,11])
    label.append([8,9,2,3])
    label.append([2,1])
    label.append([1,0])
    label.append([3,4])
    label.append([4,5])
    for i in range(10):
        _draw_seg(img,masks[label[i],:],viss[label[i]],stride)
    part_structure = img
    return part_structure

def _generate_body_mask(gt_body_masks =None,num_body_masks=None,masks= None,viss =None,ih=None,iw=None,stride = None):
    # just return body_mask as the combination of body joints masks
    if gt_body_masks is not None:
        assert masks is None
        assert viss is None
        assert num_body_masks is not None
        assert stride is None
        gt = np.zeros((1,ih,iw))
        for i in range(num_body_masks):
            gt[0,gt_body_masks[i,1]:gt_body_masks[i,3],gt_body_masks[i,0]:gt_body_masks[i,2]] = 1
    # generate body_mask based on dilation
    else :
        if stride is None:
            stride = FLAGS.heatmap_stride
        gt_tmp = _get_part_strucute(masks, viss, ih, iw, stride)
        gt = np.zeros((1,ih,iw))
        size = np.ones(2,dtype = np.int32)
        size[0] = 5 if ih > iw else 6
        size[1] = 5 if ih <= iw else 6
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size[0],size[1]))
        gt[0,:,:] = cv2.dilate(gt_tmp,kernel,iterations = 1)
    return gt


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


def _to_tfexample_coco_raw(image_id, image_data,
                           height, width,
                           gt_boxes, masks,num_masks,body_masks):
  """ just write a raw input"""
  return tf.train.Example(features=tf.train.Features(feature={
    'image/img_id': _int64_feature(image_id),
    'image/encoded': _bytes_feature(image_data),
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'label/gt_boxes': _bytes_feature(gt_boxes),  # of shape (1, 5), (x1, y1, x2, y2,1)
    'label/gt_masks': _bytes_feature(masks),  # of shape (hh,hw,14)
    'label/num_masks':_int64_feature(num_masks),
    'label/gt_body_masks': _bytes_feature(body_masks) # of (1,ih,iw)
  }))

def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'mpii_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _add_to_tfrecord_mat(record_dir,dataset_dir,annotation_path,dataset_split_name):
    assert dataset_split_name in ['testcropped','cropped_mask_rec']

    mat_file = sio.loadmat(annotation_path)['dataset'][0]
    num = len(mat_file)
    cats = 2
    print ('%s has %d images' % (dataset_split_name,num))

    num_shards = int(num/ 2500)
    num_per_shard = int(math.ceil(num / float(num_shards)))

    for shard_id in range(num_shards):
        record_filename = _get_dataset_filename(record_dir,dataset_split_name,shard_id,num_shards)
        # options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, num)
            for image_id in range(start_ndx,end_ndx):

                if image_id % 10 == 0:
                    sys.stdout.write('\r>> Converting image %d/%d shard %d\n' %(image_id+1,num,shard_id))
                    sys.stdout.flush()

                obj = mat_file[image_id]
                # load img and convert to string
                img_path = obj['image'][0]
                img = np.array(Image.open(img_path))
                old_height = int(img.shape[0])
                old_width = int(img.shape[1])
                min_side = min(old_width,old_height)
                scale_side = 512 / min_side
                if  round(scale_side *old_height) %32 != 0:
                    height = int((round(scale_side *old_height) // 32 + 1)* 32)
                else:
                    height = int(round(scale_side *old_height))
                if  round(scale_side * old_width) %32 != 0:
                    width = int((round(scale_side * old_width) // 32 + 1)* 32)
                else:
                    width = int(round(scale_side * old_width))

                w_scale = float(width) / old_width
                h_scale = float(height) / old_height
                img = cv2.resize(img,(width,height))
                img = img.astype(np.uint8)
                img_raw = img.tostring()

                # load mask and convert to string
                masks = np.array(obj['joints'][0][0],dtype = np.float32)
                # gt_ masks shape ( heatmap_h,heatmap_w,num_joints)
                masks[:,1] = masks[:,1] * w_scale
                masks[:,2] = masks[:,2] * h_scale
                gt_masks,dump = _mask_encode(masks,height,width)
                # the joints position is out of image
                if dump:
                    continue

                gt_masks = gt_masks.astype(np.uint8)
                num_masks = int(masks.shape[0])
                masks_raw = gt_masks.tostring()

                # load rec and convert to string
                gt_boxes = np.array([[obj['rec'][0][0]['x1'][0][0],obj['rec'][0][0]['y1'][0][0],
                                    obj['rec'][0][0]['x2'][0][0],obj['rec'][0][0]['y2'][0][0],1]],dtype= np.float32)
                gt_boxes[0,0:-1:2] = gt_boxes[0,0:-1:2] * w_scale
                gt_boxes[0,1:-1:2] = gt_boxes[0,1:-1:2] * h_scale
                gt_boxes_raw = gt_boxes.tostring()

                # load body mask and convert to string
                body_masks_x1  = obj['mask'][0]['x1']
                body_masks_y1  = obj['mask'][0]['y1']
                body_masks_x2  = obj['mask'][0]['x2']
                body_masks_y2  = obj['mask'][0]['y2']

                assert body_masks_x1.shape == body_masks_y1.shape
                assert body_masks_x2.shape == body_masks_y2.shape
                assert body_masks_x1.shape == body_masks_x2.shape

                num_tmp_body_masks = int(body_masks_x1.shape[0])

                tmp_body_masks = np.zeros((num_tmp_body_masks,4),dtype = np.int32)

                for i in range(num_tmp_body_masks):
                    tmp_body_masks[i][0] = round(body_masks_x1[i][0][0] * w_scale)
                    tmp_body_masks[i][1] = round(body_masks_y1[i][0][0] * h_scale)
                    tmp_body_masks[i][2] = round(body_masks_x2[i][0][0] * w_scale)
                    tmp_body_masks[i][3] = round(body_masks_y2[i][0][0] * h_scale)
                # generate body_masks based on x1,y1,x2,y2 idea

                # body_masks = _generate_body_mask(gt_body_masks=tmp_body_masks,
                #                                 ih= height,iw=width,num_body_masks= num_tmp_body_masks)

                # geneate body_masks based on joints dilation

                mpii_masks, mpii_viss = _make_mpii_to_dilate(masks)
                body_masks = _generate_body_mask(masks =mpii_masks, ih =height, iw = width, viss = mpii_viss)
                body_masks = body_masks.astype(np.uint8)
                body_masks_raw = body_masks.tostring()

                example = _to_tfexample_coco_raw(image_id, img_raw,
                                           height, width,
                                           gt_boxes_raw, masks_raw,num_masks,body_masks_raw)

                tfrecord_writer.write(example.SerializeToString())

            tfrecord_writer.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

def run(dataset_dir,dataset_split_name):
    # write mpii dataset to tf_record
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    record_dir = os.path.join(dataset_dir,'records')
    annotation_path = os.path.join(dataset_dir,dataset_split_name,'dataset.mat')

    if not tf.gfile.Exists(record_dir):
        tf.gfile.MakeDirs(record_dir)

    _add_to_tfrecord_mat(record_dir,dataset_dir,annotation_path,dataset_split_name)

    print ('\nFinished write mpii dataset to tf_record')

if __name__ == '__main__':
    tmp_mat = sio.loadmat('/home/gpu_server/lyj/FastMaskRCNN/data/mpii/testcropped/dataset.mat')
    # print (tmp_mat)
    check = tmp_mat['dataset'][0][0]
    # print (np.array(check['mask'][0]['x1']))
    # print (np.array(check['mask'][0]['x1']).shape)
    # print (check['mask'][0]['x1'][15][0][0])
    # print ('image:',check['image'])
    # print ('size:',check['size'])
    # print ('joints: ',check['joints'])
    # # print ('mask: ',check['mask'])
    # print ('mask x2 first one : ',check['mask'][0][0]['x2'])
    # print ('rec: ',check['rec'])
    # print ('rec x1 : ', check['rec']['x1'])
    # print (check['image'][0])
    # print (len(tmp_mat['dataset'][0]))
    # print (check['joints'][0][0])
    # print (check['joints'][0][0].shape)
    # print (check['rec'][0][0][0][0][0])
    # print (check['rec'][0][0][1][0][0])
    # print (check['rec'][0][0][2][0][0])
    # print (check['rec'][0][0][3][0][0])
    # box1 = np.array([[check['rec'][0][0]['x1'][0][0],check['rec'][0][0]['y1'][0][0],check['rec'][0][0]['x2'][0][0],check['rec'][0][0]['y2'][0][0],1]])
    # print (box1.shape)
    # print (check['rec']['x1'][0][0][0])


    masks = np.array(check['joints'][0][0],dtype = np.float32)
    image_path = check['image'][0]
    # print (image_path)
    # raise
    img = cv2.imread(image_path)
    ih,iw = img.shape[0],img.shape[1]
    mpii_masks,mpii_viss = _make_mpii_to_dilate(masks,14)
    body_masks = _generate_body_mask(masks = mpii_masks, viss =mpii_viss, ih = ih, iw=iw, stride =30)
    cv2.imshow('body_mask',body_masks[0,:,:])
    cv2.waitKey(0)
