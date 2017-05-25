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
from matplotlib import pyplot as plt
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

FLAGS = tf.app.flags.FLAGS

def _mask_encode(masks,viss,ih,iw,stride = None,pose_dist_stride = None,num_joints = None,hp = False):

    if num_joints is None:
        num_joints = FLAGS.num_joints

    mask_targets = np.zeros((int(ih),int(iw),num_joints),np.uint8)
    locref_mask = np.zeros((int(ih),int(iw),num_joints*2),np.uint8)
    locref_map = np.zeros((int(ih),int(iw),num_joints*2),np.uint8)


# for heatmap generation
    if hp :
        if pose_dist_stride is None:
            dist_thresh = FLAGS.pos_dist_stride
        else :
            dist_thresh = pose_dist_stride
        if stride is None:
            stride = FLAGS.heatmap_stride
        half_stride = stride / 2
        dist_thresh_sq = dist_thresh ** 2
        locref_scale = 1.0 / 7.2801
        for k in range(num_joints):
            if (viss[k] == 1):
                j_x = masks[k,0]
                j_y = masks[k,1]

                j_x_sm = round((j_x - half_stride) /stride)
                j_y_sm = round((j_y - half_stride) /stride)

                min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                max_x = round(min(j_x_sm + dist_thresh + 1, iw - 1))
                min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                max_y = round(min(j_y_sm + dist_thresh + 1, ih - 1))

                for j in range(int(min_y), int(max_y) + 1):  # range(height):
                    pt_y = j * stride + half_stride
                    for i in range(int(min_x), int(max_x) + 1):  # range(width):
                        # pt = arr([i*stride+half_stride, j*stride+half_stride])
                        # diff = joint_pt - pt
                        # The code above is too slow in python
                        pt_x = i * stride + half_stride
                        dx = j_x - pt_x
                        dy = j_y - pt_y
                        dist = dx ** 2 + dy ** 2
                        # print(la.norm(diff))
                        if dist <= dist_thresh_sq:
                            mask_targets[j, i, k] = 1
                            locref_mask[j,i,k*2 + 0] = 1
                            locref_mask[j,i,k*2 + 1] = 1
                            locref_map[j,i,k*2 + 0] = dx * locref_scale
                            locref_map[j,i,k*2 + 1] = dy * locref_scale

# for one-hot generation
    else:
        for k in range(num_joints):
            j_y,j_x = 0,0

            if (viss[k] == 1):
                j_x = masks[k,0]
                j_y = masks[k,1]
                try :
                    mask_targets[min(int(j_y),ih-1),min(int(j_x),iw-1),k] =1
                except:
                    print ('y:',j_y)
                    print ('x:',j_x)
                    print ('ih:',ih)
                    print ('iw:',iw)
    return mask_targets,locref_map, locref_mask

def _get_distance(p1,p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def _get_center(p1,p2):
    #
    # p1 = np.array((p1))
    # p2 = np.array((p2))
    return np.round((p1+p2)*0.5)

def _get_angle(p1,p2,ih,iw):
    a = _get_distance(p1,p2) + 0.01 # incase get NAN for sin\theta
    h = abs(p1[1] - p2[1])
    angle = math.asin(h/a)*180/math.pi

    if (p1[0] >= p2[0] and p1[1] >= p2[1]) or (p1[0] <= p2[0] and p1[1] <= p2[1]):
        pass
    else:
        angle = 90 + angle
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
            angle = _get_angle(joints[indexes[0],:],joints[indexes[1],:],ih,iw)
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
            angle = _get_angle(joints[0,:],joints[1,:],ih,iw)
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

def _generate_body_mask(gt_masks =None,masks= None,viss =None,ih=None,iw=None,stride = None):
    # just return body_mask as the combination of body joints masks
    if gt_masks is not None:
        assert masks is None
        assert viss is None
        assert (ih is None and iw is None)
        assert stride is None
        gt = np.zeros((1,gt_masks.shape[0],gt_masks.shape[1]))
        gt[0,:,:] = np.sum(gt_masks,axis = -1)
        keep = np.where(gt >= 1)
        gt[keep] = 1
    # generate body_mask based on dilation
    else :
        if stride is None:
            stride = FLAGS.body_stride
        gt_tmp = _get_part_strucute(masks, viss, ih, iw, stride)
        gt = np.zeros((1,ih,iw))
        size = np.ones(2,dtype = np.int32)
        size[0] = 5 if ih > iw else 6
        size[1] = 5 if ih <= iw else 6
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size[0],size[1]))
        gt[0,:,:] = cv2.dilate(gt_tmp,kernel,iterations = 1)
    return gt

def _generate_box(masks,viss,ih,iw):
    # first check is neck and uphead visable
    if (viss[12] >0 and viss[13] > 0):
        headsize = _get_distance(masks[12,:],masks[13,:])
        dis = headsize * 0.5
    else:
        dis = 0.1 * min(ih,iw)
    keep = np.where(viss > 0)

    x1 = max(1,min(masks[np.array(keep)[0],0] - dis))
    x2 = min(iw-1,max(masks[np.array(keep)[0],0] + dis))
    y1 = max(1,min(masks[np.array(keep)[0],1] - dis))
    y2 = min(ih-1,max(masks[np.array(keep)[0],1] + dis))
    gt_boxes = np.array([x1,y1,x2,y2,1],dtype = np.float32)

    return gt_boxes

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


def _to_tfexample_refine_raw(image_id, image_data,
                           height, width,
                           gt_boxes, masks,locref_map, locref_mask,num_masks,body_masks):
  """ just write a raw input"""
  return tf.train.Example(features=tf.train.Features(feature={
    'image/img_id': _int64_feature(image_id),
    'image/encoded': _bytes_feature(image_data),
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'label/gt_boxes': _bytes_feature(gt_boxes),  # of shape (1, 5), (x1, y1, x2, y2,1)
    'label/gt_masks': _bytes_feature(masks),  # of shape (hh,hw,14)
    'label/locref_map': _bytes_feature(locref_map), # of shape (hh,hw,28)
    'label/locref_mask': _bytes_feature(locref_mask), # of shape (hh,hw,28) with 0,1 masks
    'label/num_masks':_int64_feature(num_masks),
    'label/gt_body_masks': _bytes_feature(body_masks) # of (1,ih,iw)
  }))


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'lsp_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _add_to_tfrecord_mat(record_dir,dataset_dir,annotation_path,dataset_split_name):

    mat_file = sio.loadmat(annotation_path)['joints']
    num = mat_file.shape[-1]
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

                obj = mat_file[:,:,image_id]
                # load img and convert to string
                img_path = dataset_dir+'images/'+ 'im%05d.jpg'%(image_id+1)
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
                masks = np.array(obj[:,0:2],dtype = np.float32)
                viss = np.array(obj[:,2],dtype = np.int32)
                assert (masks.shape[0] == viss.shape[0])

                masks[:,0] = masks[:,0] * w_scale
                masks[:,1] = masks[:,1] * h_scale

                gt_masks,locref_map, locref_mask = _mask_encode(masks,viss,height,width,hp = True )
                gt_masks = gt_masks.astype(np.uint8)
                locref_map = locref_map.astype(np.uint8)
                locref_mask = locref_mask.astype(np.uint8)

                num_masks = len(np.where(viss > 0)[0])

                masks_raw = gt_masks.tostring()
                locref_map_raw = locref_map.tostring()
                locref_mask_raw = locref_mask.tostring()
                # use joints coordinate to generate rec and convert to string

                gt_boxes = _generate_box(masks,viss,height,width)
                gt_boxes_raw = gt_boxes.tostring()

                # use joints coordinate to generate body_masks and convert to string

                body_masks = _generate_body_mask(masks = masks, viss =viss, ih = height, iw=width)
                body_masks = body_masks.astype(np.uint8)
                body_masks_raw = body_masks.tostring()

                if FLAGS.use_refine:
                    example = _to_tfexample_refine_raw(image_id, img_raw,
                                           height, width,
                                           gt_boxes_raw, masks_raw,locref_map_raw,locref_mask_raw,num_masks,body_masks_raw)

                else :
                    example = _to_tfexample_raw(image_id, img_raw,
                                           height, width,
                                           gt_boxes_raw, masks_raw,num_masks,body_masks_raw)

                tfrecord_writer.write(example.SerializeToString())

            tfrecord_writer.close()
    sys.stdout.write('\n')
    sys.stdout.flush()

def run(dataset_dir,dataset_split_name):
    # write lsp dataset to tf_record
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    record_dir = os.path.join(dataset_dir,'records')
    annotation_path = os.path.join(dataset_dir,dataset_split_name,'joints.mat')

    if not tf.gfile.Exists(record_dir):
        tf.gfile.MakeDirs(record_dir)

    _add_to_tfrecord_mat(record_dir,dataset_dir,annotation_path,dataset_split_name)

    print ('\nFinished write lsp dataset to tf_record')

if __name__ == '__main__':
    tmp_mat = sio.loadmat('/home/hpc/ssd/lyj/FastMaskRCNN/data/lsp/lsp_train/joints.mat')
    check = tmp_mat['joints'][:,:,0]
    img_path = '/home/hpc/ssd/lyj/FastMaskRCNN/data/lsp/images/im00001.jpg'
    img = cv2.imread(img_path)
    ih,iw = img.shape[0],img.shape[1]
    #
    masks = np.array(check[:,0:2],dtype = np.float32)
    print (masks)
    viss = np.array(check[:,2],dtype = np.int32)
    print (viss)
    gt_masks = _mask_encode(masks,viss,ih,iw,stride = 1,pose_dist_stride = 10,num_joints= 14,hp = True)
    # body_masks1 = _generate_body_mask(gt_masks)
    body_masks = _generate_body_mask(masks = masks, viss =viss, ih = ih, iw=iw, stride =30)

    # kernel = np.ones((5,5),np.uint8)
    # body_dilation = cv2.dilate(body_masks[0,:,:],kernel,iterations = 2)

    # gt_boxes = _generate_box(masks,viss,ih,iw)
    # boxes = np.zeros((ih,iw))
    # boxes[int(gt_boxes[1]):int(gt_boxes[3]),int(gt_boxes[0]):int(gt_boxes[2])] = 1

    # cv2.imshow('gt_boxes',boxes)
    # cv2.imshow('mask_result',tmp[:,:,1])
    # print (np.where(body_masks>=1))
    # print (gt_masks.shape)
    # raise

    cv2.imshow('img',img)
    for i in range(10):
        cv2.imshow('gt_masks%d:'%i,gt_masks[:,:,i])
    # cv2.imshow('body_mask',body_masks[0,:,:])
    # cv2.imshow('body_mask1',body_masks1[0,:,:])
    # cv2.imshow('img',img)
    # cv2.imshow('body_erosion',body_dilation)

    cv2.waitKey(0)
    # cv2.imwrite('image.png',img)
    # keep = np.where(body_masks >= 1)
    # body_masks[keep] = 255
    # cv2.imwrite('body_mask.png',body_masks[0,:,:])
