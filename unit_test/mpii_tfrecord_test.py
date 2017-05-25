import numpy as np
from PIL import Image
import tensorflow as tf
import cv2


#
#     'image/img_id': tf.FixedLenFeature([],tf.int64),
#     'image/encoded': tf.FixedLenFeature([],tf.string),
#     'image/height': tf.FixedLenFeature([],tf.int64),
#     'image/width': tf.FixedLenFeature([],tf.int64),
#     'label/gt_boxes': tf.FixedLenFeature([],tf.string),  # of shape (1, 5), (x1, y1, x2, y2,1)
#     'label/gt_masks': tf.FixedLenFeature([],tf.string),  # of shape (hh,hw,14 )
#     'label/locref_map': tf.FixedLenFeature([],tf.string),  # of shape (hh,hw,28 )
#     'label/locref_mask': tf.FixedLenFeature([],tf.string),  # of shape (hh,hw,28 )
#     'label/num_masks':tf.FixedLenFeature([],tf.int64), # num_of valid joints
#     'label/gt_body_masks': tf.FixedLenFeature([],tf.string)  #  of (1,ih,iw)
# }

record_iterator = tf.python_io.tf_record_iterator(path = '/home/hpc/ssd/lyj/FastMaskRCNN/data/lsp/records/lsp_lsp_train_00000-of-00004.tfrecord')
count = 0
for string_record in record_iterator:

    example = tf.train.Example()
    example.ParseFromString(string_record)

    image_id = int(example.features.feature['image/img_id'].int64_list.value[0])
    height = int(example.features.feature['image/height'].int64_list.value[0])
    width = int(example.features.feature['image/width'].int64_list.value[0])


    # print (image_id , height, width, num_masks)

    img = (example.features.feature['image/encoded'].bytes_list.value[0])
    img_1d = np.fromstring(img,dtype = np.uint8)
    #
    img_f = img_1d.reshape((height,width,-1))
    #
    img_f = np.flip(img_f, 2)
    #
    cv2.imshow('img-f',img_f)
    cv2.waitKey(0)
    #
    # mask = (example.features.feature['label/gt_masks'].bytes_list.value[0])
    # mask_1d = np.fromstring(mask,dtype = np.uint8)
    # mask_f = mask_1d.reshape(height,width,-1)
    #
    # print(mask_f)
    # ch = mask_f[:,:,12]
    # print ('*************')
    # print (ch)
    # tmp = np.zeros((height,width,3))
    # # cv2.imshow('xxx',tmp)
    # # cv2.waitKey(0)
    # # raise
    # tmp[:,:,0] = ch
    # tmp[:,:,1] = ch
    # tmp[:,:,2] = ch
    # cv2.imshow('heatmap1 ',tmp)
    # cv2.waitKey(0)
    #
    # rec = (example.features.feature['label/gt_boxes'].bytes_list.value[0])
    # rec_1d = mask_1d = np.fromstring(rec,dtype = np.float32)
    # rec_f = rec_1d.reshape((1,5))
    # print (rec_f)
    #
    #
    body_mask = (example.features.feature['label/gt_body_masks'].bytes_list.value[0])
    body_mask_1d = np.fromstring(body_mask,dtype = np.uint8)
    body_mask_1d = body_mask_1d.astype(float)
    body_mask_f = body_mask_1d.reshape((1,height,width))
    body_mask = body_mask_f[0,:,:]
    cv2.imshow('body_mask',body_mask)
    cv2.waitKey(0)
    count += 1
    if count ==10 :
        raise
    #
    #
    # 'image/img_id': _int64_feature(image_id),
    # 'image/encoded': _bytes_feature(image_data),
    # 'image/height': _int64_feature(height),
    # 'image/width': _int64_feature(width),
    # 'label/gt_boxes': _bytes_feature(gt_boxes),  # of shape (1, 5), (x1, y1, x2, y2,1)
    # 'label/gt_masks': _bytes_feature(masks),  # of shape (how many joints, 2) (x, y)
    # 'label/num_masks':_int64_feature(num_masks),
    # 'label/gt_body_masks': _bytes_feature(body_masks)
