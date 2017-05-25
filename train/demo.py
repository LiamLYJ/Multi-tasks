import libs.configs.config_v1 as cfg
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from train.test import im_detect
import cv2
from libs.boxes.nms_wrapper import nms

CLASSES = []
for i in range(81):
    CLASSES.append(str(i))

def vis_detectios(im, class_name,dets,thresh = 0.5):
    inds = np.where(dets[:,-1] >=thresh )[0]

    if len(inds) == 0:
        return

    im = im[:,:,(2,1,0)]
    fig, ax = plt.subplots(figsize = (12,12))
    ax.imshow(im, aspect = 'equal')

    for i in inds:
        bbox = dets[i,:4]
        score = dets[i,-1]

        ax.add_patch(
            plt.Rectangle((bbox[0] - bbox[1]),
                        (bbox[2] - bbox[0]),
                        bbox[3] - bbox[1], fill = False ,
                        edgecolor ='red', linewidth = 3.5)
            )

        ax.text(bbox[0], bbox[1] -2,
            '{:s}{:.3f}'.format(class_name,score),
            bbox = dict(facecolor = 'blue',alpha = 0.5),
            fontsize = 14, color = 'white')
    ax.set_tittle(('{} detections with ' 'p({} | box) >= {:.f}')
            .format(class_name, class_name, thresh), fontsize = 14 )

    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(img):
    im = cv2.imread(img)
    scores,boxes,masks = im_detect(im)

    for cls_ind,cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        cls_boxes = boxes[:,4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:,cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:,np.newaxis])).astype(np.float32)

        keep = nms(dets,0.3)
        dets = dets[keep, :]
        masks = masks[keep,:]
        vis_detectios(im,cls,dets,masks)
    print masks

if __name__ == '__main__':
    image = './img.png'
    demo(image)
    plt.show()
