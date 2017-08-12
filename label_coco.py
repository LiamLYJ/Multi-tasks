from libs.datasets.pycocotools.coco import COCO
import numpy as np
import cv2
import easygui
head_top = [0,0,0]
neck_top = [0,0,0]
flag = 0

def click(event,x,y,flags,param):
    global head_top
    global neck_top
    global flag
    if event == cv2.EVENT_LBUTTONDOWN :
        if flag == 0 :
            head_top = [x, y,1]
            flag = flag + 1
        elif flag == 1:
            neck_top = [x,y,1]
            flag = flag + 1
        else:
            easygui.msgbox('u clicked three times, plz press 'R' to re-lable')

annFile = '/home/hpc/ssd/lyj/Multi-tasks/data/coco/annotations_2/person_keypoints_train2014.json'
coco = COCO(annFile)
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds = catIds)
file_head = '/home/hpc/ssd/lyj/Multi-tasks/data/coco/train2014/'
# imgIds = [15151,393207,262136,524273,131058]
print ('num_ all :',len(imgIds))
GT_id,GT_box,GT_kp = [],[],[]
# load the formal labelled data
tmp = np.load('annotation_tmp.npz')
GT_id = list(tmp['id'])
GT_box = list(tmp['box'])
GT_kp = list(tmp['kp'])
_counter = tmp['counter']

counter = 0
for index,imgId in enumerate(imgIds):
    counter = index

    if counter < _counter  :
        continue

    print ('index / num_all/ imgId:', '%d/%d/%d'%(counter,len(imgIds),imgId))
    try:
        data = coco.loadImgs(imgId)[0]

        annIds = coco.getAnnIds(imgIds = data['id'])
        ann = coco.loadAnns(annIds)
        damp = True
        i = 0
        gt_bboxs = []
        gt_kps = []

        while (i <len(ann)):
            flag = 0
            head_top = [0,0,0]
            neck_top = [0,0,0]
            num_joints = ann[i]['num_keypoints']
            bbox = ann[i]['bbox']
            kps = ann[i]['keypoints']
            if num_joints == 0:
                i = i+1
                continue
            else:
                # print (bbox)
                damp = False
                img = cv2.imread(file_head + data['file_name'])
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[0] + bbox[2]
                y2 = bbox[1] + bbox[3]
                cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0))
                cv2.namedWindow('img')
                cv2.setMouseCallback('img', click)
                cv2.imshow('img',img)
                kps_update = kps[:]
                i = i + 1
                key = cv2.waitKey(0) & 0xFF
                kps_update = kps_update + head_top + neck_top
                gt_bboxs.append([x1,y1,x2,y2,1])
                gt_kps.append(kps_update)
                if key == ord('r'):
                    print ('plz, label this again')
                    i = i -1
                    gt_kps.pop()
                    gt_bboxs.pop()
                if key == ord('p'):
                    print (' pass this box')
                    gt_kps.pop()
                    gt_bboxs.pop()
                if key == ord('m'):
                    print ('mute head_top and neck_top in this box')
                if key == ord('q'):
                    raise
                if key == ord('k'):
                    damp = True
                    break
                if key == ord('b'):
                    print (' go back to former one')
                    i = i -2
                    gt_kps.pop()
                    gt_bboxs.pop()
                    gt_kps.pop()
                    gt_bboxs.pop()
                # print ('head_top:', head_top)
                # print ('neck_top:', neck_top)
        if damp or (len(gt_bboxs) == 0):
            continue
        assert len(gt_bboxs) == len(gt_kps)
        GT_box.append(gt_bboxs)
        GT_kp.append(gt_kps)
        GT_id.append(imgId)
    except :
        print ('ops, looks like somethings bad happend')
        np.savez('annotation.npz', box=np.array(GT_box),kp=np.array(GT_kp),id=np.array(GT_id),counter = np.array(counter))
        print ('saved once intermidatly')
        raise
assert len(GT_box) == len(GT_kp) and len(GT_kp) == len(GT_id)
# bbox [x1,y1,x2,y2,1(person cat_id)]
# kp :19 = origial 17 + 2 poitns
# id : img_id in COCO data
# counter : end index for imgIds
np.savez('annotation.npz', box=np.array(GT_box),kp=np.array(GT_kp),id=np.array(GT_id),counter = np.array(counter))
