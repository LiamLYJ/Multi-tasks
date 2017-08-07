from libs.datasets.pycocotools.coco import COCO
import numpy as np
import cv2

head_top = []
neck_top = []
flag = True
def click(event,x,y,flags,param):
    global head_top
    global neck_top
    global flag
    if event == cv2.EVENT_LBUTTONDOWN :
        if flag:
            head_top = [x, y,1]
            flag = False
        else:
            neck_top = [x,y,1]
            flag = True

annFile = '/home/hpc/ssd/lyj/Multi-tasks/data/coco/annotations_2/person_keypoints_train2014.json'
coco = COCO(annFile)
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds = catIds)
file_head = '/home/hpc/ssd/lyj/Multi-tasks/data/coco/train2014/'
# imgIds = [15151,393207,262136,524273,131058]
print ('num_ all :',len(imgIds))
GT_id,GT_box,GT_kp = [],[],[]
counter = 0
for index,imgId in enumerate(imgIds):
    counter = index
    try:
        data = coco.loadImgs(imgId)[0]

        annIds = coco.getAnnIds(imgIds = data['id'])
        ann = coco.loadAnns(annIds)
        damp = True
        i = 0
        gt_bboxs = []
        gt_kps = []

        while (i <len(ann)):
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
                gt_bboxs.append([x1,y1,x2,y2,1])
                cv2.namedWindow('img')
                cv2.setMouseCallback('img', click)
                cv2.imshow('img',img)
                kps_update = kps[:]
                key = cv2.waitKey(0) & 0xFF
                kps_update = kps_update + head_top + neck_top
                i = i + 1
                gt_kps.append(kps_update)
                if key == ord('r'):
                    print ('plz, label this again')
                    i = i -1
                    gt_kps.pop()
                    gt_bboxs.pop()
                if key == ord('p'):
                    print (' skip this box')
                    gt_kps.pop()
                    gt_bboxs.pop()
                if key == ord('q'):
                    raise
                if key == ord('k'):
                    damp = True
                    break
                # print ('head_top:', head_top)
                # print ('neck_top:', neck_top)
        if damp:
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
