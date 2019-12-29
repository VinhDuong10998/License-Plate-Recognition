import os
import cv2
import numpy as np

def warp(box, img, w=200, h=64):
    pts1 = np.float32([[[box[0],box[1]]], [[box[2],box[3]]], [[box[6],box[7]]], [[box[4],box[5]]]])
    pts2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    res = cv2.warpPerspective(img, matrix, (w,h))
    
    return res

def str2arr(string):
    re = []
    num =''
    for char in string:
        if char == ',':
            re.append(int(num))
            num =''
        else:
            num += char
    re.append(int(num))
    return re

def checkbox(box, points):

    xs = [box[0], box[2], box[4], box[6]]
    left = min(xs)
    right = max(xs)
    ys = [box[1], box[3], box[5], box[7]]
    top = min(ys)
    bot = max(ys)

    for point in points:
        if point[0] < right and point[0] > left and point[1] < bot and point[1] > top:
            return True
    return False

def cutbox(box, img, n, des):
    xss = [box[0], box[2], box[4], box[6]]
    yss = [box[1], box[3], box[5], box[7]]
    top = min(yss)
    bot = max(yss)
    left = min(xss)
    right = max(xss)
    off = 0
    temp = img[top-off:bot+off, left-off:right+off]
    temp = warp(box,temp)
    cv2.imwrite( des + '/'+ str(n) + '.jpg', temp)
def by_row(box):
   return 1*box[0]/box[1]

def create_temp_box(boxes):
    bigbox = []
    i = 0
    bigbox = [boxes[0][0],boxes[0][1],boxes[-1][2],boxes[-1][3],boxes[-1][-4],boxes[-1][-3],boxes[0][-2],boxes[0][-1]] #top-l, top-r, bot-r, bot-l
    # img = cv2.imread('data/test.jpg')
    # k=0
    # for i in range(0,len(bigbox),2):
    #     print((bigbox[k],bigbox[k+1]))
    #     cv2.circle(img, (bigbox[k],bigbox[k+1]), 1, (255,0,0), -1)
    #     cv2.imshow('demo',img)
    #     cv2.waitKey(0)
    #     k+=2
    return bigbox

def boxes2bigbox(boxes):
    top = 100000
    bot = 0
    left = 100000
    right = 0
    for box in boxes:
        for i in range(0,4):
            if(box[i*2 + 1] < top): 
                top = box[i*2 + 1]
            if(box[i*2] < left):
                left = box[i*2]
            if(box[i*2] > right):
                right = box[i*2]
            if(box[i*2 + 1] > bot):
                bot = box[i*2 + 1]
    return bot, top, left, right

import glob
path = 'result/'
links = glob.glob(path+'/*.txt')

n = 1
idn = 1
bn = 1
for link in links:
    img = cv2.imread('{}/{}.jpg'.format('input/',link.split('/')[-1][4:-4]))
    #img = cv2.imread('/home/phucvinh98/ID_OCR_project/raw_img/data/CCCD_0.jpg')
    height = len(img)
    width = len(img[0])
    PInName1 = []
    for i in range(5):
        PInName1.append((int(width*50/200) + int(i*width/8), int(height*60/200)))

    PInName2 = []
    for i in range(6):
        PInName2.append((int(width*35/200) + int(i*width/8), int(height*150/200)))

    f = open(link, 'r')
    s = f.readlines()
    boxes = []
    for line in s:
        boxes.append(str2arr(line))
    
    name1_boxes = []
    name2_boxes = []
    ID_boxes = []
    bday_boxes = []
    chosen_boxes = []

    for box in boxes:
        if checkbox(box, PInName1) : 
            name1_boxes.append(box)
            chosen_boxes.append(box)
        elif checkbox(box, PInName2) : 
            name2_boxes.append(box)
            chosen_boxes.append(box)

    for ii in range(len(name1_boxes)):
        for jj in range(ii + 1,len(name1_boxes)):
            if(name1_boxes[ii][0] > name1_boxes[jj][0]):
                tem = name1_boxes[ii]
                name1_boxes[ii] = name1_boxes[jj]
                name1_boxes[jj] = tem
                
    if not name1_boxes == []:
            off = 0
            #b,t,l,r = boxes2bigbox(name1_boxes)
            #final_ID = img[t-off:b+off, l-off:r+off]
            temp_box = create_temp_box(name1_boxes)
            w = temp_box[2]-temp_box[0]
            h = temp_box[-1]-temp_box[1]
            final_ID = warp(temp_box,img, w = int(w*64/h))
            cv2.imwrite('output/{}_{}.jpg'.format(link.split('/')[-1][4:-4], idn), final_ID)
            idn += 1

    for ii in range(len(name2_boxes)):
        for jj in range(ii + 1,len(name2_boxes)):
            if(name2_boxes[ii][0] > name2_boxes[jj][0]):
                tem = name2_boxes[ii]
                name2_boxes[ii] = name2_boxes[jj]
                name2_boxes[jj] = tem
        
    if not name2_boxes == []:
            off = 0
            #b,t,l,r = boxes2bigbox(name2_boxes)
            #final_ID = img[t-off:b+off, l-off:r+off]
            temp_box = create_temp_box(name2_boxes)
            
            w = temp_box[2]-temp_box[0]
            h = temp_box[-1]-temp_box[1]
            final_ID = warp(temp_box,img, w = int(w*64/h))
            cv2.imwrite('output/{}_{}.jpg'.format(link.split('/')[-1][4:-4],idn), final_ID)
            idn += 1
            
    
