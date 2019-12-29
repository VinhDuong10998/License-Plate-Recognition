import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from .craft import CRAFT
from collections import OrderedDict
from . import craft_utils
from . import imgproc
from . import file_utils
from . import regconize
from PIL import Image

import glob
import cv2
from skimage import io
import numpy as np

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

'''
canvas_size (int):image size for inference
mag_ratio (float):image magnification ratio
'''
def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image,1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    
    # refine link
    if refine_net is not None:
        y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if True : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



'''
trained_model(str): pretrained model
'''
def model(trained_model,cuda=False,refine=False,refiner_model='weights/craft_refiner_CTW1500.pth'):
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + trained_model + ')')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))
        refine_net.eval()
    return net,refine_net


def run_model(net,refine_net,img,name,text_model,text_threshold=0.7,link_threshold=0.4,low_text=0.4,poly=False,cuda=False):
    t = time.time()
    poly = True

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    # image_list, _, _ = get_files(test_folder)
    # load data
    # for k, image_path in enumerate(image_list):
    #     print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
    #     image = imgproc.loadImage(image_path)
    print(name)
    bboxes, polys, score_text = test_net(net, img, text_threshold, link_threshold, low_text, cuda, poly, refine_net)
    if len(bboxes)>1:
        file_utils.saveResult(name, img[:,:,::-1], polys, dirname=result_folder)
        os.system("python crop_LP.py")
        os.remove("{}/res_{}.txt".format("result",name))
        recog=[]
        recog = regconize.regconizeText("./output",text_model)
        print(regconize.regconizeText("./output",text_model))
        files = glob.glob('output/*.jpg')
        for f in files:
            os.remove(f)
        if len(recog) == 1:
            return '{} {}'.format(recog[0],'None')
        elif len(recog) == 2:
            return '{} {}'.format(recog[0],recog[1])
        else: return 'None'


    # save score text
    # filename, file_ext = os.path.splitext(os.path.basename(image_path))
    # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    # cv2.imwrite(mask_file, score_text)

        
    print("elapsed time : {}s".format(time.time() - t))

