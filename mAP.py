from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.utils.test import test_net
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from lib.datasets.factory import get_imdb

import xml.dom.minidom

TEST_DIR = "./data/VOCDevkit2007/VOC2007/ImageSets/main"
Annotations_DIR = "./data/VOCDevkit2007/VOC2007/Annotations"

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


CLASSES = ('__background__',
            'mobilephone', 'key', 'powerbank', 'laptop',
            'bottle', 'umbrella')
def vis_detections(image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
 
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        path = os.path.join("./output/test", class_name+".txt")
        fw = open(path,'a')
        fw.write(str(image_name)+' '+str(score)+' '+str(int(bbox[0]))+' '+str(int(bbox[1]))+' '+str(int(bbox[2]))+' '+str(int(bbox[3]))+'\n')
        fw.close()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo', image_name)
    im_file = os.path.join(cfg.FLAGS2["data_dir"], "VOCDevkit2007/VOC2007/JPEGImages/", image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)  ## scores(shape = [300, 7]) 置信度   boxes(shape = [300, 28])
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(image_name, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    args = parser.parse_args()
    return args

def read_xml(filename):
    items = []
    dom = xml.dom.minidom.parse(filename)
    root = dom.documentElement
    itemlist = root.getElementsByTagName('name')
    itemlist = itemlist[1:]
    for item in itemlist:
        items.append(item.childNodes[0].data)
    return items

if __name__ == '__main__':
    args = parse_args()
    # model path
    demonet = args.demo_net
    dataset = args.dataset
    filename = 'vgg16_faster_rcnn_iter_{:d}'.format(cfg.FLAGS.max_iters) + '.ckpt'
    tfmodel = os.path.join("./default/voc_2007_trainval/default", filename)
    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    net.create_architecture(sess, "TEST", 7,
                            tag='default', anchor_scales=[8, 16, 32])
    print("loading model...")
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))
    imdb = get_imdb(args.imdb_name)
    test_net(sess, net, imdb,weights_filename=None ,max_per_image=100, thresh=0.5)

    '''
    im_names = open(os.path.join(TEST_DIR, "test.txt")).readlines()
    m = len(im_names)
    for i in range(m):
        im_name = im_names[i].strip("\n")
        im_name = im_name + ".jpg"
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)
    '''






    
