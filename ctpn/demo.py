# USAGE
# python ./ctpn/demo.py --image images/credit_card_01.png

from __future__ import print_function
import tensorflow as tf
from tensorflow import Graph
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

import pytesseract
from PIL import Image

from keras.models import load_model

import os
import json

from imutils import contours
import imutils

import argparse


global class_graph
global detect_graph




def classify(img, c_model):
    global class_graph
    """ classifies images in a given folder using the 'model'"""

    #img = load_img(im_path,target_size=(input_height, input_width))
    #img = img_to_array(img)
    im_size = 128
    # resize 

    img = cv2.resize(img, (im_size,im_size))

    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)

    with class_graph.as_default():

        predictions = c_model.predict(img)[0]

    return predictions


def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f

def draw_boxes(img,image_name,boxes,scale):
    for box in boxes:
        if box[8]>=0.9:
            color = (0,255,0)
        elif box[8]>=0.8:
            color = (255,0,0)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

    #base_name = image_name.split('/')[-1]
    base_name = image_name.split('\\')[-1]
    img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)
    
def ctpn(sess, net, image_name):
    global detect_graph
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    with detect_graph.as_default():
        scores, boxes = test_ctpn(sess, net, img)

        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        draw_boxes(img, image_name, boxes, scale)
        timer.toc()
        print(('Detection took {:.3f}s for '
            '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

        return boxes,img,scale   

def font_digits(ref_image_path):

	# load the reference OCR-A image from disk, convert it to grayscale,
	# and threshold it, such that the digits appear as *white* on a
	# *black* background
	# and invert it, such that the digits appear as *white* on a *black*
	#ref = cv2.imread(args["reference"])
	ref = cv2.imread(ref_image_path)
	ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
	ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

	# find contours in the OCR-A image (i.e,. the outlines of the digits)
	# sort them from left to right, and initialize a dictionary to map
	# digit name to the ROI
	refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
	refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
	digits = {}

	# loop over the OCR-A reference contours
	for (i, c) in enumerate(refCnts):
		# compute the bounding box for the digit, extract it, and resize
		# it to a fixed size
		(x, y, w, h) = cv2.boundingRect(c)
		roi = ref[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))

		# update the digits dictionary, mapping the digit name to the ROI
		digits[i] = roi

	return digits





if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")

    args = vars(ap.parse_args())

    im_name = args["image"]

    # Method 1 : Credit card Number

    digits1 = font_digits("data/Credit-Card0.png")
    digits2 = font_digits("data/ocr_a_reference.png")


    #############################

    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # load the input image, resize it, and convert it to grayscale
    #print(args["image"])
    image = cv2.imread(args["image"])

    #print(image)
    image = imutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply a tophat (whitehat) morphological operator to find light
    # regions against a dark background (i.e., the credit card numbers)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
        ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    locs = []

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # since credit cards used a fixed size fonts with 4 groups
        # of 4 digits, we can prune potential contours based on the
        # aspect ratio
        if ar > 2.5 and ar < 4.0:
            # contours can further be pruned on minimum/maximum width
            # and height
            if (w > 40 and w < 55) and (h > 10 and h < 20):
                # append the bounding box region of the digits group
                # to our locations list
                locs.append((x, y, w, h))

    # sort the digit locations from left-to-right, then initialize the
    # list of classified digits
    locs = sorted(locs, key=lambda x:x[0])
    output1 = []
    output2 = []

    # loop over the 4 groupings of 4 digits
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        groupOutput1 = []
        groupOutput2 = []

        # extract the group ROI of 4 digits from the grayscale image,
        # then apply thresholding to segment the digits from the
        # background of the credit card
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # detect the contours of each individual digit in the group,
        # then sort the digit contours from left to right
        digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
        digitCnts = contours.sort_contours(digitCnts,
            method="left-to-right")[0]


        # loop over the digit contours
        for c in digitCnts:
            # compute the bounding box of the individual digit, extract
            # the digit, and resize it to have the same fixed size as
            # the reference OCR-A images
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))

            # initialize a list of template matching scores	
            scores1 = []

            scores2 = []

            # loop over the reference digit name and digit ROI
            for (digit, digitROI) in digits1.items():
                # apply correlation-based template matching, take the
                # score, and update the scores list
                result = cv2.matchTemplate(roi, digitROI,
                    cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores1.append(score)

            for (digit, digitROI) in digits2.items():
                # apply correlation-based template matching, take the
                # score, and update the scores list
                result = cv2.matchTemplate(roi, digitROI,
                    cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores2.append(score)

            # the classification for the digit ROI will be the reference
            # digit name with the *largest* template matching score
            groupOutput1.append(str(np.argmax(scores1)))
            groupOutput2.append(str(np.argmax(scores2)))

        # draw the digit classifications around the group
        cv2.rectangle(image, (gX - 5, gY - 5),
            (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput1), (gX, gY - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # update the output digits list
        output1.extend(groupOutput1)
        output2.extend(groupOutput2)


    output1 =''.join(output1)
    output1 =' '.join([output1[i:i+4] for i in range(0, len(output1), 4)])

    output2 =''.join(output2)
    output2 =' '.join([output2[i:i+4] for i in range(0, len(output2), 4)])

    print ("METHOD 1")
    print ("############")
    print("card number 1:",output1)
    print("card number 2:",output2)



    # Method 2 : Other info

    cfg_from_file('ctpn/text.yml')

    print ("METHOD 2")
    print ("############")

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    detect_graph = tf.get_default_graph()

    # load model
    model_path = "data/credit-card.model"
    class_model = load_model(model_path)

    class_graph = tf.get_default_graph()



    with detect_graph.as_default():

        try:
            ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('done')
        except:
            raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

        im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
        for i in range(2):
            _, _ = test_ctpn(sess, net, im)

    # read labels
    label_path = "data/labels.txt"
    label_map = json.load(open(label_path))
    labels = list(label_map.keys())



    

    result_dict ={}
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(('Demo for {:s}'.format(im_name)))

    base_name = im_name.split('\\')[-1]

    with detect_graph.as_default():
        boxes,img,scale = ctpn(sess, net, im_name)


    xx= 0
    for box in boxes:
        xx=xx+1
        x= int(box[0])
        y= int(box[1])
        w= int(box[2]) - int(box[0])
        h= int(box[5]) - int(box[1])
        #print ("x,y,w,h:", x,y,w,h)
        im_to_crop = img.copy()
        crop_img = im_to_crop[y:y+h, x:x+w]
        
        with class_graph.as_default():
            predictions = classify(crop_img, class_model)



        m_index = predictions.argmax()


        info_type =  labels[m_index]
        print("Type:", info_type)

        

        # Read
        #crop_img = cv2.resize(crop_img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
        #gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        gray = crop_img
        #gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #gray = cv2.medianBlur(gray, 3)
        

        img_new = Image.fromarray(gray)


        text = pytesseract.image_to_string(img_new, lang='eng',config='--psm 7 --oem 1')


        
        print("  text:", text)

        
        if info_type != "other":
            result_dict[info_type]= text

    #print("RESULT:",result_dict)
    #print("")




