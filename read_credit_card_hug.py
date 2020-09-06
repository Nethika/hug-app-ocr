# USAGE
# python read_credit_card.py --image images/credit_card_01.png

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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



from scanner import scan
from scan import *
import string
import cv2
import imutils

global class_graph
global detect_graph
global net
global sess



# init session
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)


cfg_from_file('ctpn/text.yml')

print ("ctpn is loading ............")
print ("##################################")


# load network
net = get_network("VGGnet_test")
# load model
print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
saver = tf.train.Saver()

detect_graph = tf.get_default_graph()

print ("classification model is loading ............")
print ("##################################")
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


def find_score(info_type,text):

    if info_type == 'expire':
        #print('/ + numbers!')

        chars = [s for s in text]

        if "/" in chars:
            s_1 = 0.8
        else:
            s_1 = 0.1

        rest = set(chars) - set(["/"])
        score_ex =[]
        for ch in rest:
            if ch in string.digits:
                s_2 = 0.8
            else:
                s_2 = 0.1
            score_ex.append(s_2)

        score_ex.append(s_1)

        if len(text) == 5:
            s_3 = 0.8
        else:
            s_3 = 0.01
        score_ex.append(s_3)

        score = np.mean(score_ex)
        print("score:",score)


    if info_type == 'name':
        #print('everything should be chars!')
        chars = [s for s in text]
        score_na =[]
        for ch in chars:
            if ch.isalpha():
                s_3 = 0.8
            elif ch == " ":
                s_3 = 0.5
            else:
                s_3 = 0.1
            score_na.append(s_3)

        score = np.mean(score_na)

                
    if info_type == 'number':
        #print('everything should be numbers!')    

        chars = [s for s in text]
        score_num =[]
        for ch in chars: 
            if ch in string.digits:
                s_1 = 0.8
            elif ch == " ":
                s_1 = 0.8
            else:
                s_1 = 0.1
            score_num.append(s_1)
        if len(text) == 19:
            s_2 = 0.8
        else:
            s_2 = 0.2
        score_num.append(s_2)
        score = np.mean(score_num)
        print("score:",score)

    return score



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

    base_name = os.path.basename(image_name)
        
    img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)
    
def ctpn(net, img,image_name):
    global detect_graph
    #global net
    global sess

    timer = Timer()
    timer.tic()

    #img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    with detect_graph.as_default():
        scores, boxes = test_ctpn(sess, net, img)

        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        draw_boxes(img.copy(), image_name, boxes, scale)
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






def read_credit_card(im_path,unique_id,f_or_b):
    global detect_graph
    global class_graph
    global net
    global sess
    """
    crop_image_path = "app/static/output/" +class_name+ str(j) +"_" + unique_id 
    cv2.imwrite(crop_image_path, crop_img)
    """

    print("Image path:",im_path)

    ##### Preprocess Image

  
    # scanner
    """
    scanner = DocScanner()

    sharp_im, thresh_1 = scanner.scan(im_path)


    pre_image = thresh_1
    """



    img = cv2.imread(im_path)  # Read input image
    print("@@@@@@@@ im_path:",im_path)

    # Convert from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the saturation color channel - all gray pixels are zero, and colored pixels are above zero.
    s = hsv[:, :, 1]

    # Convert to binary using automatic threshold (use cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pre_image = thresh

    save_processed_path = "app/static/output/processed_" + f_or_b  + unique_id 

    cv2.imwrite(save_processed_path, pre_image)

    """
    # Find contours (in inverted thresh)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    # Find the contour with the maximum area.
    c = max(cnts, key=cv2.contourArea)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(c)

    # Crop the bounding rectangle out of thresh
    thresh_card = thresh[y:y+h, x:x+w].copy()

    pre_image = thresh_card

    """




    # read labels
    label_path = "data/labels.txt"
    label_map = json.load(open(label_path))
    labels = list(label_map.keys())



    

    result_dict ={}
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(('Classification for {:s}'.format(im_path)))


    base_name = os.path.basename(im_path)

    # get preprocessed image (pre_image) from scanner

    #convert to 3 chanels
    img2 = cv2.merge((pre_image,pre_image,pre_image))
    #img2 = pre_image

    with detect_graph.as_default():
        #boxes,img,scale = ctpn(sess, net, im_path)
        boxes,img,scale = ctpn( net, img2,im_path)

    save_detection_path = "app/static/output/after_detection"  +f_or_b+ unique_id 

    cv2.imwrite(save_detection_path, img)



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
        #print("Type:", info_type)

        

        # Read
        #crop_img = cv2.resize(crop_img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
        #gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        gray = crop_img
        #gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #gray = cv2.medianBlur(gray, 3)
        

        img_new = Image.fromarray(gray)

        save_name = "data/text_read/" +f_or_b+ info_type + "_" + base_name
        img_new.save(save_name)

        save_image_path = "app/static/output/" +f_or_b+ info_type +"_" + unique_id 
        img_new.save(save_image_path)

        t_dict ={}
        if info_type != "other":
            print("*********************************")
            print("Type:",info_type)
            print("")
            print("Model: LSTM") 
            text1 = pytesseract.image_to_string(img_new, lang='eng',config='--psm 7 --oem 1')
            score1 = find_score(info_type,text1)
            print("  text1:", text1)
            print("  score1:", score1)
            t_dict[text1] = score1

            lang = ['engmorse','digits','mcr','osd','ocr', 'equ']

            for ll in lang:  
                print("Model:",ll)              
                text_ = pytesseract.image_to_string(img_new, lang= ll)
                score_ = find_score(info_type,text_)
                print("  text_:", text_)
                print("  score_:", score_)
                print("")
                t_dict[text_] = score_


            text  = max(t_dict, key=t_dict.get)
            score = t_dict[text]
            print(" max text:", text)
            print(" max score:", score)            
            print("")



            result_dict[info_type]= text


        

    print("RESULT:",result_dict)
    print("")


    if 'expire' not in result_dict:
        print('expire not found!')


    if 'name' not in result_dict:
        print('name not found!')

    if 'number' not in result_dict:
        print('number not found!')

    return save_detection_path, result_dict





