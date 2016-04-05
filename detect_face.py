import time
import os
import cv2
import numpy as np
import urllib
import re
import logging
import pdb

from facepp import API,File
from person_info import PERSON_INFO


logger = logging.getLogger()
fh = logging.FileHandler("face_detection.log")
logger.addHandler(fh)
logger.setLevel(logging.NOTSET)
logger.info('face detection')

ALIGN_POINTS = list(range(0,25))
OVERLAY_POINTS=list(range(0,25))

RIGHT_EYE_POINTS = list(range(2, 6))
LEFT_EYE_POINTS = list(range(4, 8))

FEATHER_AMOUNT = 11
SCALE_FACTOR = 1
COLOUR_CORRECT_BLUR_FRAC = 0.6

HERE = os.path.dirname(__file__)

classifier=cv2.CascadeClassifier(r"..\haarcascades\haarcascade_frontalface_alt.xml")

def encode(obj):
    if type(obj) is unicode:
        return obj.encode('utf-8')
    if type(obj) is dict:
        return {encode(k): encode(v) for (k, v) in obj.iteritems()}
    if type(obj) is list:
        return [encode(i) for i in obj]
    return obj

def drawText(frame, person_name, person_title, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, person_name, (x, y-30), font, 0.5, (255,0,0))
    cv2.putText(frame, person_title, (x, y-10), font, 0.5, (0,0,255))


def get_max_possibilities_person_name(face_list, index):
    index_person_names = []
    for face in face_list:
        for k,v in face.iteritems():
            if k == index:
                index_person_names.append(v)
    
    person_names = []
    for item in index_person_names:
        person_names.append(item.values()[0])
    
    return get_max_possibilities_person(person_names)

def get_max_possibilities_person(person_names):        
    person_names_set = set(person_names)
    person_names_count = {}
    for item in person_names_set:
        person_names_count[item] = person_names.count(item)
    person_names_count = {v:k for k, v in person_names_count.iteritems()}
    plist = list(person_names_count.keys())
    
    return person_names_count[max(plist)]

def get_index_len(face_list, index):
    len = 0
    for face in face_list:
        for k,v in face.iteritems():
            if k == index:
                len += 1
    
    return len

def add_person_to_group(group_name, is_person_exists = False):    
    group_path = os.path.join(HERE, 'group')
    added_person = []
    for directory in os.listdir(group_path):
        # create a new group and add those persons in it
        
        first_person = True
        for _file in os.listdir(os.path.join(group_path, directory)):
            img_name = os.path.join(HERE, 'group', directory, _file)
            result = api.detection.detect(img=File(img_name), post=True)
            if result['face']:
                if first_person:
                    if not is_person_exists:
                    # create person using the face_id
                        rst = api.person.create(person_name = directory,
                                                face_id = result['face'][0]['face_id']
                                                )
                    first_person = False
                rst = api.group.add_person(group_name = group_name, person_name = directory)
                
        added_person.append(directory)

def group_remove_person(group_name, person_name):
    api.group.remove_person(group_name = group_name, person_name = person_name)

def create_group(group_name):
    api.group.create(group_name = group_name)
            
def train(group_name):
    # train the model
    rst = api.train.identify(group_name = group_name)
    # wait for training to complete
    rst = api.wait_async(rst['session_id'])
        
                    
if __name__ == '__main__':

    api = API(API_KEY, API_SECRET)
    
    group_name = 'friend'
    
    isFirst = 1
    if isFirst == 1:
        # This is for you want to remove preson from one group, becase you want to add the person again
        group_remove_person(group_name = 'friend', person_name="FengXi")
        add_person_to_group(group_name, is_person_exists=True)
        train(group_name)
    elif isFirst == 2:
        # This is for the person that hasn't been added to the train data
        add_person_to_group(group_name)
        train(group_name)

    isCapture = True
    isRestart = False
    isSlowmode = False
    isDrawText = False
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cap=cv2.VideoCapture(0)
    success,frame=cap.read()

    i = 0
    face_list = []
    
    while success:
        # capture frame-by-frame
        success,frame=cap.read()
        if isCapture:
            if isRestart:
                face_list = []
                isRestart = False
            if isSlowmode:
                face_list = []
            size=frame.shape[:2]
            image=np.zeros(size,dtype=np.float16)
            # operation on the frame come here
            image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image,image)
            
            divisor=8
            h,w=size
            
            minSize=(w/divisor, h/divisor)
            faceRects=classifier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, minSize)
            previous_person = {}
            face_len = len(faceRects)
            if face_len>0:
                for index, faceRect in enumerate(faceRects):
                    x,y,w,h = faceRect
                    # The last parameter of rectangle() is thickness of the rectangle                    
                    roi = image[y:y+h, x:x+w]
                    face_info = {}
                    
                    folder = os.path.join(HERE, 'face', str(index))
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    face_name = os.path.join(folder, str(i) + '_.jpg')    
                    
                    cv2.imwrite(face_name, roi)   
                    face_info[index] = {}
                    
                    if get_index_len(face_list, index) > 10:
                        person_name = get_max_possibilities_person_name(face_list, index)
                        #logger.info("person_name=%s" %person_name)q
                        person_title = PERSON_INFO[person_name]
                        if isDrawText:
                            drawText(frame, person_name, person_title, x, y)
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
                    else:
                        rst = api.recognition.identify(group_name = 'friend', img = File(face_name), async=False)
                        
                        if rst['face']:
                            person_name = rst['face'][0]['candidate'][0]['person_name']
                            previous_person[index] = person_name
                            face_info[index][i] = person_name   
                            if get_index_len(face_list, index) <= 10: 
                                face_list.append(face_info)                    
                            person_title = PERSON_INFO[person_name]
#                            logger.info("face_info=%s" %face_info)
#                            logger.info("face_list=%s" %face_list)
                            
                            if isDrawText:
                                drawText(frame, person_name, person_title, x, y)                   
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)    
                                    
            i += 1    
        cv2.imshow("Camera", frame)
         
        key=cv2.waitKey(5)
        c=chr(key&255)
        if c in ['q','Q']:
            break
        if c in [chr(27)]:
            isCapture = False
        if c in ['a', 'A']:
            isCapture = True
        if c in ['r', 'R']:
            isRestart = True
        if c in ['s', 'S']:
            isSlowmode = True
        if c in ['d', 'D']:
            isDrawText = True
    cv2.destroyWindow("test")
