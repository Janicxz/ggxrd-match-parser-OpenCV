# Python program to illustrate 
# template matching

import os

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image


MASKS_DIRPATH = os.path.join(
    os.path.dirname(__file__),
    'templates',
)

class Mask(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.template = cv2.imread('{}/{}'.format(MASKS_DIRPATH, filepath),0)

#PLAYER_MASKS = [
#    Mask('players/{}'.format(filepath))
#    for filepath in os.listdir('{}/players'.format(MASKS_DIRPATH))
#]

#TEST_MASK = Mask('johnny-right.png')
TEST_MASKS = [
    Mask('test/{}'.format(filepath))
    for filepath in os.listdir('{}/test'.format(MASKS_DIRPATH))
]
VS_MASK = Mask('vs.png')

# Read the main image
img_rgb = cv2.imread('test_image.png')

# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
#template = cv2.imread('omito-right_test.png',0)

def matchTemplate(template, image, treshold):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    location = np.where( result >= treshold)

    print(location)
    if len(location[0])!= 0:
        print("Found template.")
        return True
    else:
        print("Didn't find anything")
        return False

videoFileName = "video.webm"
SKIP_SECS = 20
SEEK_SECS = 0.5
if __name__ == '__main__':
    # https://www.youtube.com/watch?v=q3DY8EPtFMY
    clip = VideoFileClip(videoFileName, audio=False)
    sec_matches = []
    next_sec = 0

    for sec, clip_frame in clip.iter_frames(with_times=True,dtype="uint8"):
        if sec < next_sec:
            continue
       # clip_frame_rgb = clip_frame.flatten()
        print(clip_frame)
        clip_frame_bgr = cv2.cvtColor(clip_frame, cv2.COLOR_RGB2BGR)
        clip_frame_gray = cv2.cvtColor(clip_frame, cv2.COLOR_RGB2GRAY)
       # nparray = np.fromstring(clip_frame, np.uint8)
        #img_np = cv2.imdecode(clip_frame, cv2.IMREAD_UNCHANGED)
        #print(nparray)
        for test_mask in TEST_MASKS:
            #test_mask.template = cv2.imread(test_mask.filepath,0)
            #print(test_mask.template)
            print(test_mask.filepath)

            w, h = test_mask.template.shape[::-1] # invert W, H. shape returns them inverted.
            result = cv2.matchTemplate(clip_frame_gray,test_mask.template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            location = np.where( result >= threshold) 
            if len(location[0])!= 0:
                print("Found template: ", test_mask.filepath)
            else:
                print("Didn't find anything")
            # Draw a rectangle around the matched region.
            for pt in zip(*location[::-1]):
                cv2.rectangle(clip_frame_bgr, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
        
        cv2.imshow('Frame',clip_frame_bgr)
        cv2.waitKey(0)
        next_sec = sec + SEEK_SECS
        
    template = cv2.imread('templates/test/johnny-right.png',0)
    # Store width and heigth of template in w and h
    w, h = template.shape[::-1]

    # Perform match operations.
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

    # Specify a threshold
    threshold = 0.8

    # Store the coordinates of matched area in a numpy array
    loc = np.where( res >= threshold) 

    if len(loc[0])!= 0:
        print("Found player.")
    else:
        print("Didn't find anything")
    print(loc)
    # Draw a rectangle around the matched region.
    #for pt in zip(*loc[::-1]):
    #    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

    # Show the final image with the matched area.
    #cv2.imshow('Template',template)
    cv2.imshow('Detected',img_rgb)
    
    cv2.waitKey(0)