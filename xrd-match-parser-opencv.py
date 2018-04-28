import os
import datetime
import argparse
import subprocess

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image


MASKS_DIRPATH = os.path.join(
    os.path.dirname(__file__),
    'templates',
)

class Match(object):
    def __init__(self, timeStamp, charLeft, charRight, playerOne, playerTwo):
        self.timeStamp = timeStamp
        self.charLeft = charLeft
        self.charRight = charRight
        self.playerOne = playerOne
        self.playerTwo = playerTwo

class Mask(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.template = cv2.imread('{}/{}'.format(MASKS_DIRPATH, filepath),0)

PLAYER_MASKS = [
    Mask('players/{}'.format(filepath))
    for filepath in os.listdir('{}/players'.format(MASKS_DIRPATH))
]
CHARACTER_MASKS = [
    Mask('characters/{}'.format(filepath))
    for filepath in os.listdir('{}/characters'.format(MASKS_DIRPATH))
]
VS_MASK = Mask('vs.png')
INSERT_COIN_MASK = Mask('insert-coin-left.png')

FILE_OUTPUT = "matches.html"

# Read the main image
img_rgb = cv2.imread('test_image.png')

# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
#template = cv2.imread('omito-right_test.png',0)

def matchTemplate(template, image, treshold):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    location = np.where( result >= treshold)

    #print(location)
    if len(location[0])!= 0:
       # print("Found template.")
        return True
    else:
       # print("Didn't find anything")
        return False

def format_timestamp(secs):
    return '[{}]'.format(
        str(datetime.timedelta(seconds=int(secs))),
    )

#URL = "https://www.youtube.com/watch?v=q3DY8EPtFMY"
def writeToHtml(foundMatches, youtubeUrl):
    youtubeVideoCode = youtubeUrl.split('=')[-1]
    htmlFile = open(FILE_OUTPUT, "w")
    htmlFile.write('<iframe width="420" height="345" src="https://www.youtube.com/embed/{}"></iframe><br>'.format(youtubeVideoCode))
    for match in foundMatches:
        htmlFile.write('<a href={}#t={}>{} {} ({}) vs {} ({})</a><br>\n'.format(
            youtubeUrl,
            match.timeStamp,
            format_timestamp(match.timeStamp),
            match.charLeft,
            match.playerOne,
            match.charRight,
            match.playerTwo,
            ))
    htmlFile.close()
#def findPlayerName(filename):
#     player_name = lambda s: os.path.basename(s).split('-')[0]



videoFileName = "video.webm"
SKIP_SECS = 20
SEEK_SECS = 0.5
if __name__ == '__main__':
    # https://www.youtube.com/watch?v=q3DY8EPtFMY test clip

    parser = argparse.ArgumentParser(description="Parse XRD matches")
    parser.add_argument(
        'youtube_url',
        type=str,
        help='youtube video URL (e.g. https://www.youtube.com/watch?v=fOvG_TfnCVo)',
    )
    parser.add_argument(
        '--already-downloaded',
        action='store_true',
        help='use existing youtube video file already on disk',
    )
    args = parser.parse_args()

    # delete previous file
    if os.path.exists(videoFileName) and not args.already_downloaded:
        os.remove(videoFileName)

    # download the video with youtube-dl
    if not args.already_downloaded: 
        subprocess.check_call(
        'youtube-dl --format "bestvideo[height<=240]" --no-continue --output {} '
        '"{}"'.format(
            videoFileName,
            args.youtube_url,
        ),
        shell=True,
        )

    clip = VideoFileClip(videoFileName, audio=False)
    foundMatches = []
    next_sec = 0

    for sec, clip_frame in clip.iter_frames(with_times=True,dtype="uint8"):
        if sec < next_sec:
            continue

        # Convert to BRG so we can display it properly for debug.
        #clip_frame_bgr = cv2.cvtColor(clip_frame, cv2.COLOR_RGB2BGR)
        # Convert to grayscale for faster analysis
        clip_frame_gray = cv2.cvtColor(clip_frame, cv2.COLOR_RGB2GRAY)

        # Found training mode, skip it
        if matchTemplate(INSERT_COIN_MASK.template, clip_frame_gray, 0.8):
            next_sec = sec + 8
            continue

        
        # Found VS screen
        if matchTemplate(VS_MASK.template, clip_frame_gray, 0.8):
            #print("found VS")
            foundMatch = Match(0, "unknown","unknown","unknown","unknown")
            #Search for all characters
            ###########
            for char_mask in CHARACTER_MASKS:
                #We already found both characters, no need to continue searching.
                if foundMatch.charLeft != "unknown" and foundMatch.charRight != "unknown":
                    #debug print("both characters found!")
                    continue

                w, h = char_mask.template.shape[::-1] # invert W, H. shape returns them inverted.
                result = cv2.matchTemplate(clip_frame_gray,char_mask.template,cv2.TM_CCOEFF_NORMED)
                threshold = 0.7

                location = np.where( result >= threshold) 
                if len(location[0])!= 0:
                    #debug print("Found template: ", char_mask.filepath," At time: ", sec)
                    if '-left' in os.path.basename(char_mask.filepath):
                        foundMatch.charLeft = os.path.basename(char_mask.filepath).split('-')[0]
                        #debug print("Character left: ", foundMatch.charLeft)
                    else:
                        foundMatch.charRight = os.path.basename(char_mask.filepath).split('-')[0]
                        #debug print("Character right: ", foundMatch.charRight)
                    #os.path.basename(char_mask.filepath).split('-')[0]
                    # Draw a rectangle around the matched region.
                    #for pt in zip(*location[::-1]):
                    #    cv2.rectangle(clip_frame_bgr, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
            
            #Search for all known player names
            ##########
            for player_mask in PLAYER_MASKS:
                w, h = player_mask.template.shape[::-1] # invert W, H. shape returns them inverted.
                result = cv2.matchTemplate(clip_frame_gray,player_mask.template,cv2.TM_CCOEFF_NORMED)
                threshold = 0.8

                location = np.where( result >= threshold) 
                if len(location[0])!= 0:
                    #debug print("Found template: ", player_mask.filepath," At time: ", sec)
                    if '-left' in os.path.basename(player_mask.filepath):
                        foundMatch.playerOne = os.path.basename(player_mask.filepath).split('-')[0]
                        #debug print("Player one: ", foundMatch.playerOne)
                    else:
                        foundMatch.playerTwo = os.path.basename(player_mask.filepath).split('-')[0]
                        #debug print("Player two: ", foundMatch.playerTwo)
                    #os.path.basename(char_mask.filepath).split('-')[0]
                    # Draw a rectangle around the matched region.
                    #for pt in zip(*location[::-1]):
                    #    cv2.rectangle(clip_frame_bgr, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
            foundMatch.timeStamp = sec
            print("Found match, {} ({}) vs {} ({}) at {}".format(foundMatch.charLeft, 
            foundMatch.playerOne, foundMatch.charRight, foundMatch.playerTwo, format_timestamp(foundMatch.timeStamp)
            ))
            foundMatches.append(foundMatch)

           # cv2.imshow('Frame',clip_frame_bgr)
            #cv2.waitKey(0)
            
            next_sec = sec + SKIP_SECS
        else:
            next_sec = sec + SEEK_SECS
    print("Writing to html.")
    writeToHtml(foundMatches, args.youtube_url)