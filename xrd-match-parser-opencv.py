import os
import datetime
import argparse
import subprocess
import time

import cv2
import numpy as np
#from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

# TODO:
# too many false positives on 720P char name matching.. quality STILL too low for this??
# backport char/player search functions to 240p ver.

# Optimize: restrict template matching search area? RESULT: does not seem to be any faster.
# extract ranks from the vid? (color at least? translate the ranks from VS screen?) (green(1-?), blue(?-21?), red(?-24), gold)
# RESULT: 240p and below too low quality to extract rank colors.
# Look into 360p/480p processing? parse from ROUND 1 screen instead of VS?
# if we're parsing from round start, could look for side independent char template,
# and check the template match position (left/right side of the screen?)
# Save matches to sqlite database instead of html file
# Check if we've processed the vod already earlier (sqlite?)
# Dynamic site that loads from the db?
# Fetch latest videos from joniosan YT channel with RSS?

MASKS_DIRPATH = os.path.join(
    os.path.dirname(__file__),
    'templates',
)
#USE_OPENCL = cv2.ocl.haveOpenCL()
USE_OPENCL = False

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
        if USE_OPENCL: self.template = cv2.UMat(cv2.imread('{}/{}'.format(MASKS_DIRPATH, filepath),0))
        else: self.template = cv2.imread('{}/{}'.format(MASKS_DIRPATH, filepath),0)

PLAYER_MASKS = [
    Mask('players/{}'.format(filepath))
    for filepath in os.listdir('{}/players'.format(MASKS_DIRPATH))
]
CHARACTER_MASKS = [
    Mask('characters/{}'.format(filepath)) #240P
    for filepath in os.listdir('{}/characters'.format(MASKS_DIRPATH)) #240P
]
VS_MASK = Mask('vs.png') #240P

TRAINING_MODE_MASKS = [
    Mask('insert_coin_left.png'),
    Mask('insert_coin_right.png'),
    Mask('vs_ai_opponent.png')
]
PRESS_START_MASK = Mask('press_start.png')
ROUND_START = Mask('round_timer_99.png') #240P
FILE_OUTPUT = "matches.html"

def matchTemplate(template, image, threshold, return_location=False):
    # _NORMED returns correlation based value from 1.0 to 0.0 (1.0 being perfect match)
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
       # print("Found template.")
        if return_location:
           return max_loc
        return True
    else:
       # print("No template matched.")
        return False

def format_timestamp(secs):
    return '[{}]'.format(
        str(datetime.timedelta(seconds=int(secs))),
    )

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

def downloadFrames(foundMatches, url):
    #TODO: Delete previously downloaded frames
    count = 0
    for match in foundMatches:
        downloadUrl = subprocess.check_output('youtube-dl --format "bestvideo[height=720]" -g {}'.format(url)).decode('utf-8').strip()
        subprocess.check_call('ffmpeg -ss {} -i "{}" -t 1 -map 0:v -c:v libvpx frames/frame_{}.webm'.format(
            str(datetime.timedelta(seconds=int(match.timeStamp))),
            downloadUrl,
            count
        ))
        count = count+1

def searchForChars(image, FoundMatch):
    foundMatch = FoundMatch
    for char_mask in CHARACTER_MASKS:
        if foundMatch.charLeft != "unknown" and foundMatch.charRight != "unknown":
            #print("both characters found!")
            break
        location = matchTemplate(char_mask.template, image, 0.7, True)
        if location != False:
            #print("Found template: ", char_mask.filepath," At time: ", format_timestamp(sec), " At location: ", location)
            #if location[0] < cv2.UMat.get(image).shape[::-1][0]/2: # if on left side of screen
            if '-left' in os.path.basename(char_mask.filepath):
                foundMatch.charLeft = os.path.basename(char_mask.filepath).split('-')[0]
                #print("Character left: ", foundMatch.charLeft)
            else:
                foundMatch.charRight = os.path.basename(char_mask.filepath).split('-')[0]
                #print("Character right: ", foundMatch.charRight)
    return foundMatch

def searchForPlayers(image, FoundMatch):
    foundMatch = FoundMatch
    for player_mask in PLAYER_MASKS:
        location = matchTemplate(player_mask.template, image, 0.8, True)
        if location != False:
            #print("Found template: ", player_mask.filepath," At time: ",  format_timestamp(sec), " At location: ", location)
            #if location[0] < cv2.UMat.get(image).shape[::-1][0]/2: # if on left side of screen
            if '-left' in os.path.basename(player_mask.filepath):
                foundMatch.playerOne = os.path.basename(player_mask.filepath).split('-')[0]
                #print("Player one: ", foundMatch.playerOne)
            else:
                foundMatch.playerTwo = os.path.basename(player_mask.filepath).split('-')[0]
                #print("Player two: ", foundMatch.playerTwo)
    return foundMatch

'''
FRAMES_PATH = os.path.join(os.path.dirname(__file__), "frames")
def process720P(FoundMatches):
    foundMatches = FoundMatches
    count = 0
    for videoFilename in os.listdir(FRAMES_PATH):
        videoFilePath = "{}/{}".format(FRAMES_PATH,videoFilename)

        clip = VideoFileClip(videoFilePath, audio=False)
        clip_frame = clip.get_frame(0)
        clip_frame_gray = cv2.cvtColor(clip_frame, cv2.COLOR_RGB2GRAY)
        print("Processing frame(720P):", count)
        foundMatches[count] = searchForChars(clip_frame_gray, foundMatches[count])
        foundMatches[count] = searchForPlayers(clip_frame_gray, foundMatches[count])
        print("Processed 720P match, {} ({}) vs {} ({}) at {}".format(foundMatches[count].charLeft, 
            foundMatches[count].playerOne, foundMatches[count].charRight, foundMatches[count].playerTwo, format_timestamp(foundMatches[count].timeStamp)
            ))
        #Clean up
        clip.reader.close()
        #clip.audio.reader.close_proc()
        count = count+1
        
    return foundMatches
'''
videoFileName = "video.webm" 
SKIP_SECS = 60 #20
SEEK_SECS = 1 #0.5
if __name__ == '__main__':
    # https://www.youtube.com/watch?v=q3DY8EPtFMY test clip

    #print('OpenCL supported: ', cv2.ocl.haveOpenCL())

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

    #clip = VideoFileClip(videoFileName, audio=False)
    foundMatches = []
    next_sec = 0
    # 2m 55s on 240p with roundstart search
    # 00:03:24 with practice mode detection
    # 00:03:24 with cropped frames, pointless optimization?
    # Processed the 240P video in  00:02:35 with openCL
    # Processed the 240P video in  00:02:28 with OpenCL
    # Processed the 240P video in  00:01:50 without OpenCL
    # Processed the 240P video in  00:01:30 with opencv video instead of moviepy.
    # around 2m on 240p without roundstart and pmode detect
    # Processing speed ATM:  clip 1H:1M = processing 1M:1S
    
    benchmarkTimeStart = time.time() #see how long we took to process the video file.
    lookingForRoundOne = False
    inTrainingMode = False
    foundMatch = Match(0, "unknown","unknown","unknown","unknown")

    cap = cv2.VideoCapture(videoFileName)
    while(cap.isOpened()):
        ret, clip_frame = cap.read() # Read next frame
        if not ret: break #EOF
        sec = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if sec < next_sec:
            continue

        if USE_OPENCL: clip_frame_gray = cv2.UMat(cv2.cvtColor(clip_frame, cv2.COLOR_BGR2GRAY))
        else: clip_frame_gray = cv2.cvtColor(clip_frame, cv2.COLOR_BGR2GRAY)

        #DEBUG
        #cv2.imshow('frame',clip_frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'): #wait 1ms
        #    break

        # Check if the player has no opponent.
        for training_mask in TRAINING_MODE_MASKS:
            if matchTemplate(training_mask.template, clip_frame_gray, 0.8):
                inTrainingMode = True
                break
        # Idle machine, skip ahead.
        if matchTemplate(PRESS_START_MASK.template, clip_frame_gray, 0.8):
            next_sec = sec + SKIP_SECS
            continue
        if inTrainingMode:
            inTrainingMode = False
            next_sec = sec + 8
            #print("Found training mode at ", sec)
            continue
        # Found round start
        if lookingForRoundOne and matchTemplate(ROUND_START.template, clip_frame_gray, 0.8):
            foundMatch.timeStamp = int(round(sec))
            print("Found match: {} ({}) vs {} ({}) at {}".format(foundMatch.charLeft, 
            foundMatch.playerOne, foundMatch.charRight, foundMatch.playerTwo, format_timestamp(foundMatch.timeStamp)
            ))
            foundMatches.append(foundMatch)

            lookingForRoundOne = False
            next_sec = sec + SKIP_SECS
            continue

        # Found VS screen
        if not lookingForRoundOne and matchTemplate(VS_MASK.template, clip_frame_gray, 0.7):
            # Reset the found match info
            foundMatch = Match(0, "unknown","unknown","unknown","unknown")

            #Extract the character and player info from VS screen.
            foundMatch = searchForChars(clip_frame_gray, foundMatch)
            foundMatch = searchForPlayers(clip_frame_gray, foundMatch)
            #cv2.imshow('frame',clip_frame)
            #cv2.waitKey(0) 
            # Found the info from VS screen, now find the round start.
            #print('Found VS info at: ', format_timestamp(sec))
            lookingForRoundOne = True
            next_sec = sec + 6 # about 6s to load into intros from VS 
        else:
            next_sec = sec + SEEK_SECS

    cap.release()
    benchmarkTimeEnd = time.time()
    benchmarkTimeElapsed = benchmarkTimeEnd - benchmarkTimeStart
    print("Processed the 240P video in ", time.strftime("%H:%M:%S", time.gmtime(benchmarkTimeElapsed)))
    
    #Download 720P frames
    #downloadFrames(foundMatches, args.youtube_url)
    #Process the 720P
    #benchmarkTimeStart = time.time()
    #foundMatches = process720P(foundMatches)
    #benchmarkTimeEnd = time.time()
    #benchmarkTimeElapsed = benchmarkTimeEnd - benchmarkTimeStart
    #print("Processed the 720P frames in ", time.strftime("%H:%M:%S", time.gmtime(benchmarkTimeElapsed)))
    
    print("Writing to html.")
    writeToHtml(foundMatches, args.youtube_url)