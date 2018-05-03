import os
import datetime
import argparse
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

MASKS_DIRPATH = os.path.join(
    os.path.dirname(__file__),
    'templates',
)

#TODO: Automated video rebuilding? if found >30min of specific player footage, get timestamps to matches (start - end)
# download the 720P clips and cut all back together with ffmpeg?
# automated YT upload of the footage?
# Make browser userscript that loads these info instead and automatically plays the video?
# Automated montages that way without having to reupload vids?
# https://developers.google.com/youtube/iframe_api_reference?csw=1
# ytplayer = document.getElementById("movie_player");
# ytplayer.getCurrentTime();

#USE_OPENCL = cv2.ocl.haveOpenCL()
USE_OPENCL = False

class Match(object):
    def __init__(self, timeStamp, timeStampEnd, charLeft, charRight, playerOne, playerTwo):
        self.timeStamp = timeStamp
        self.timeStampEnd = timeStampEnd
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
    Mask('characters/{}'.format(filepath))
    for filepath in os.listdir('{}/characters'.format(MASKS_DIRPATH))
]
VS_MASK = Mask('vs.png')

TRAINING_MODE_MASKS = [
    Mask('insert_coin_left.png'),
    Mask('insert_coin_right.png'),
    Mask('vs_ai_opponent.png')
]
PRESS_START_MASK = Mask('press_start.png')
ROUND_START = Mask('round_timer_99.png')
ROUND_END = Mask('round_end_chest.png')
#ROUND_END_DAREDEVIL = Mask('here_comes_daredevil.png')
FILE_OUTPUT = "matches.html"

def matchTemplate(template, image, threshold, return_location=False):
    #image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    #template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
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
    htmlFile.write('<iframe width="420" height="345" src="https://www.youtube.com/embed/{}"></iframe><br>\n'.format(youtubeVideoCode))
    for match in foundMatches:
        matchString = '<span class="matchInfo" data-startTime="{}" data-endTime="{}" data-char-left="{}" data-char-right="{}" data-player-one="{}" data-player-two="{}"></span>'.format(
            match.timeStamp, match.timeStampEnd, match.charLeft, match.charRight, match.playerOne, match.playerTwo
        )
        htmlFile.write('<a href={}#t={}>{}{}-{} {} ({}) vs {} ({})</a><br>\n'.format(
            youtubeUrl,
            match.timeStamp,
            matchString,
            format_timestamp(match.timeStamp),
            format_timestamp(match.timeStampEnd),
            match.charLeft,
            match.playerOne,
            match.charRight,
            match.playerTwo,
            ))
    htmlFile.close()

# Download the whole matches
def downloadMatches(foundMatches, url):
    #TODO: Delete previously downloaded matches
    count = 0
    for match in foundMatches:
        downloadUrl = subprocess.check_output('youtube-dl --format "best" -g {}'.format(url)).decode('utf-8').strip()
        subprocess.check_call('ffmpeg -ss {} -i "{}" -t {} -map 0:v -c:v libvpx matches/match_{}.webm'.format(
            str(datetime.timedelta(seconds=int(match.timeStamp))),
            downloadUrl,
            match.timeStampEnd - match.timeStamp,
            count
        ))
        count = count+1

# Download only second into the match round start for info parsing
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

CONCAT_LISTNAME = 'concat_list.txt'
CONCAT_VIDEONAME = 'concat_output.webm'
# Use ffmpeg to generate concetenated video.
def conCatVideo():
    # Delete previous concat list
    if os.path.exists(CONCAT_LISTNAME): os.remove(CONCAT_LISTNAME)
    # Generate new list
    file = open(CONCAT_LISTNAME, "w")

    count = 0
    for filename in os.listdir('matches'):
        file.write("file '{}'\n".format(os.path.abspath('{}/matches/match_{}.webm'.format(os.path.dirname(__file__), count))))
        count = count+1
    file.close()
    # ffmpeg -f concat -safe 0 -i list.txt -c copy output.webm
    if (os.path.exists(CONCAT_VIDEONAME)): os.remove(CONCAT_VIDEONAME)
    subprocess.check_call(
        'ffmpeg -f concat -safe 0 -i {} -c copy {}'.format(
            CONCAT_LISTNAME,
            CONCAT_VIDEONAME,
        ),
        shell=True,
        )

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
SEARCH_ENDOFMATCHES = True
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
    # Processed the 240P video in  00:01:30 with cropped image, seems to be no difference.
    # Processed the 240P video in  00:01:37 with match end search
    # around 2m on 240p without roundstart and pmode detect
    # Processing speed ATM:  clip 1H:1M = processing 1M:1S
    
    benchmarkTimeStart = time.time() #see how long we took to process the video file.
    lookingForRoundOne = False
    lookingForMatchEnd = False
    inTrainingMode = False
    foundMatch = Match(0, 0, "unknown", "unknown", "unknown", "unknown")

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

        # Search for end of the current match
        if lookingForMatchEnd and matchTemplate(ROUND_END.template, clip_frame_gray, 0.9):
            foundMatch.timeStampEnd = int(round(sec))
            timePassed = format_timestamp(foundMatch.timeStampEnd - foundMatch.timeStamp)
            print("Found match: {} ({}) vs {} ({}) at {} - {} len: {}".format(
                foundMatch.charLeft, 
                foundMatch.playerOne, 
                foundMatch.charRight, 
                foundMatch.playerTwo, 
                format_timestamp(foundMatch.timeStamp),
                format_timestamp(foundMatch.timeStampEnd),
                timePassed
            ))
            foundMatches.append(foundMatch)

            lookingForRoundOne = False
            lookingForMatchEnd = False
            next_sec = sec + 5
            continue
        # Found round start
        #cropped_frame_rstart_gray = clip_frame_gray[0:40,0:426]  # Crop from {x, y, w, h } 
        if lookingForRoundOne and matchTemplate(ROUND_START.template, clip_frame_gray, 0.7):
            foundMatch.timeStamp = int(round(sec))
            
            lookingForRoundOne = False
            if SEARCH_ENDOFMATCHES:
                lookingForMatchEnd = True
            else:
                SEARCH_ENDOFMATCHES: print("Found match: {} ({}) vs {} ({}) at {}".format(foundMatch.charLeft, 
                foundMatch.playerOne, foundMatch.charRight, foundMatch.playerTwo, format_timestamp(foundMatch.timeStamp)
                ))
                foundMatches.append(foundMatch)
            next_sec = sec + SKIP_SECS
            continue

        # Found VS screen
        if not lookingForRoundOne and matchTemplate(VS_MASK.template, clip_frame_gray, 0.7):
            # Reset the found match info
            foundMatch = Match(0, 0, "unknown", "unknown", "unknown", "unknown")
            #cropped_frame_char_gray = clip_frame_gray[0:40,0:426]  # Crop from {x, y, w, h } 
            #cropped_frame_player_gray = clip_frame_gray[150:240,0:426]  # Crop from {x, y, w, h } 
            #cv2.imshow('frame',cropped_frame_player_gray)
            #cv2.waitKey(0) 
            #Extract the character and player info from VS screen.
            foundMatch = searchForChars(clip_frame_gray, foundMatch)
            foundMatch = searchForPlayers(clip_frame_gray, foundMatch)
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