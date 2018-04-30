import os
import datetime
import argparse
import subprocess
import time

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

# TODO:
# Flag round start times at 240p/worst quality and download 720P frames from the specific times we found?
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
VS_MASK = Mask('vs.png') #240P
#VS_MASK = Mask('vs_360p.png') #360p

TRAINING_MODE_MASKS = [
    Mask('insert_coin_left.png'),
    Mask('insert_coin_right.png'),
    Mask('vs_ai_opponent.png')
]
PRESS_START_MASK = Mask('press_start.png')
#ROUND_ONE = Mask('duel_one.png')
#ROUND_START = Mask('lets_rock.png') 360P
#ROUND_TIMER_99 = Mask('lets_rock.png') #360P
ROUND_TIMER_99 = Mask('round_timer_99.png') #240P
FILE_OUTPUT = "matches.html"

def matchTemplate(template, image, treshold, return_location=False):
    # _NORMED returns correlation based value from 1.0 to 0.0 (1.0 being perfect match)
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= treshold:
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

#Download only the frames we're interested in at 720P
def downloadFrames(foundMatches, url):
    #TODO:
    # Delete previously downloaded frames
    count = 0
    for match in foundMatches:
        #downloadUrl = subprocess.check_output(['youtube-dl', '--format "bestvideo[height=720]"', '-g', url])
        downloadUrl = subprocess.check_output('youtube-dl --format "bestvideo[height=720]" -g {}'.format(url)).decode('utf-8').strip()
        subprocess.check_call('ffmpeg -ss {} -i "{}" -t 1 -map 0:v -c:v libvpx frames/frame_{}.webm'.format(
            str(datetime.timedelta(seconds=int(match.timeStamp))),
            downloadUrl,
            count
        ))
        count = count+1

videoFileName = "video.webm" #240P
#videoFileName = "test_360p.webm" #360P
SKIP_SECS = 60 #20
SEEK_SECS = 1 #0.5
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
    # 2m 55s on 240p with roundstart search
    # 00:03:24 with practice mode detection
    # 00:03:24 with cropped frames, pointless optimization?
    # 00:01:52 with optimal seek/skip times. Does not seem to be less accurate with adjusted values.
    # 00:02:46 with 0.5 SEEK and 60 skip
    # around 2m on 240p without roundstart and pmode detect
    benchmarkTimeStart = time.time() #see how long we took to process the video file.
    lookingForRoundOne = False
    inTrainingMode = False
    foundMatch = Match(0, "unknown","unknown","unknown","unknown")

    for sec, clip_frame in clip.iter_frames(with_times=True,dtype="uint8"):
        if sec < next_sec:
            continue

        # Convert to BRG for debug frame
        #clip_frame_bgr = cv2.cvtColor(clip_frame, cv2.COLOR_RGB2BGR)
        # Convert to grayscale for faster analysis
        clip_frame_gray = cv2.cvtColor(clip_frame, cv2.COLOR_RGB2GRAY)

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
            next_sec = sec + 8 # Skip a bit ahead out of training mode VS screen
            #print("Found training mode at ", sec)
            continue
        # Found round start
        if lookingForRoundOne and matchTemplate(ROUND_TIMER_99.template, clip_frame_gray, 0.8):
            # RANK DETECTION
            #rank_color_p1 = clip_frame[58,21]
            #rank_color_p2 = clip_frame[58,409]
            #p1_rank = ""
            #p2_rank = ""
            #if rank_color_p1[0] > 60 and rank_color_p1[1] < 50 and rank_color_p1[1] < 50:
            #    p1_rank = "red"
            #elif rank_color_p1[0] >= 150 and rank_color_p1[1] > 130 and rank_color_p1[1] < 80:
            #    p1_rank = "gold"
            #else:
            #    p1_rank = "blue"
            #if rank_color_p2[0] > 60 and rank_color_p2[1] < 50 and rank_color_p2[1] < 50:
            #    p2_rank = "red"
            #elif rank_color_p2[0] > 150 and rank_color_p2[1] > 130 and rank_color_p2[1] < 80:
            #    p2_rank = "gold"
            #else:
            #    p2_rank = "blue"
            #print("Found ranks, RGB p1: ", rank_color_p1, " p2: ", rank_color_p2, " Detect P1 ", p1_rank, " P2 ", p2_rank)

            foundMatch.timeStamp = sec
            print("Found match, {} ({}) vs {} ({}) at {}".format(foundMatch.charLeft, 
            foundMatch.playerOne, foundMatch.charRight, foundMatch.playerTwo, format_timestamp(foundMatch.timeStamp)
            ))
            foundMatches.append(foundMatch)

            lookingForRoundOne = False
            next_sec = sec + SKIP_SECS
            #show debug frame
            # cv2.imshow('Frame',clip_frame_bgr)
            #cv2.waitKey(0)
            continue

        # Found VS screen
        if not lookingForRoundOne and matchTemplate(VS_MASK.template, clip_frame_gray, 0.7):
            # Reset the found match info
            foundMatch = Match(0, "unknown","unknown","unknown","unknown")
            ###########
            #Search for all characters
            ###########
            for char_mask in CHARACTER_MASKS:
                #We already found both characters, no need to continue searching.
                if foundMatch.charLeft != "unknown" and foundMatch.charRight != "unknown":
                    #debug print("both characters found!")
                    break

                #for debug drawing
                #w, h = char_mask.template.shape[::-1] # invert W, H. shape returns them inverted.
                '''
                #Better way to find out side. only need one template.
                location = matchTemplate(char_mask.template, clip_frame_gray, 0.8, True)
                if location != False:
                    #print("Found template: ", char_mask.filepath," At time: ", format_timestamp(sec), " At location: ", location)
                    if location[0] < clip_frame_gray.shape[::-1][0]/2: #320: # if on left side of screen
                '''
                if matchTemplate(char_mask.template, clip_frame_gray, 0.7):
                    #debug print("Found template: ", char_mask.filepath," At time: ", sec)
                    if '-left' in os.path.basename(char_mask.filepath):
                        foundMatch.charLeft = os.path.basename(char_mask.filepath).split('-')[0]
                        #debug print("Character left: ", foundMatch.charLeft)
                    else:
                        foundMatch.charRight = os.path.basename(char_mask.filepath).split('-')[0]
                        #debug print("Character right: ", foundMatch.charRight)

                    # Draw a rectangle around the matched region.
                    #for pt in zip(*location[::-1]):
                    #    cv2.rectangle(clip_frame_bgr, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
            ##########
            #Search for all known player names
            ##########
            for player_mask in PLAYER_MASKS:
                #for drawing
                #w, h = player_mask.template.shape[::-1] # invert W, H. shape returns them inverted.
                if matchTemplate(player_mask.template, clip_frame_gray, 0.8):
                    #debug print("Found template: ", player_mask.filepath," At time: ", sec)
                    if '-left' in os.path.basename(player_mask.filepath):
                        foundMatch.playerOne = os.path.basename(player_mask.filepath).split('-')[0]
                        #debug print("Player one: ", foundMatch.playerOne)
                    else:
                        foundMatch.playerTwo = os.path.basename(player_mask.filepath).split('-')[0]
                        #debug print("Player two: ", foundMatch.playerTwo)

                    # Draw a rectangle around the matched region.
                    #for pt in zip(*location[::-1]):
                    #    cv2.rectangle(clip_frame_bgr, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
            # Found the info from VS screen, now find the round start.
            lookingForRoundOne = True
            next_sec = sec + 6 # about 6s to load into intros from VS 
        else:
            next_sec = sec + SEEK_SECS

    benchmarkTimeEnd = time.time()
    benchmarkTimeElapsed = benchmarkTimeEnd - benchmarkTimeStart
    print("Processed the video in ", time.strftime("%H:%M:%S", time.gmtime(benchmarkTimeElapsed)))
    print("Writing to html.")
    writeToHtml(foundMatches, args.youtube_url)
    print("Downloading 720P frames")
    downloadFrames(foundMatches, args.youtube_url)