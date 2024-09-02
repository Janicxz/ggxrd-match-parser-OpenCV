@echo off
SET /p URL="Enter youtube URL (e.g. https://www.youtube.com/watch?v=fOvG_TfnCVo): "
xrd-match-parser-opencv.py %URL%

pause