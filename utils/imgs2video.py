import os
import cv2
import time


# convert imgs to video

def picvideo(dirImgs, pathTargetVideo, w, h):
    fileslist = os.listdir(dirImgs)
    fps = 25

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 'I','4','2','0' means avi file
    video = cv2.VideoWriter(pathTargetVideo, fourcc, fps, (w,h))

    for item in fileslist:

        if item.endswith('.jpg'):
            item = dirImgs + '/' + item
            img = cv2.imread(item)
            video.write(img)
    video.release()


if __name__ == '__main__':
    picvideo(10)





