# input is a sequence number and process a single sequence
# output is images

import cv2
import os.path as osp
import os
import argparse


def extract_frames_from_avi(pathFile,  dirTarget):
    vidcap = cv2.VideoCapture(pathFile)
    success, image = vidcap.read()
    count = 1

    while success:
        cv2.imwrite(osp.join(dirTarget, '{:06d}.jpg'.format(count)), image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: {}, number is {}'.format( success, count))
        count += 1



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=74)
    parser.add_argument("--input_dir", type=str, default='/home/lk/data/DREYEVE_DATA/')
    parser.add_argument("--output_dir", type=str, default='/home/lk/data/DREYEVE_DATA/')
    args = parser.parse_args()


    filePath = osp.join(args.output_dir, '{:02d}'.format(args.seq), 'video_garmin.avi')
    dirTarget = osp.join(args.output_dir, '{:02d}'.format(args.seq), 'frames')
    if not osp.exists(dirTarget):
        os.makedirs(dirTarget)
    extract_frames_from_avi(filePath, dirTarget)




