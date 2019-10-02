import cv2
import os.path as osp
import os
import argparse

import multiprocessing


logAll = []

def extract_frames_from_avi(pathFile, secquence, dirTarget):
    vidcap = cv2.VideoCapture(pathFile)
    success, image = vidcap.read()
    count = 1

    while success:
        cv2.imwrite(osp.join(dirTarget, '{:06d}.jpg'.format(count)), image)  # save frame as JPEG file
        success, image = vidcap.read()
        if count %1000 ==0:
            print('Seq: {}: {} frames are precessed'.format(secquence, count))
        count += 1

    log = 'frames of Seq {} is {}'.format(secquence, count-1)
    logAll.append(log)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument("--seq", type=int, default=74)
    parser.add_argument("--input_dir", type=str, default='/home/lk/data/DREYEVE_DATA/')
    parser.add_argument("--output_dir", type=str, default='/tmp/lk/data/DREYEVE_DATA/')
    args = parser.parse_args()

    def singleSeq(indexSeq):
        filePath = osp.join(args.input_dir, '{:02d}'.format(indexSeq), 'video_garmin.avi')
        dirTarget = osp.join(args.output_dir, '{:02d}'.format(indexSeq), 'frames')
        if not osp.exists(dirTarget):
            os.makedirs(dirTarget)
        extract_frames_from_avi(filePath, indexSeq, dirTarget)

    import multiprocessing

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=16)


    indexRange = range(1, 75)
    pool.map(singleSeq, indexRange)









