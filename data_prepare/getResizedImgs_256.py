# input is a sequence number and process a single sequence
# output is resized images

import cv2
import os.path as osp
import os
import numpy as np
import argparse

def process_single_img(image_dir, indexSequence, indexFrame, size_new=256, dirFrames_new_256='' ):
    if dirFrames_new_256=='':
        assert dirFrames_new_256 == 0,"dirFrames_new_256 is None."
    pathImg = os.path.join(image_dir, '{:02d}'.format(indexSequence), 'frames', '{:06d}.jpg'.format(indexFrame))

    img = cv2.imread(pathImg)
    if img is None:
        print('{} for invalid img: path'.format(pathImg))
    img = img[:, :, [2, 1, 0]]
    img_h_w = cv2.resize(img, (size_new, size_new))

    cv2.imwrite(osp.join(dirFrames_new_256, '{:06d}.jpg'.format(indexFrame)),img_h_w)
    # after save the images, the imread do not need transpose([2, 0, 1]) any more

    if indexFrame % 1000 ==0 :
        print('frames {} of seq {} is processed'.format(indexFrame, indexSequence))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=74)
    dreyeve_dir = '/home/lk/data/DREYEVE_DATA/'
    target_dir = '/tmp/lk/data/DREYEVE_DATA/'
    args = parser.parse_args()

    # dirFrames = osp.join(dreyeve_dir, '{:02d}'.format(args.seq), 'frames')
    def singleSeq(index):
        dirFrames_new_256 = osp.join(target_dir, '{:02d}'.format(index), 'frames-256')
        if not os.path.exists(dirFrames_new_256):
            os.makedirs(dirFrames_new_256)

        dirFrames_new_112 = osp.join(target_dir, '{:02d}'.format(index), 'frames-112')
        if not os.path.exists(dirFrames_new_112):
            os.makedirs(dirFrames_new_112)

        for indexFrame in range(1, 7501 + 1):
            process_single_img(dreyeve_dir, index, indexFrame, 256, dirFrames_new_256)

    import multiprocessing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=16)

    indexRange = range(1,75)
    pool.map(singleSeq, indexRange)









