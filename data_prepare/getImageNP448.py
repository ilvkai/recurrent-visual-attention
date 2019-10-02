# input is a sequence number and process a single sequence
# output is images

import cv2
import os.path as osp
import os
import numpy as np
import argparse

def process_single_img(image_dir, indexSequence, indexFrame, size_new, dirFrames_new_448, dirFrames_new_112 ):
    pathImg = os.path.join(image_dir, '{:02d}'.format(indexSequence), 'frames', '{:06d}.jpg'.format(indexFrame))

    img = cv2.imread(pathImg)
    if img is None:
        print('{}s for invalid img: path'.format(pathImg))
    img = img[:, :, [2, 1, 0]]
    img_h_w = cv2.resize(img, (size_new, size_new))
    imgResize = cv2.resize(img_h_w, (int(size_new / 4), int(size_new / 4)))
    img_h_w = img_h_w / 255.
    imgResize = imgResize / 255.

    singleImgResize = imgResize.transpose([2, 0, 1])
    singleImg_h_w = img_h_w.transpose([2, 0, 1])

    np.save(osp.join(dirFrames_new_112, '{:06d}'.format(indexFrame)),singleImgResize)
    np.save(osp.join(dirFrames_new_448, '{:06d}'.format(indexFrame)),singleImg_h_w)

    if indexFrame % 100 ==0 :
        print('frames {} of seq {} is processed'.format(indexFrame, indexSequence))





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=74)
    dreyeve_dir = '/home/lk/data/DREYEVE_DATA/'
    args = parser.parse_args()

    # dirFrames = osp.join(dreyeve_dir, '{:02d}'.format(args.seq), 'frames')
    def singleSeq(index):
        dirFrames_new_448 = osp.join(dreyeve_dir, '{:02d}'.format(index), 'frames-448')
        if not os.path.exists(dirFrames_new_448):
            os.makedirs(dirFrames_new_448)

        dirFrames_new_112 = osp.join(dreyeve_dir, '{:02d}'.format(index), 'frames-112')
        if not os.path.exists(dirFrames_new_112):
            os.makedirs(dirFrames_new_112)

        for indexFrame in range(1, 7501 + 1):
            process_single_img(dreyeve_dir, index, indexFrame, 448, dirFrames_new_448, dirFrames_new_112)

    import multiprocessing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    indexRange = range(1,40)
    pool.map(singleSeq, indexRange)









