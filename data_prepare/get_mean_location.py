import numpy as np
import os
import os.path as osp
from utils.data.my_utils import read_image

from config import dreyeve_dir

def get_mean_loc(indexSeq):
    print('processing seq : {:02d}'.format(indexSeq))
    dirSave = 'locations'
    if not osp.exists(dirSave):
        os.mkdir(dirSave)
    file = open(osp.join(dirSave, 'loc_{:02d}.txt'.format(indexSeq)), 'w')
    # the location ranges from -1 to 1 and (x, y)
    for indexFrame in range(1, 7500+1):
        if indexFrame % 1000==0:
            print('Seq {:02d} has processed {} images'.format(indexSeq, indexFrame))
        pathFix = os.path.join(dreyeve_dir, '{:02d}'.format(indexSeq), 'saliency_fix', '{:06d}.png'.format(indexFrame))
        fixationMap = read_image(pathFix, channels_first=True, color=False) / 255
        loc = np.mean(np.where(fixationMap == np.max(fixationMap)), axis=1)
        loc = loc / [fixationMap.shape[0:2]] * 2 - 1

        line = '{:02d} {:04d} {:.3f} {:.3f}\n'.format(indexSeq, indexFrame, loc[0,0], loc[0,1])
        file.writelines(line)


    file.close()

if __name__ == '__main__':
    print('get speeds and courses from location.txt')
    # print('there are some collection errors for courses and is modified manually')

    # import multiprocessing
    #
    # cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=32)
    #
    # indexRange = range(1, 75)
    # pool.map(get_mean_loc, indexRange)


    with open('locations.txt', 'w') as outfile:
        for indexSeq in range(1, 75):
            print('concatenating seq {:02d}'.format(indexSeq))
            fname = osp.join('locations', 'loc_{:02d}.txt'.format(indexSeq))
            with open(fname) as infile:
                outfile.write(infile.read())

