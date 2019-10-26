import numpy as np
import os
import os.path as osp
from utils.data.my_utils import read_image

from config import dreyeve_dir

def get_mean_distance(indexSeq):
    print('processing seq : {:02d}'.format(indexSeq))
    dirSave = 'scale'
    if not osp.exists(dirSave):
        os.mkdir(dirSave)
    file = open(osp.join(dirSave, 'scale_{:02d}.txt'.format(indexSeq)), 'w')
    # the location ranges from -1 to 1 and (x, y)
    for indexFrame in range(1, 7500+1):
        if indexFrame % 1000==0:
            print('Seq {:02d} has processed {} images'.format(indexSeq, indexFrame))
        pathFix = os.path.join(dreyeve_dir, '{:02d}'.format(indexSeq), 'saliency_fix', '{:06d}.png'.format(indexFrame))
        fixationMap = read_image(pathFix, channels_first=True, color=False) / 255
        locs = np.where(fixationMap == np.max(fixationMap))
        loc_center = np.mean(np.where(fixationMap == np.max(fixationMap)), axis=1)
        distance = abs(np.transpose(np.asarray(locs), (1, 0)) - loc_center)
        scale_h = np.sort(distance[:,0])[int(np.floor(distance.shape[0]*0.8))]/fixationMap.shape[0]
        scale_w = np.sort(distance[:,1])[int(np.floor(distance.shape[0]*0.8))]/fixationMap.shape[1]

        # mean_distance = mean_distance / [fixationMap.shape[0:2]] * 2
        scale =  np.max((scale_w, scale_h))
        # print('mean_distance is {}'.format(mean_distance))

        line = '{:02d} {:04d} {:.3f}\n'.format(indexSeq, indexFrame, scale)
        file.writelines(line)



    file.close()

if __name__ == '__main__':
    print('get mean location scale (mean distance)')
    # get_mean_distance(1)
    # print('there are some collection errors for courses and is modified manually')

    import multiprocessing

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=32)

    indexRange = range(1, 75)
    pool.map(get_mean_distance, indexRange)


    with open('scales.txt', 'w') as outfile:
        for indexSeq in range(1, 75):
            print('concatenating seq {:02d}'.format(indexSeq))
            fname = osp.join('scale', 'scale_{:02d}.txt'.format(indexSeq))
            with open(fname) as infile:
                outfile.write(infile.read())

