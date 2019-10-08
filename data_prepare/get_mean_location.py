import numpy as np


def get_mean_loc(self, indexSeq, indexFrame):
    # the location ranges from -1 to 1 and (x, y)
    pathFix = os.path.join(self.root, '{:02d}'.format(indexSeq), 'saliency_fix', '{:06d}.png'.format(indexFrame))
    fixationMap = read_image(pathFix, channels_first=True, color=False) / 255
    loc = np.mean(np.where(fixationMap == np.max(fixationMap)), axis=1)
    loc = loc / [fixationMap.shape[0:2]] * 2 - 1
    # cv2.imread('temp.jpg',fixationMap)

    # pt_loc = (int(loc[0]), int(loc[1]))
    return loc