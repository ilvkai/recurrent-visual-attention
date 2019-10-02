import numpy as np
import cv2


def blend_map(img, map, factor=0.5, colormap=cv2.COLORMAP_JET):
    """
    Function to blend an image and a probability map.

    :param img: The image
    :param map: The map
    :param factor: factor * img + (1-factor) * map
    :param colormap: a cv2 colormap.
    :return: The blended image.
    """

    assert 0 < factor < 1, 'factor must satisfy 0 < factor < 1'

    map = np.float32(map)
    map /= map.max()
    map *= 255
    map = map.astype(np.uint8)

    blend = cv2.addWeighted(src1=img, alpha=factor,
                            src2=cv2.applyColorMap(map, colormap), beta=(1-factor),
                            gamma=0)

    return blend

def blend_map_with_focus_rectangle(img, map, loc, scale=0.3, factor=0.5, color= (255, 0, 0), colormap=cv2.COLORMAP_JET):
    """
        Function to blend an image and a probability map with rectangle.

        :param img: The image
        :param map: The map
        :param loc: x, y location range from -1 to 1 and (0,0) is the center
        :param scale: FoV scale range from 0 to 1
        :param factor: factor * img + (1-factor) * map
        :param colormap: a cv2 colormap.
        :return: The blended image.
        """

    assert 0 < factor < 1, 'factor must satisfy 0 < factor < 1'

    map = np.float32(map)
    map /= map.max()
    map *= 255
    map = map.astype(np.uint8)

    img = np.asarray(img, np.uint8)

    blend = cv2.addWeighted(src1=img, alpha=factor,
                            src2=cv2.applyColorMap(map, colormap), beta=(1 - factor),
                            gamma=0)

    # draw location and rectangle
    # notice that blend.shape is (h, w, c)
    centerLoc = 0.5* (loc+1.0)* blend.shape[0:2]
    shortSideLen = np.min([blend.shape[0],blend.shape[1]]) * scale
    pt1 = (int(np.max([0, centerLoc[1] - shortSideLen /2])), int(np.max([0, centerLoc[0] - shortSideLen /2])) )
    pt2 = (int(np.min([blend.shape[1]-1, centerLoc[1] + shortSideLen/2])), int(np.min([blend.shape[0]-1, centerLoc[0] + shortSideLen/2])))

    # color  = (255, 0, 0)
    thickness = 2

    blend = cv2.rectangle(blend, pt1, pt2, color, thickness)

    return blend

def blend_map_with_focus_circle(img, map, loc, scale=0.3, factor=0.5, color= (255, 0, 0), colormap=cv2.COLORMAP_JET):
    """
        Function to blend an image and a probability map with rectangle.

        :param img: The image
        :param map: The map
        :param loc: x, y location range from -1 to 1 and (0,0) is the center
        :param scale: FoV scale range from 0 to 1
        :param factor: factor * img + (1-factor) * map
        :param colormap: a cv2 colormap.
        :return: The blended image.
        """

    assert 0 < factor < 1, 'factor must satisfy 0 < factor < 1'

    map = np.float32(map)
    map /= map.max()
    map *= 255
    map = map.astype(np.uint8)

    img = np.asarray(img, np.uint8)

    blend = cv2.addWeighted(src1=img, alpha=factor,
                            src2=cv2.applyColorMap(map, colormap), beta=(1 - factor),
                            gamma=0)

    # draw location and rectangle
    # notice that blend.shape is (h, w, c)
    centerLoc = 0.5* (loc+1.0)* blend.shape[0:2]
    # shortSideLen = np.min([blend.shape[0],blend.shape[1]]) * scale
    # pt1 = (int(np.max([0, centerLoc[0] - shortSideLen /2])), int(np.max([0, centerLoc[1] - shortSideLen /2])) )
    # pt2 = (int(np.min([blend.shape[1]-1, centerLoc[0] + shortSideLen/2])), int(np.min([blend.shape[0]-1, centerLoc[1] + shortSideLen/2])))

    # color  = (255, 0, 0)
    thickness = 2

    # centerLoc = (30,200)
    centerLoc = (int(centerLoc[1]), int(centerLoc[0]))
    blend = cv2.circle(blend, centerLoc, 20, color, thickness)

    return blend

if __name__ == '__main__':
    img = cv2.imread('/home/lk/data/DREYEVE_DATA/01/frames/000001.jpg')
    map = cv2.imread('/home/lk/data/DREYEVE_DATA/01/saliency_fix/000001.png')
    loc = np.array([0,0])

    blend = blend_map_with_focus(img, map, loc)
    cv2.imwrite('test.jpg', blend)
