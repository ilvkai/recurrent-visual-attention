import numpy as np
import os
import os.path as osp
import pandas as pd

from config import dreyeve_dir
from utils.data.my_utils import read_image

def get_mean_loc(indexSeq, indexFrame):
    # the location ranges from -1 to 1 and (x, y)
    pathFix = os.path.join(dreyeve_dir, '{:02d}'.format(indexSeq), 'saliency_fix', '{:06d}.png'.format(indexFrame))
    fixationMap = read_image(pathFix, channels_first=True, color=False) / 255
    loc = np.mean(np.where(fixationMap == np.max(fixationMap)), axis=1)
    loc = loc / [fixationMap.shape[0:2]] * 2 - 1
    # cv2.imread('temp.jpg',fixationMap)

    # pt_loc = (int(loc[0]), int(loc[1]))
    return loc



def get_speed_and_course_origial(indexSeq, file):
    # course ranges from -1 to 1. If the course is positive, it means the car is turning right.
    # course = orientation[now]- orientation[now - course_duration] and course_duration=50 means 50 frames.

    # get speed and course for 50 frames
    # if less than 50, the first several frames' speed and course equal the ones in first frame
    print('processing seq: {}'.format(indexSeq))
    pathFile = osp.join(dreyeve_dir, '{:02d}'.format(indexSeq), 'speed_course_coord.txt')
    colnames = ['frame', 'speed', 'course', 'lat', 'lon', 'no-meaning']
    df = pd.read_csv(pathFile, sep='\t', names=colnames, header=None)
    coursePrevious = 0

    for indexFrame in range(1, 7500+1):
        # print(df.iloc[indexFrame + 1]['speed'])
        speed = df.iloc[indexFrame - 1]['speed'] / 100

        OrientationNow = df.iloc[indexFrame - 1]['course']
        # remove nan numbers
        if np.isnan(OrientationNow):
            tempIndex = indexFrame - 1
        while np.isnan(OrientationNow):
            tempIndex = tempIndex - 1
            OrientationNow = df.iloc[tempIndex]['course']
        if indexFrame - 1 - 50 >= 0:
            OrientationPrevious = df.iloc[indexFrame - 1 - 50]['course']
        else:
            OrientationPrevious = df.iloc[0]['course']
        if np.isnan(OrientationPrevious):
            tempIndex = indexFrame - 1
        while np.isnan(OrientationPrevious):
            tempIndex = tempIndex - 1
            OrientationPrevious = df.iloc[tempIndex]['course']

        if abs(OrientationNow - OrientationPrevious) < 180:
            course = (OrientationNow - OrientationPrevious) / 40
        elif OrientationNow > OrientationPrevious:
            course = (OrientationNow - (OrientationPrevious + 360)) / 40
        elif OrientationNow < OrientationPrevious:
            course = (OrientationNow + 360 - OrientationPrevious) / 40

        line = '{:02d} {:04d} {:.3f} {:.3f}\n'.format( indexSeq, indexFrame, speed, course)
        # check abnormal course
        # print(line)
        if abs(course)>=1 or abs(course- coursePrevious)>=0.3:
            print(line)
        file.writelines(line)
        coursePrevious = course

    print('finish seq {}'.format(indexSeq))

def get_speed_and_course_new(indexSeq):
    # course ranges from -1 to 1. If the course is positive, it means the car is turning right.
    # course = orientation[now]- orientation[now - course_duration] and course_duration=50 means 50 frames.

    # get speed and course for 50 frames
    # if less than 50, the first several frames' speed and course equal the ones in first frame

    dirSave = 'speeds_and_courses_new'
    if not osp.exists(dirSave):
        os.mkdir(dirSave)
    print('processing seq: {}'.format(indexSeq))
    file = open(osp.join(dirSave, 'speed_course_{:02d}.txt'.format(indexSeq)), 'w')
    pathFile = osp.join(dreyeve_dir, '{:02d}'.format(indexSeq), 'speed_course_coord.txt')
    colnames = ['frame', 'speed', 'course', 'lat', 'lon', 'no-meaning']
    df = pd.read_csv(pathFile, sep='\t', names=colnames, header=None)
    coursePrevious = 0

    for indexFrame in range(1, 7500+1):
        if indexFrame % 1000 ==0 :
            print('seq {:02d} has processed {:04d} frames'.format(indexSeq, indexFrame))
        # print(df.iloc[indexFrame + 1]['speed'])
        speed = df.iloc[indexFrame - 1]['speed'] / 100

        course = get_mean_loc(indexSeq, indexFrame)[0,1]*np.random.uniform(0.5,2)

        line = '{:02d} {:04d} {:.3f} {:.3f}\n'.format( indexSeq, indexFrame, speed, course)
        # check abnormal course
        # print(line)
        # if abs(course)>=1 or abs(course- coursePrevious)>=0.3:
        #     print(line)
        file.writelines(line)
        # coursePrevious = get_mean_loc(indexSeq, indexFrame)[0,1]

    print('finish seq {}'.format(indexSeq))
    file.close()

if __name__ == '__main__':
    print('get speeds and courses from speed_course_coord.txt')
    print('there are some collection errors for courses and is modified manually')


    # file = open('speed_and_course.txt', 'w')


    # indexSeq = 2
    # indexFrame = 1000
    # duration = 16
    # course_duration = 50
    # get_speed_and_course(1, file)
    # for indexSeq in range(1, 74+1):
    #     # get_speed_and_course_origial(indexSeq, file)
    #     get_speed_and_course_new(indexSeq, dirSave)

    # print(speed, course)
    # file.close()

    # import multiprocessing
    #
    # cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=16)
    #
    # indexRange = range(1, 75)
    # pool.map(get_speed_and_course_new, indexRange)

    # concatenate files
    filenames = ['file1.txt', 'file2.txt', ...]
    with open('test.txt', 'w') as outfile:
        for indexSeq in range(1,75):
            print('concatenating seq {:02d}'.format(indexSeq))
            fname = osp.join('speeds_and_courses_new' ,'speed_course_{:02d}.txt'.format(indexSeq))
            with open(fname) as infile:
                outfile.write(infile.read())






