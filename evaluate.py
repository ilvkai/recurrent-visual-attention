import numpy as np
import pandas as pd


pathLocation = '/home/lk/code/ram-dreye/output.txt'
colnames_location = ['seq', 'frame', 'loc_gt_h', 'loc_gt_w', 'loc_h', 'loc_w', 'scale_gt']
dfLocations = pd.read_csv(pathLocation, sep=' ', names=colnames_location, header=None)

dist_total = 0
count = 0
for index in range(len(dfLocations)):
    gt_h = dfLocations.iloc[index]['loc_gt_h']
    gt_w = dfLocations.iloc[index]['loc_gt_w']
    h = dfLocations.iloc[index]['loc_h']
    w = dfLocations.iloc[index]['loc_w']

    dist = ((gt_h-h)**2 + (gt_w -w)**2  )**0.5
    dist_total = dist_total + dist
    count = count +1

    if index % 1000 ==0:
        print('{}/{}'.format(index,len(dfLocations)))

    # print(dist)

print(dist_total/count)
