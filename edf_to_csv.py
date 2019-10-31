import numpy as np
import mne
import sys
import pandas as pd
from mne.io.edf.edf import _read_annotations_edf

##edf = mne.io.read_raw_edf('S001R03.edf')
##header = ','.join(['Time','C3','C4'])
##np.savetxt('S001R03.csv', (edf.get_data().T)[:,[0,8,12]], delimiter=',', header=header)
file1 = 'S004R07.edf'
file2 = 'S004R07.csv'
file3 = 'T004R07.csv'
onset, duration, description = _read_annotations_edf(file1)
onset = np.array(onset, dtype=float)
duration = np.array(duration, dtype=float)
description = np.array(description)
##annotations = mne.Annotations(onset=onset, duration=duration,description=description,orig_time=None)
##np.savetxt('S001R0.csv', duration, delimiter=',', header=header)

for i in range(len(description)):
    if description[i]=='T0':
        description[i]=0
    if description[i]=='T1':
        description[i]=1
    if description[i]=='T2':
        description[i]=2
        

csv_file = pd.read_csv(file2)
load_file_time = csv_file.iloc[1:,0].values
load_file_channels = csv_file.iloc[1:,[9,13]].values
time_stamps1 = np.empty((0,4))
j = 0
k = 0
for i in load_file_time:
    if onset[j]>120:
        onset[j]=round(onset[j]-120,1)
    if onset[j]>60:
        onset[j]=round(onset[j]-60,1)
    if float(i[3:7])==onset[j]:
        movement=description[j]
        j+=1
##        print(j)
    time_stamps = np.empty((0))
    time_stamps = np.append(time_stamps, float(i[3:7]))
    time_stamps = np.append(time_stamps, float(load_file_channels[k,0]))
    time_stamps = np.append(time_stamps, float(load_file_channels[k,1]))
    k += 1
    time_stamps = np.append(time_stamps, int(movement))
    if j>=30:
        break
    time_stamps1 = np.append(time_stamps1, time_stamps.reshape(-1,4), axis=0)
##    print(i[3:7]+'--'+load_file_channels[k]+'--'+movement)

np.savetxt(file3, time_stamps1, delimiter=',')
