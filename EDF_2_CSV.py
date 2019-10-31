import numpy as np
import mne
import pandas as pd
from mne.io.edf.edf import _read_annotations_edf

##edf = mne.io.read_raw_edf('S001R03.edf')
##header = ','.join(['Time','C3','C4'])
##np.savetxt('S001R03.csv', (edf.get_data().T)[:,[0,8,12]], delimiter=',', header=header)

for sub_no in range(10,51):
    print(sub_no)
    for file_no in range(3,12,4):
        print(file_no)
        if file_no<10:
            edf_file = 'S0'+str(sub_no)+'R0'+str(file_no)+'.edf'
            csv_file = 'S0'+str(sub_no)+'R0'+str(file_no)+'.csv'
            dump_file_no = 'S'+str(sub_no)+'_'+str(file_no)+'.csv'
        else:
            edf_file = 'S0'+str(sub_no)+'R'+str(file_no)+'.edf'
            csv_file = 'S0'+str(sub_no)+'R'+str(file_no)+'.csv'
            dump_file_no = 'S'+str(sub_no)+'_'+str(file_no)+'.csv'
            
        onset, duration, description = _read_annotations_edf(edf_file)
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
                
        
        rd_csv_file = pd.read_csv(csv_file)
        load_file_time = rd_csv_file.iloc[1:,0].values
        load_file_channels = rd_csv_file.iloc[1:,[9,13]].values
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
        
        np.savetxt(dump_file_no, time_stamps1, delimiter=',')
