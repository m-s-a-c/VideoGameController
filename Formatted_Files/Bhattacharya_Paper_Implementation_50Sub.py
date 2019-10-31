import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
import math

listOfLeft = []
listOfRight = []
listOfRest = []
rmsVal = np.empty((0,3))
rmsValForLeft = np.empty((0,3))
rmsValForRight = np.empty((0,3))


for sub_no in range(1,51):
    print(sub_no)
    for file_no in range(3,12,4):
        csv_file_name = 'S'+str(sub_no)+'_'+str(file_no)+'.csv'
        csv_file = pd.read_csv(csv_file_name)
        rd_csv_file = csv_file.iloc[:,[1,2,3]].values

        rmsValForRest = np.empty((0,3))

        GoingMovement=0
        GoingCounter=0

        for i in range(len(rd_csv_file)):
            CurrentMovement=rd_csv_file[i,2]
            CurrentCounter=i
            if(CurrentMovement==GoingMovement and CurrentMovement==0):
                rmsValForRest=np.append(rmsValForRest,rd_csv_file[i,:].reshape(-1,3),axis=0)
                                        
            elif(CurrentMovement==GoingMovement and CurrentMovement==1):
                rmsValForLeft=np.append(rmsValForLeft,rd_csv_file[i,:].reshape(-1,3),axis=0)
                                        
            elif(CurrentMovement==GoingMovement and CurrentMovement==2):
                rmsValForRight=np.append(rmsValForRight,rd_csv_file[i,:].reshape(-1,3),axis=0)
            
            else:
                if(GoingMovement==0):
                    listOfRest.append(rmsValForRest)
                    rmsValForRest = np.empty((0,3))
                    GoingMovement=CurrentMovement
                    GoingCounter=CurrentCounter
                elif(GoingMovement==1):
                    listOfLeft.append(rmsValForLeft)
                    rmsValForLeft = np.empty((0,3))
                    GoingMovement=CurrentMovement
                    GoingCounter=CurrentCounter
                elif(GoingMovement==2):
                    listOfRight.append(rmsValForRight)
                    rmsValForRight = np.empty((0,3))
                    GoingMovement=CurrentMovement
                    GoingCounter=CurrentCounter

##plt.plot(A1[0][:,0],'r',label='Rest')
##plt.plot(A2[0][:,0],'g',label='Right')
##plt.plot(A3[0][:,0],'b',label='Left')
##plt.xlabel('Number of Samples --->>>')
##plt.ylabel('RootMeanSquare of the Signal --->>>')
##plt.legend()
##plt.grid()
##plt.show()

#----------------- Designing Butterworth Bandpass And Notch Filter ---------- #
from scipy.signal import butter, lfilter

b, c = butter(5, [0.5/80,30/80], btype='band')

#--------------------- Aplying Filters on Left Brain -------------------------#
y4 = np.empty((0,2))
for i in range(1050):
    listOfLeft[i][:,:2] = lfilter(b,c,listOfLeft[i][:,:2])
    
#--------------------- Aplying Filters on Right Brain -------------------------#
y4 = np.empty((0,2))
for i in range(1050):
    listOfRight[i][:,:2] = lfilter(b,c,listOfRight[i][:,:2])
    
#--------------------- Aplying Filters on Resting Brain -------------------------#
y4 = np.empty((0,2))
for i in range(2100):
    listOfRest[i][:,:2] = lfilter(b,c,listOfRest[i][:,:2])

##############################--------------DWT--------------##########################  
##############################-----------Left Hand-----------##########################
Level = 3
i = 1

mn = np.empty((0,88))

for k in range(1050):
    A1 = np.empty((0,Level+1))
    A1 = pywt.wavedec(listOfRight[k][:,:2], 'db4', level=Level,axis=0)    
##    print('Rest',A1[i][:,1].shape)
    mn = np.append(mn, (A1[i][:,1]-A1[i][:,0]).reshape(-1,88),axis=0)
    
yl = mn.reshape(-1,1050)
yl = np.append(mn, np.zeros((mn.shape[0],1)), axis=1)
yl[:,88] = 2

##############################-----------Right Hand-----------#########################
mn = np.empty((0,88))

for k in range(1050):
    A2 = np.empty((0,Level+1))
    A2 = pywt.wavedec(listOfLeft[k][:,:2], 'db4', level=Level,axis=0)
##    print('Left',(A2[i][:,1]-A2[i][:,0]).shape)
##    print('Left',A2[i][:,1].shape)
##    print('Left',A2[i][:,0].shape)
    mn = np.append(mn, (A2[i][:,1]-A2[i][:,0]).reshape(-1,88),axis=0)
      
yr = mn.reshape(-1,1050)
yr = np.append(mn, np.zeros((mn.shape[0],1)), axis=1)
yr[:,88] = 1

##############################-----------Resting State-----------#########################
mn = np.empty((0,88))

for k in range(1050):
    A3 = np.empty((0,Level+1))
    A3 = pywt.wavedec(listOfRest[k][:,:2], 'db4', level=Level,axis=0)
##    print('Rest',A3[i][:88,1].shape)
    mn = np.append(mn, (A3[i][:88,1]-A3[i][:88,0]).reshape(-1,88),axis=0)

ym = mn.reshape(-1,1050)
ym = np.append(mn, np.zeros((mn.shape[0],1)), axis=1)
ym[:,88] = 0

##############################--------------PSD--------------##########################
##############################-----------Left Hand-----------##########################
from scipy import signal

psdl = np.empty((0,33))

for k in range(1050):
    f, Pxx_den = signal.welch(listOfLeft[k][:,:2], fs=160,window='hamming', nperseg=64,axis=0)
##    print('Left',(A2[i][:,1]-A2[i][:,0]).shape)
##    print('Left',A2[i][:,1].shape)
##    print('Left',A2[i][:,0].shape)
    psdl = np.append(psdl, (Pxx_den[:,1]-Pxx_den[:,0]).reshape(-1,33),axis=0)

pl = psdl.reshape(-1,1050)
pl = np.append(psdl, np.zeros((psdl.shape[0],1)), axis=1)
pl[:,33] = 1

##############################-----------Right Hand-----------#########################
psdr = np.empty((0,33))

for k in range(1050):
    f, Pxx_den = signal.welch(listOfRight[k][:,:2], fs=160,window='hamming', nperseg=64,axis=0)
##    print('Left',(A2[i][:,1]-A2[i][:,0]).shape)
##    print('Left',A2[i][:,1].shape)
##    print('Left',A2[i][:,0].shape)
    psdr = np.append(psdr, (Pxx_den[:,1]-Pxx_den[:,0]).reshape(-1,33),axis=0)

pr = psdr.reshape(-1,1050)
pr = np.append(psdr, np.zeros((psdr.shape[0],1)), axis=1)
pr[:,33] = 2

##############################-----------Rest State-----------#########################
psdm = np.empty((0,33))

for k in range(1050):
    f, Pxx_den = signal.welch(listOfRest[k][:,:2], fs=160,window='hamming', nperseg=64,axis=0)
##    print('Left',(A2[i][:,1]-A2[i][:,0]).shape)
##    print('Left',A2[i][:,1].shape)
##    print('Left',A2[i][:,0].shape)
    psdm = np.append(psdm, (Pxx_den[:,1]-Pxx_den[:,0]).reshape(-1,33),axis=0)

pm = psdm.reshape(-1,1050)
pm = np.append(psdm, np.zeros((psdm.shape[0],1)), axis=1)
pm[:,33] = 0

##############################-----------Average Power-----------##########################
##############################-------------Left Hand-------------##########################
rmsl = np.empty((0,1))

for k in range(1050):
    rmsl = np.append(rmsl, (np.sqrt((1/len(listOfLeft[i][:,1])) * np.sum(np.power(listOfLeft[i][:,1],2),axis=0))-(np.sqrt((1/len(listOfLeft[i][:,0])) * np.sum(np.power(listOfLeft[i][:,0],2),axis=0))).reshape(-1,1)),axis=0)

##############################-------------Right Hand-------------##########################
rmsr = np.empty((0,1))

for k in range(1050):
    rmsr = np.append(rmsr, (np.sqrt((1/len(listOfRight[i][:,1])) * np.sum(np.power(listOfRight[i][:,1],2),axis=0))-(np.sqrt((1/len(listOfRight[i][:,0])) * np.sum(np.power(listOfRight[i][:,0],2),axis=0))).reshape(-1,1)),axis=0)

##############################-------------Rest State-------------##########################
rmsm = np.empty((0,1))

for k in range(1050):
    rmsm = np.append(rmsm, (np.sqrt((1/len(listOfRest[i][:,1])) * np.sum(np.power(listOfRest[i][:,1],2),axis=0))-(np.sqrt((1/len(listOfRest[i][:,0])) * np.sum(np.power(listOfRest[i][:,0],2),axis=0))).reshape(-1,1)),axis=0)

################################-----------Categorisation-----------#########################
leftMove = yl[:,:88]
leftMove = np.append(leftMove,rmsl,axis=1)
leftMove = np.append(leftMove,pl[:,:34],axis=1)
##leftMove = np.append(leftMove,np.zeros(420,1), axis=1)
##leftMove[:,-1] = 1

rightMove = yr[:,:88]
rightMove = np.append(rightMove,rmsr,axis=1)
rightMove = np.append(rightMove,pr[:,:34],axis=1)

restMove = ym[:,:88]
restMove = np.append(restMove,rmsm,axis=1)
restMove = np.append(restMove,pm[:,:34],axis=1)

allData = leftMove
allData = np.append(allData,rightMove,axis=0)
allData = np.append(allData,restMove,axis=0)


X = allData[:,:122]
y = allData[:, 122]

# Encoding categorical data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#pca = LDA(n_components = 1)
#X_train = pca.fit_transform(X_train, y_train)
#X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_

# Fitting Kernel SVM to the Training set
##from sklearn.svm import SVC
##classifier = SVC(kernel = 'rbf', random_state=0)
##classifier.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Predicting the Test set results
##y_pred = classifier.predict(X_test)

# Making the Confusion Matrix for Test Set
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("cm:= \n",cm)

#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X =X_train, y=y_train, cv=150)

# Making the Confusion Matrix for Training Set
y_predr = classifier.predict(X_train)
cmr = confusion_matrix(y_train, y_predr)
print("cmr:= \n",cmr)
