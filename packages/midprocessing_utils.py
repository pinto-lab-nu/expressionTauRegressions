import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
import scipy.io
from datetime import datetime

dpi=600


def decay_func_2(t, A_0, tau_0, A_1, tau_1, offset):
    return (A_0 * np.exp(-t/(tau_0))) + (A_1 * np.exp(-t/(tau_1))) + offset


def decay_func_1(t, A, tau, offset):
    return (A * np.exp(-t/(tau))) + offset


def LL(y_true, y_pred, sigma):
    n = y_true.shape[0]
    var = sigma**2
    RSS = np.sum((y_true - y_pred)**2)
    return (-n/2) * np.log(2 * np.pi * var) - (RSS/(2 * var))


def patchMaker(colors,labels,ax,title):
    all_patches = []
    for color,label in zip(colors,labels):
        color = matplotlib.colors.to_rgba((np.array(color)/255)+0.0)
        current_patch = mpatches.Patch(color=color,label=label)
        all_patches.append(current_patch)

    return ax.legend(handles=all_patches,bbox_to_anchor=(1.04,.5),loc='center left',title=title)


def NclassesToRGB(classProb,colors):
    for colorIDX,color in enumerate(colors):
        currentColorMod = np.array(color).reshape(3,1) / 255
        if colorIDX == 0:
            colorArray = currentColorMod
        else:
            colorArray = np.hstack([colorArray,currentColorMod])

    pointRGB = np.zeros((classProb.shape[0],3))
    for point in range(classProb.shape[0]):
        currentProb = classProb[point,:]
        pointRGB[point,:] = colorArray @ currentProb

    return pointRGB / np.max(pointRGB)


def OODpercentileReplace(signal,U_percent,L_percent):
    #meanPixDFF = np.mean(signal)
    #adjustedDFF = np.abs(signal-meanPixDFF)
    OOD_IDX = np.where((signal > np.percentile(signal,U_percent)) | (signal < np.percentile(signal,L_percent)))
    signal[OOD_IDX] = np.nan
    signal = np.array(pd.Series(signal).interpolate().tolist())
    return signal.reshape(-1)


def single_decay_func(t, A_0, tau_0, offset):
    return (A_0 * np.exp(-t/(tau_0))) + offset

def dual_decay_func(t, A_0, tau_0, A_1, tau_1, offset):
    return (A_0 * np.exp(-t/(tau_0))) + (A_1 * np.exp(-t/(tau_1))) + offset


# def multi_decay_func(t, A, tau, offset, N=2):
#     multi_decay_points = np.zeros(t.shape[0])
#     for expIDX in N:
#         multi_decay_pints += A[expIDX] * np.exp(-t/(tau[expIDX]*Fs))
#     return multi_decay_pints + offset


# def plotTauMap():
#     plt.figure()
#     plt.imshow(tau_matrix.reshape((128,128)))
#     plt.colorbar()
#     plt.title(currentSubject+'_'+str(currentDate)+'_'+str(currentSession)+' Tau (s) by Pixel\nBounds='+str(bounds))


def constructPassByClass(numClasses,noiseGMMpredictions,noiseFeaturesRaw,SNRbutterworthThresh):
    Class_SNR_butterworth = []
    GMM_current_class_pass = np.zeros(0,dtype=int)
    for i in range(numClasses):
        GMM_class_pass = np.array(np.where(noiseGMMpredictions[:,i] > 0.95))[0]
        Class_SNR_butterworth.append(np.median(noiseFeaturesRaw[GMM_class_pass,3]))
        if Class_SNR_butterworth[-1] > SNRbutterworthThresh:
            GMM_current_class_pass = np.hstack((GMM_current_class_pass,GMM_class_pass))
    return GMM_current_class_pass


def constructPassingPixels(noiseFeaturesRaw,SNRbutterworthThresh,pixelMaskIDX):
    passing = np.array(np.where(noiseFeaturesRaw[:,3] > SNRbutterworthThresh))[0]
    #passingPixSpace = pixelMaskIDX[passing]
    return passing#passingPixSpace


def CCFfromKey(key,VM,projectPath,verbose):
    q = (VM['widefield'].ReferenceIm & key)
    tform_allen2mouse = q.fetch('tform_allen2mouse')[0]
    ####print(tform_allen2mouse)
    mouse2allen = np.linalg.inv(tform_allen2mouse.T)
    image = q.fetch('image')
    allen_area_coord_array = np.array(q.fetch('allen_area_coord'))[0]
    MOUSEbregma = q.fetch('bregma')[0][0]
    MOUSElambda = q.fetch('lambda')[0][0]
    CCFbregma = (mouse2allen @ np.hstack((MOUSEbregma,np.array(1))).reshape(-1,1))[0:-1]
    CCFlambda = (mouse2allen @ np.hstack((MOUSElambda,np.array(1))).reshape(-1,1))[0:-1]
    #allen_area_labels = np.array(CCFtesting.fetch('allen_area_labels'))

    ####print(np.hstack((MOUSEbregma,np.array(1))).T.shape)
    
    # if allen_area_coord.shape[0] > 0:
    #     allen_area_coord_array = np.array(allen_area_coord)[0]
    # else:
    #     print(f"{key['subject_fullname']}_{key['session_date']}_{key['session_number']}: no coordinates")

    allTransformedCoord = np.empty((3,0))
    allCoord = np.empty((2,0))
    if verbose:
        skipVal = 64 #downsample plotted points by factor of skipVal, will get alliasing but this is just for viz
        fig, axes = plt.subplots(1,2,figsize=(10,5))
        #axes[0].imshow(np.rot90(image[0],1))
        for regionIDX in range(allen_area_coord_array.shape[0]):
            currentRegionArray = allen_area_coord_array[regionIDX].reshape(-1,2)
            plottingSubSamp = np.arange(0,currentRegionArray.shape[0],skipVal)
            if currentRegionArray.shape[0] > 0:
                axes[0].scatter(currentRegionArray[plottingSubSamp,0],currentRegionArray[plottingSubSamp,1],s=.05)
                axes[0].scatter([MOUSEbregma[1],MOUSElambda[1]],[MOUSEbregma[0],MOUSElambda[0]],color='black',s=5,marker='x')
                currentRegionArrayTransformed = mouse2allen @ np.hstack((currentRegionArray[:,[1,0]],np.ones((currentRegionArray.shape[0],1)))).T
                allCoord = np.hstack((currentRegionArray.T,allCoord))
                allTransformedCoord = np.hstack((currentRegionArrayTransformed[:,:],allTransformedCoord))
                axes[1].scatter(currentRegionArrayTransformed.T[plottingSubSamp,0],currentRegionArrayTransformed.T[plottingSubSamp,1],s=.05)
                axes[1].scatter([CCFbregma[0],CCFlambda[0]],[CCFbregma[1],CCFlambda[1]],color='black',s=5,marker='x')
        axes[0].set_title(f'Pre-Transform (mouse)')
        axes[1].set_title(f'Transformed (allen)')
        plt.suptitle(f"{key['subject_fullname']}_{key['session_date']}_{key['session_number']}")
        plt.savefig(os.path.join(projectPath,f"{key['subject_fullname']}_{key['session_date']}_{key['session_number']}_allenAffine.pdf"),dpi=100,bbox_inches="tight")
        plt.close()

    #subsamplePoints = np.arange(0,allTransformedCoord.shape[1],64)
    #plt.figure()
    #plt.scatter(allCoord[0,subsamplePoints],allCoord[1,subsamplePoints])
    #plt.imshow(image[0].T)
    
    ####print(np.hstack((currentRegionArray,np.ones((currentRegionArray.shape[0],1)))).T.shape)
    return MOUSEbregma, MOUSElambda, CCFbregma, CCFlambda, allTransformedCoord, allCoord, mouse2allen


def passingCensus(projectPath,task,initialLineFilterIDX,endLineFilterIDX,lineFilter,VM):
    passedCount = 0
    vascCount = 0
    mouse2allenList = []
    with open(os.path.join(projectPath,f'{task}_census.txt'), "w") as file:
            file.write(task)
    passingSessions = []
    to_preproc = pd.DataFrame(columns=['subject_fullname', 'session_date', 'session_number'])

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    axes[0].axis('equal')
    axes[1].axis('equal')

    fig1, axes1 = plt.subplots(1,2,figsize=(10,10))
    axes1[0].axis('equal')
    axes1[1].axis('equal')
    
    for lineFilterIDX in np.arange(initialLineFilterIDX,endLineFilterIDX,1):
        currentSubject = lineFilter['subject_fullname'][lineFilterIDX]
        currentDate = lineFilter['session_date'][lineFilterIDX]
        currentSession = lineFilter['session_number'][lineFilterIDX]

        key = {'subject_fullname': str(currentSubject)
            ,'session_date': str(currentDate)
            ,'session_number': currentSession
        }

        QCtesting = pd.DataFrame((VM['widefield'].WFQualityControl & key).fetch())

        if QCtesting.shape[0] > 0:
            passed = max(QCtesting['passed_all'])
            maxRev = max(QCtesting['revision'])
            if passed == 1:
                passedCount += 1
                passingSessions.append(np.array([currentSubject,currentDate,currentSession]))

                VascMask = pd.DataFrame((VM['widefield'].VascMask & key).fetch('mask_binned'))
                
                with open(os.path.join(projectPath,f'{task}_census.txt'), "a") as file:
                    file.write(f'\nsubject:{currentSubject}, date:{currentDate}, session:{currentSession}, maxRev:{maxRev}, passed:{passed}, VascMask:{VascMask[0].shape[0] == 1}')

                if VascMask[0].shape[0] == 1:
                    vascCount += 1
                    MOUSEbregma, MOUSElambda, CCFbregma, CCFlambda, allTransformedCoord, allCoord, mouse2allen = CCFfromKey(key,VM,projectPath,verbose=True)
                    axes[0].plot([MOUSEbregma[0],MOUSElambda[0]],[MOUSEbregma[1],MOUSElambda[1]])
                    axes[1].plot([CCFbregma[0],CCFlambda[0]],[CCFbregma[1],CCFlambda[1]])

                    mouse2allenList.append(mouse2allen)

                    subsamplePoints = np.arange(0,allTransformedCoord.shape[1],256)
                    axes1[0].scatter(allCoord[0,subsamplePoints],allCoord[1,subsamplePoints])
                    axes1[1].scatter(allTransformedCoord[0,subsamplePoints],allTransformedCoord[1,subsamplePoints])
                else:
                    new_session = pd.DataFrame({'subject_fullname': [str(currentSubject)], 'session_date': [str(currentDate)], 'session_number': [currentSession]})
                    to_preproc = pd.concat([to_preproc, new_session], ignore_index=True)
    
    current_date = datetime.now()
    formatted_date = current_date.strftime('%Y%m%d')
    #sessions_to_preproc = scipy.io.matlab.struct()
    #for column in to_preproc.columns:
    #    sessions_to_preproc[column] = to_preproc[column].values
    sessions_to_preproc = {col: to_preproc[col].values for col in to_preproc.columns}
    structured_array = [dict(zip(sessions_to_preproc, row)) for row in zip(*sessions_to_preproc.values())]
    #sessions_to_preproc = []
    #for _, row in to_preproc.iterrows():
    #    sessions_to_preproc.append({col: row[col] for col in to_preproc.columns})
    scipy.io.savemat(os.path.join(projectPath,f'sessions_to_preproc_{formatted_date}_LAAS.mat'), {'sessions_to_preproc': structured_array})

    with open(os.path.join(projectPath,f'{task}_census.txt'), "a") as file:
        file.write(f'\nFraction of Passing Sessions:{round(passedCount/endLineFilterIDX,3)}')
        file.write(f'\nFraction of Passing Sessions with VascMask:{round(vascCount/passedCount,3)}')
    
    return passingSessions, mouse2allenList


def butterworthFilterPixelPlots(projectPath,savePlots,numRows,fileNameSession,DFF_only,pixelMaskIDX,Fs,U_OODpercentileCutoff,L_OODpercentileCutoff):
    fig,axes = plt.subplots(numRows,6,figsize=(16,16))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.suptitle(fileNameSession+'\n3Hz Low- & High-Pass Butterworth')
    for subplotRow in range(numRows):
        linewidth = .5
        currentPixelIDX = np.random.randint(pixelMaskIDX.shape[0])
        currentPixel = pixelMaskIDX[currentPixelIDX]
        t = np.arange(0,DFF_only.shape[1]/Fs,1/Fs)

        ax = axes[subplotRow,0]
        ax.plot(t,DFF_only[currentPixel,:],linewidth=linewidth,color='black')
        ax.set_title('Pixel '+str(currentPixel))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('DFF')

        ax = axes[subplotRow,1]
        ax.psd(DFF_only[currentPixel,:],Fs=Fs,linewidth=linewidth,color='black')
        ax.set_ylabel("PSD")
        ax.set_title('Raw Signal')

        ax = axes[subplotRow,2]
        sos = signal.butter(5, 3, 'lp', fs=Fs, output='sos')
        lowFiltered = signal.sosfilt(sos, DFF_only[currentPixel,:])
        ax.plot(t,lowFiltered,linewidth=linewidth,color='green')
        ax.set_title('Low-Pass Pixel '+str(currentPixel))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('DFF')

        ax=axes[subplotRow,3]
        ax.psd(lowFiltered,Fs=Fs,linewidth=linewidth,color='green')
        ax.set_ylabel("PSD")
        ax.set_title('Low-Pass Butterworth')

        ax = axes[subplotRow,4]
        sos = signal.butter(5, 3, 'hp', fs=Fs, output='sos')
        highFiltered = signal.sosfilt(sos, DFF_only[currentPixel,:])
        highFiltered = OODpercentileReplace(highFiltered,U_OODpercentileCutoff,L_OODpercentileCutoff)
        ax.plot(t,highFiltered,linewidth=linewidth,color='red')
        ax.set_title('High-Pass Pixel '+str(currentPixel))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('DFF')

        ax=axes[subplotRow,5]
        ax.psd(highFiltered,Fs=Fs,linewidth=linewidth,color='red')
        ax.set_ylabel("PSD")
        ax.set_title('High-Pass Butterworth')

    if savePlots:
        plt.savefig(os.path.join(projectPath,'PixelButterworth_'+fileNameSession+'.pdf'),dpi=dpi,bbox_inches="tight")