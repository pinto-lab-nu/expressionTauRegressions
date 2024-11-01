import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
import scipy.io
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit




dpi=600


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

    # fig1, axes1 = plt.subplots(1,2,figsize=(10,10))
    # axes1[0].axis('equal')
    # axes1[1].axis('equal')
    
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
                    # MOUSEbregma, MOUSElambda, CCFbregma, CCFlambda, allTransformedCoord, allCoord, mouse2allen = CCFfromKey(key,VM,projectPath,verbose=True)
                    # axes[0].plot([MOUSEbregma[0],MOUSElambda[0]],[MOUSEbregma[1],MOUSElambda[1]])
                    # axes[1].plot([CCFbregma[0],CCFlambda[0]],[CCFbregma[1],CCFlambda[1]])

                    # mouse2allenList.append(mouse2allen)

                    # subsamplePoints = np.arange(0,allTransformedCoord.shape[1],256)
                    # axes1[0].scatter(allCoord[0,subsamplePoints],allCoord[1,subsamplePoints])
                    # axes1[1].scatter(allTransformedCoord[0,subsamplePoints],allTransformedCoord[1,subsamplePoints])
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



def decay_func_dual(t, A_0, tau_0, A_1, tau_1, offset):
    return (A_0 * np.exp(-t/(tau_0))) + (A_1 * np.exp(-t/(tau_1))) + offset


def decay_func_mono(t, A, tau, offset):
    return (A * np.exp(-t/(tau))) + offset


def LL(y_true, y_pred, sigma):
    n = y_true.shape[0]
    var = sigma**2
    RSS = np.sum((y_true - y_pred)**2)
    return (-n/2) * np.log(2 * np.pi * var) - (RSS/(2 * var))


def fullSignalTau(signal_array, autocorrelation_in:bool, Fs:float, correlation_window:int=30, maxfev:int=16000, p0_dual=[1,0.04,1,4,0], p0_mono=[1,1,0], bounds_dual=([-10,0,-10,0.05,-5],[10,0.05,10,30,5]), bounds_mono=([-10,0,-5],[10,30,5])):
    '''
    Calculate the characteristic decay constant (tau) from the full signal, using mono- and bi-exponential fits.

    Parameters
    ----------
    signal_array : np.ndarray
        Array representing the signal data to be analyzed.
        
    autocorrelation_in : bool
        Flag indicating whether the signal handed in is already an autocorrelation.
        
    Fs : float
        Sampling frequency, in Hz.
        
    correlation_window : int
        Window length (in timesteps) used for computing the autocorrelation, if applicable.
        
    maxfev : int, optional
        Maximum number of itterations for curve fitting (default is 16000).
        
    p0_dual : list, optional
        Initial parameter estimates for the dual-exponential fit. Default is [scalar_0:1, tau_0:0.04, scalar_1:1, tau_1:4, offset:0] -> (param:seconds).
        
    p0_mono : list, optional
        Initial parameter estimates for the mono-exponential fit. Default is [scalar:1, tau:1, offset:0] -> (param:seconds).
        
    bounds_dual : tuple, optional
        Bounds on parameters for the dual-exponential fit, default is...
        ([scalar_0:-10, tau_0:0, scalar_1:-10, tau_1:0.05, offset:-5], [scalar_0:10, tau_0:0.05, scalar_1:10, tau_1:30, offset:5]) -> ([lower_bound:seconds], [upper_bound:seconds]).
        
    bounds_mono : tuple, optional
        Bounds on parameters for the mono-exponential fit, default is... 
        ([scalar:-10, tau:0, offset:-5], [scalar:10, tau:30, ofset:5]) -> ([lower_bounds:seconds], [upper_bounds:seconds])

    Returns
    -------
    tau_DF : DataFrame
        Estimated decay constant (tau) derived from the signal, as well as goodness-of-fit metrics for both mono- and dual-exponential fits.

        bestTau      : best tau fit, for easy access
        akaike_w     : relative probability of the dual timescale model being the best fit
        tau_fit_0    : dual tau model, fit lower tau
        tau_fit_1    : dual tau model, fit upper tau
        R2_Fit_Dual  : R squared of dual tau fit
        dualRSS      : dual model residual sum of squares
        dualSigmaHat : dual model predicted sigma, passed to likelihood function
        dualLL       : dual model log likelihood
        dualAICc     :
        dualBIC      :
        tau_fit_mono :
        R2_Fit_Mono  :
        monoRSS      :
        monoSigmaHat :
        monoLL       :
        monoAICc     :
        monoBIC      :

    '''
    
    AICC = lambda n, LL, k: -(2*LL) + (2*k) + ((2*k*(k+1)) / (n - k - 1)) # Since n is small we want to correct AIC->AICc, this is what the last term does,
                                                                     # should converge on true AIC if the length of our timeseries becomes arbitrarily long,
                                                                     # k is the number of parameters, *including* intercept
    BIC = lambda n, LL, k: -(2*LL) + (k*np.log(n))

    n_rows, T = signal_array.shape[0], signal_array.shape[1]

    if autocorrelation_in:
        correlation_window = T

    column_names = ['bestTau','akaike_w','tau_fit_0','tau_fit_1','R2_Fit_Dual','dualRSS','dualSigmaHat','dualLL','dualAICc','dualBIC','tau_fit_mono','R2_Fit_Mono','monoRSS','monoSigmaHat','monoLL','monoAICc','monoBIC']
    
    corr_t = np.arange(0, correlation_window, 1/Fs)
    tau_DF = pd.DataFrame(np.zeros((n_rows,len(column_names))))
    
    for row in range(0, n_rows, 1):
        row_signal = signal_array[row,:].reshape(-1)

        if not(sum(row_signal) == 0.0):
            if not autocorrelation_in:
                row_signal_a = (row_signal - np.mean(row_signal)) / (np.std(row_signal) * len(row_signal))
                row_signal_v = (row_signal - np.mean(row_signal)) /  np.std(row_signal)
                acf = np.correlate(row_signal_a, row_signal_v, 'full')
                acf_split = acf[row_signal.shape[0]-1:row_signal.shape[0]-1+(correlation_window*Fs)]
            else:
                acf_split = row_signal

            popt, pcov = curve_fit(decay_func_dual, corr_t, acf_split, p0_dual, bounds=bounds_dual, maxfev=maxfev)
            np.linalg.cond(pcov)
            A_fit_0, tau_fit_0, A_fit_1, tau_fit_1, offset_fit = popt

            popt, pcov = curve_fit(decay_func_mono, corr_t, acf_split, p0_mono, bounds=bounds_mono, maxfev=maxfev)
            np.linalg.cond(pcov)
            A_fit_mono, tau_fit_mono, offset_fit_mono = popt
            #tauMatrix[simNum,2,SNR_IDX] = tau_fit_mono

            decayFitPointsDual = decay_func_dual(corr_t,A_fit_0,tau_fit_0,A_fit_1,tau_fit_1,offset_fit)
            R2_Fit_Dual = r2_score(acf_split, decayFitPointsDual)
            dualRSS = np.sum((acf_split-decayFitPointsDual)**2)
            dualSigmaHat = np.sqrt(dualRSS/acf_split.shape[0]) #np.std(acf_split-decayFitPointsMono)
            dualLL = LL(acf_split, decayFitPointsDual, dualSigmaHat)
            dualAICc = AICC(acf_split.shape[0], dualLL, 5)
            dualBIC = BIC(acf_split.shape[0], dualLL, 5)

            decayFitPointsMono = decay_func_mono(corr_t,A_fit_mono,tau_fit_mono,offset_fit_mono)
            R2_Fit_Mono = r2_score(acf_split, decayFitPointsMono)
            monoRSS = np.sum((acf_split-decayFitPointsMono)**2)
            monoSigmaHat = np.sqrt(monoRSS/acf_split.shape[0]) #np.std(acf_split-decayFitPointsMono)
            monoLL = LL(acf_split, decayFitPointsMono, monoSigmaHat)
            monoAICc = AICC(acf_split.shape[0], monoLL, 3)
            monoBIC = BIC(acf_split.shape[0], monoLL, 3)

            deltaAICc = dualAICc - monoAICc
            akaike_w = np.exp(-0.5 * deltaAICc) / (1 + np.exp(-0.5 * deltaAICc))

            if dualBIC < monoBIC:
                bestTau = tau_fit_1
            else:
                bestTau = tau_fit_mono
            
            tau_DF.iloc[row,:] = np.array([bestTau,akaike_w,tau_fit_0,tau_fit_1,R2_Fit_Dual,dualRSS,dualSigmaHat,dualLL,dualAICc,dualBIC,tau_fit_mono,R2_Fit_Mono,monoRSS,monoSigmaHat,monoLL,monoAICc,monoBIC])

    tau_DF.columns = column_names

    return tau_DF



def behaviorGLM(signal_array, behavioral_signal, behavioral_lag:int, alphas, n_splits:int, random_state:int=1):
    ''' 
    Calculate residual signals independently for each row of a signal array, using a behavioral metric (behavioral_signal) at various lags
    as a predictor.

    Parameters
    ----------
    signal_array :
        Array of shape (observations x time) where rows should have been filtered (vasc mask, SNR mask) prior to being handed in.
    
    behavioral_signal :
        Array of shape (1 x time) where the time dimension should agree with that of the signal_array time dimension.

    behavioral_lag : int
        Lag value (in timesteps) used as the maximum positive and negative lag of the behavioral signal as a predictor for signal_array.

    alphas : np.ndarray
        Array of alpha values for ridge regression.

    n_splits : int
        Number of splits for K-fold cross validation.

    random_state : int, optional
        Random state seed for the K-fold split.

    Returns
    -------
    glm_residuals_array : np.ndarray
        Array of residual signals from the original signal_array that are not explained by the behavioral_signal (as defined by the ridge regression)
    '''

    n_rows = signal_array.shape[0]
    T = signal_array.shape[1]

    glm_residuals_array = np.zeros((n_rows, T-(2*behavioral_lag)))

    for row in np.arange(0, n_rows,1):

        row_signal = signal_array[row,:]
        row_signal = np.array(np.reshape(row_signal,-1))

        lags_vec = np.arange(-behavioral_lag, behavioral_lag+1, 1)
        behavior_signal_lag = np.zeros((len(behavioral_signal), lags_vec.shape[0]))
        for i,lag in enumerate(lags_vec):
            behavior_signal_lag[:, i] = np.roll(behavioral_signal, lag)
        behavior_signal_lag = behavior_signal_lag[behavioral_lag:behavior_signal_lag.shape[0]-behavioral_lag, :]
        
        row_signal = row_signal[behavioral_lag:T-behavioral_lag]

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        ########################################################
        ### Construct a GLM for determining signal residuals ###
        glm_predicted_signal = []
        glm_predicted_IDX = []
        for train_index, test_index in kfold.split(row_signal):
            training_behavior = behavior_signal_lag[train_index]
            testing_behavior = behavior_signal_lag[test_index]
            training_signal = row_signal[train_index]
            testing_signal = row_signal[test_index]

            # GLM (Basic, Identity Linker) with L2 Regularization
            ridge_weight = []
            alpha_r2 = []
            for alpha in alphas:
                ridge = Ridge(alpha=alpha)
                ridge.fit(training_behavior, training_signal)
                pred_DFF = ridge.predict(testing_behavior)
                r2_glm_l2 = r2_score(testing_signal,pred_DFF)

                ridge_weight.append(ridge.coef_)
                alpha_r2.append(r2_glm_l2)
        
            best_alpha = alphas[np.where(np.array(alpha_r2) == np.max(np.array(alpha_r2)))[0][0]]
            #GLM_R2 = np.max(np.array(alpha_R2)) # R2 of the alpha being selected for the model

            # Now predict test fold using the best alpha
            ridge = Ridge(alpha=best_alpha) # Choose the best alpha
            ridge.fit(training_behavior, training_signal)
            glm_predicted_signal.append(ridge.predict(testing_behavior))
            glm_predicted_IDX.append(test_index)

        glm_predicted_signal_allFolds = np.concatenate(glm_predicted_signal)
        glm_predicted_IDX_allFolds = np.concatenate(glm_predicted_IDX)

        sorted_indices = np.argsort(glm_predicted_IDX_allFolds)
        glm_predicted_signal_sorted = glm_predicted_signal_allFolds[sorted_indices]

        glm_residuals_array[row,:] = np.transpose(row_signal - glm_predicted_signal_sorted)
    
    return glm_residuals_array



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