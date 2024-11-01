import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import math
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.signal as signal
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.mixture import GaussianMixture
import matplotlib.patches as mpatches
import pickle
import platform
import datajoint
import packages.connect_to_dj as connect_to_dj
from multiprocessing import Pool
from scipy.linalg import lstsq
from scipy.stats import gamma
from packages.midprocessing_utils import *
import itertools


# Connects to database and creates virtual modules
VM = connect_to_dj.get_virtual_modules()

AIC = lambda n, LL, k: -(2*LL) + (2*k) + ((2*k*(k+1)) / (n - k - 1)) # this is actually not AIC but AICc, since n is small we want to correct, this is what the last term does,
                                                                     # should converge on true AIC if the length of our timeseries becomes arbitrarily long,
                                                                     # k is the number of parameters, *including* intercept
BIC = lambda n, LL, k: -(2*LL) + (k*np.log(n))


##########
### Params
#SNRthresh = 200 #threshold for SNR mask
lineSelection  = ['Rpb4-Ai96','Cux2-Ai96','C57BL6/J','PV-Ai96'][1]
task = 'IntoTheVoid' #'DelayedMatchToEvidence'
SNRbutterworthThresh = 6
printTauMap = False
printTauSubset = False
loadDFF = False
calcTau = False
justPlot = True #mod for calcTau, remove later or patch
processTau = True
summarizeLineTau = True
savePlots = True
Verbose = True

Fs = float(10)
dffSnipSecStart = 15 #number of seconds to remove from the start of the DFF (to remove session start artifacts)
dffSnipSecEnd = 10 #number of seconds to remove from the end of the DFF
runGLM = False
#firstPixel = 0 #14000 #start processing from this pixel, iterate through to end, for GLM
overwriteResiduals = False #when checking for an existing residuals file, ignore and start recalculating from the start
U_OODpercentileCutoff = 99.5
L_OODpercentileCutoff = 100 - U_OODpercentileCutoff
butterOrder = 10 #order of butterworth filter
butterThresh = 3 #in Hz

binnedPixels = 128 #binned image size, just get this from a table
rawImageSize = 1024 #OG image size, again, get this from a table -> reduce user-defined params ***more generally do this for all such relevant params!!***
binningFactor = rawImageSize // binnedPixels
numPixels = binnedPixels**2
version = 4

resolution = 25 #in um, CCF voxel resolution

maxfev = 16000
p0_dual = [1, 0.04, 1, 4, 0]  # Initial parameter guesses
p0_mono = [1, 1, 0]
longestToleratedTau = 30 # in seconds
bounds_2 = ([-10,0,-10,0.05,-5],[10,0.05,10,longestToleratedTau,5])
bounds_1 = ([-10,0,-5],[10,longestToleratedTau,5])
correlationWindow = int(30)


areaColors = ['#ff0000','#ff704d',                      #MO, reds
            '#4dd2ff','#0066ff','#003cb3','#00ffff',    #VIS, blues
            '#33cc33',                                  #SSp, greens
            '#a366ff']                                  #RSP, purples
structListMerge = np.array(['MOp','MOs','VISa','VISp','VISam','VISpm','SS','RSP'])



my_os = platform.system()
if my_os == 'Linux':
    uIn = True
if my_os == 'Windows':
    uIn = False

if uIn:
    lineSelection  = ['Rpb4-Ai96','Cux2-Ai96','C57BL6/J','PV-Ai96'][int(sys.argv[2])]

lineFilter = pd.DataFrame((VM['subject'].Subject * VM['session'].Session * VM['behavior'].BehavioralSession & 'experiment_type="widefield"' & 'task="'+task+'"' & 'line="'+lineSelection+'"').fetch())
if lineSelection == 'C57BL6/J':
    lineFilter = pd.DataFrame((VM['subject'].Subject * VM['session'].Session * VM['behavior'].BehavioralSession & 'experiment_type="widefield"' & 'line="'+lineSelection+'"').fetch())
    lineSelection = 'C57BL6J'

projectFolder = "lineFilter" + lineSelection

if my_os == 'Linux':
    projectPath = os.path.join(r'/mnt/fsmresfiles/Tau_Processing/',projectFolder+'/')
if my_os == 'Windows':
    projectPath = os.path.join(r'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing',projectFolder)



initialLineFilterIDX, endLineFilterIDX = 0, lineFilter.shape[0]

passingSessions, mouse2allenList = passingCensus(projectPath,task,initialLineFilterIDX,endLineFilterIDX,lineFilter,VM)

if uIn:
    initialPassingIDX, endPassingIDX = int(sys.argv[1]), int(sys.argv[1])+1
else:
    initialPassingIDX, endPassingIDX = 0, len(passingSessions)

crossMouseRegionTotal = np.zeros(structListMerge.shape[0])
mean_gamma_params = np.zeros((structListMerge.shape[0],3))
cross_mouse_region_tau_pop = [np.empty((0,1)) for _ in range(structListMerge.shape[0])]

### define a post-binned coordinate system for the widefield image in the original image space ###
binnedCoord_1D = np.arange(binningFactor//2,binnedPixels*binningFactor,binningFactor)
grid = np.asarray(list(itertools.product(binnedCoord_1D, binnedCoord_1D)))
grid = np.hstack((grid,np.ones((numPixels,1)))) #adds support for homogeneity in coordinates for 2D affine transform (homogenous 3rd dim)

Tallen_bregma_list =   [[[498.09172075],[367.0382196]],  #These values are coming from five Cux2-Cre mice with the reverse transform applied to bregma coordinates
                        [[503.05808372],[378.60458417]], #We can hard-code the tform_Tallen2CCF affine matrix with these values, since both these coordinate spaces should be invarient to the mouse
                        [[508.08991634],[359.81332211]], #(technically there's some varience in the Tallen-coordinate representation of pixel space due to imperfect registration of bregma-lambda coordinates in pixelspace)
                        [[505.91084397],[337.26708482]],
                        [[508.16826408],[334.7284952]]]
Tallen_lambda_list =   [[[493.93767265],[998.28975898]],
                        [[495.24134627],[954.04512525]],
                        [[508.31562383],[974.8140937]],
                        [[501.77957046],[1007.445369]],
                        [[505.94373286],[1002.75173691]]]

bregma_lambda_distance_avg = 4.1 #in mm, approximation from github.com/petersaj/neuropixels_trajectory_explorer
bregma_lambda_dist_CCF = bregma_lambda_distance_avg * 1000 / resolution
CCF_bregma = [228, 216] #25um bregma in CCF: (216, 18, 228) -> (AP, DV, ML)
CCF_lambda = [228, 380] #based on above, CCF lambda: (380, 18, 228)
Tallen_bregma_avg = np.mean(Tallen_bregma_list,axis=0)
Tallen_lambda_avg = np.mean(Tallen_lambda_list,axis=0)
trueCenterML = np.mean([Tallen_bregma_avg[0],Tallen_lambda_avg[0]])
Tallen_to_CCF_scale = (CCF_bregma[1] - CCF_lambda[1]) / (Tallen_bregma_avg[1] - Tallen_lambda_avg[1])
Tallen_to_CCF_AP_offset = CCF_bregma[1] - (Tallen_to_CCF_scale * Tallen_bregma_avg[1])
Tallen_to_CCF_ML_offset = CCF_bregma[0] - (Tallen_to_CCF_scale * Tallen_bregma_avg[0])
tform_Tallen2CCF = np.asarray([[Tallen_to_CCF_scale[0],0,Tallen_to_CCF_ML_offset[0]],
                               [0,Tallen_to_CCF_scale[0],Tallen_to_CCF_AP_offset[0]],
                               [0,0,1]])
allTauCCF_Coords = np.empty((3,0))

fig, axCCF = plt.subplots(1,2,figsize=(16,8))

for currentPassingSession in passingSessions[initialPassingIDX:endPassingIDX]:
    preprocessingOutputs = {}
    currentSubject,currentDate,currentSession = currentPassingSession[0],currentPassingSession[1],currentPassingSession[2]
    fileNameSession = str(currentSubject)+'_'+str(currentDate)+'_'+str(currentSession)

    key = {'subject_fullname': str(currentSubject)
        ,'session_date': str(currentDate)
        ,'session_number': currentSession
    }

    void_session_PixelAreaLabel = pd.DataFrame((VM['behavior'].BehavioralSession * VM['widefield'].PixelAreaLabel & key).fetch())
    areaLabels = void_session_PixelAreaLabel['area_label']
    areaLabelsSet = list(set(areaLabels))
    preprocessingOutputs['areaLabels'] = areaLabels
    preprocessingOutputs['areaLabelsSet'] = np.array(areaLabelsSet)
    vasc_mask = pd.DataFrame((VM['widefield'].VascMask & key).fetch('mask_binned'))
    voidSync = (VM['widefield'].BehavSync & key)
    voidSync_VelXY = pd.DataFrame(voidSync.fetch('velocity_by_im_frame'))
    #voidSync_Time = pd.DataFrame(voidSync.fetch('im_frame_timestamps'))[0][0][0]

    metaFilterCondition = (voidSync_VelXY.shape[0]>0) and (vasc_mask.shape[0]>0) and (len(areaLabelsSet)>0)

    if metaFilterCondition and loadDFF:
        #VascMask = np.invert(VascMask[0][0].reshape(-1))
        anti_vasc_mask = np.invert(vasc_mask[0][0].reshape(128,128).T.reshape(-1))
        #VascMask = VascMask.T
        #VascMask = VascMask.reshape(-1)

        void_session_DFF = pd.DataFrame((VM['behavior'].BehavioralSession * VM['widefield'].WFsession * VM['widefield'].Dff & key).fetch('dff'))

        if void_session_DFF.shape[0] > 0:

            voidSync_VelXY = voidSync_VelXY[0][0]
            voidSync_Vel = np.sqrt((voidSync_VelXY[:,0] ** 2) + (voidSync_VelXY[:,1] ** 2))            
            voidSync_Vel = np.array(pd.Series(voidSync_Vel).interpolate().tolist())
            preprocessingOutputs['voidSync_Vel'] = voidSync_Vel

            #####
            dffPixels = np.hstack(void_session_DFF[0])
            dffPixels = np.transpose(dffPixels)
            # Remove first {dffSnipSecStart} s and last {dffSnipSecEnd} s to prevent artifacts
            dffPixels = dffPixels[:,int(Fs*dffSnipSecStart):]
            dffPixels = dffPixels[:,:-int(Fs*dffSnipSecEnd)]


            for pixIDX in range(numPixels):
                dffPixels[pixIDX,:] = OODpercentileReplace(dffPixels[pixIDX,:],U_OODpercentileCutoff,L_OODpercentileCutoff)
            
            minPixels = np.percentile(dffPixels,.1,axis=1).reshape(-1,1) # Use percentile due to extreme values
            maxPixels = np.percentile(dffPixels,99.9,axis=1).reshape(-1,1)
            DFF_only = (dffPixels-minPixels)/(maxPixels-minPixels)
            ####

            #DFF_only = void_session_DFF['dff']
            trialLength = DFF_only[0].shape[0]
            preprocessingOutputs['trialLength'] = trialLength
            
            pixelMaskIDX = np.arange(0,numPixels,1)[anti_vasc_mask]
            preprocessingOutputs['pixelMaskIDX'] = pixelMaskIDX



            ##############################################
            ### Butterworth Variance Random Pixel Plot ###
            numRows = 7
            butterworthFilterPixelPlots(projectPath,savePlots,numRows,fileNameSession,DFF_only,pixelMaskIDX,Fs,U_OODpercentileCutoff,L_OODpercentileCutoff)



            noiseFeaturesRaw = np.zeros((pixelMaskIDX.shape[0],4))
            for IDX,pixel in enumerate(pixelMaskIDX):
                currentSignal = DFF_only[pixel,:]
                noiseFeaturesRaw[IDX,0] = max(currentSignal)

                sos = signal.butter(butterOrder, butterThresh, 'lp', fs=Fs, output='sos')
                lowFiltered = signal.sosfilt(sos,currentSignal)
                lowSigVar = np.var(lowFiltered)
                noiseFeaturesRaw[IDX,1] = lowSigVar

                sos = signal.butter(butterOrder, butterThresh, 'hp', fs=Fs, output='sos')
                highFiltered = signal.sosfilt(sos,currentSignal)
                highFiltered = OODpercentileReplace(highFiltered,U_OODpercentileCutoff,L_OODpercentileCutoff)
                highSigVar = np.var(highFiltered)
                noiseFeaturesRaw[IDX,2] = highSigVar

                noiseFeaturesRaw[IDX,3] = lowSigVar / highSigVar
            
            preprocessingOutputs['noiseFeaturesRaw'] = noiseFeaturesRaw
            
            ###########################################
            ### Add plotting for lowVar/highVar SNR ###


            if runGLM:
                ABS_LAG = 20

                if (not os.path.isfile(os.path.join(projectPath,f'{fileNameSession}_pixelGLMresiduals_{version}.npy'))) or (overwriteResiduals):
                    pixelGLMresiduals = np.zeros((pixelMaskIDX.shape[0],DFF_only.shape[1]-(2*ABS_LAG)))
                else:
                    pixelGLMresiduals = np.load(os.path.join(projectPath,f'{fileNameSession}_pixelGLMresiduals_{version}.npy'))
                
                firstPixel = np.min(np.argwhere(np.sum(pixelGLMresiduals,axis=1)==0))

                lastPixel = pixelMaskIDX[-1]
                resSave = 500 #save the residual signals every Nth pixel
                startTime = datetime.now()
                for pixelIDX in np.arange(firstPixel,pixelMaskIDX.shape[0],1):
                    pixel = pixelMaskIDX[pixelIDX]
                    if anti_vasc_mask[pixel]:
                        if Verbose:
                            #print('Currently on pixel '+str(pixel), end='\r')
                            with open(os.path.join(projectPath,f'{fileNameSession}_pixelGLMprogress_{version}.txt'), "w") as file:
                                file.write('Currently on pixel '+str(pixel))
                            if not(pixel == firstPixel):
                                currentTime = datetime.now()
                                timeElapsed = currentTime - startTime
                                timePerPixel = timeElapsed / (pixelIDX - pixelMaskIDX[firstPixel] + 1)
                                pixelsLeft = pixelMaskIDX.shape[0] - pixelIDX
                                timeLeft = pixelsLeft * timePerPixel
                                timeDone = currentTime + timeLeft
                                #print('Currently on pixel '+str(pixel)+'. Time Done Estimate: '+str(timeDone.hour)+':'+str(timeDone.minute)+' on '+str(timeDone.day)+'.'+str(timeDone.month)+'.'+str(timeDone.year)+'          ',end='\r')
                                with open(os.path.join(projectPath,f'{fileNameSession}_pixelGLMprogress_{version}.txt'), "w") as file:
                                    file.write('Currently on pixel '+str(pixel)+'. Time Done Estimate: '+str(timeDone.hour)+':'+str(timeDone.minute)+' on '+str(timeDone.day)+'.'+str(timeDone.month)+'.'+str(timeDone.year))
                        else:
                            with open(os.path.join(projectPath,f'{fileNameSession}_testPixelProgress_{version}.txt'), "a") as file:
                                file.write('\nCurrently on pixel '+str(pixel))

                        DFF = DFF_only[pixel,:]
                        DFF = np.array(np.reshape(DFF,-1))


                        LagsVec = np.arange(-ABS_LAG,ABS_LAG+1,1)
                        Vel_Lag = np.zeros((voidSync_Vel.shape[0], LagsVec.shape[0]))
                        for i,lag in enumerate(LagsVec):
                            Vel_Lag[:, i] = np.roll(voidSync_Vel, lag)
                        Vel_Lag = Vel_Lag[ABS_LAG:Vel_Lag.shape[0]-ABS_LAG,:]
                        DFF = DFF[ABS_LAG:DFF.shape[0]-ABS_LAG]

                        OOD_IDX = np.where(np.abs(DFF)>3*np.std(DFF)) #~0.448% of data is falling outside of 3 standard deviations
                        DFF[OOD_IDX] = np.mean(DFF) #should interpolate

                        #train_Vel,test_Vel,train_DFF,test_DFF = train_test_split(Vel_Lag,DFF, test_size=0.2, random_state=42)
                        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

                        #####################################################
                        ### Construct a GLM for determining DFF residuals ###
                        GLMpredictDFF = []
                        GLMpredictIDX = []
                        for train_index, test_index in kfold.split(DFF):
                            train_Vel = Vel_Lag[train_index]
                            test_Vel = Vel_Lag[test_index]
                            train_DFF = DFF[train_index]
                            test_DFF = DFF[test_index]

                            #GLM (Basic, Identity Linker) with L2 Regularization
                            alphas = np.power(10, np.linspace(3, 6.5, num=10)) #np.power(10, np.linspace(-3, 7, num=30)) #This might be too high, taking too long in processing
                            ridge_weight = []
                            alpha_R2 = []
                            for alpha in alphas:
                                ridge = Ridge(alpha=alpha)
                                ridge.fit(train_Vel, train_DFF)
                                pred_DFF = ridge.predict(test_Vel)
                                #mse_L2 = mean_squared_error(test_DFF,pred_DFF)
                                R2_GLM_L2 = r2_score(test_DFF,pred_DFF)

                                ridge_weight.append(ridge.coef_)
                                alpha_R2.append(R2_GLM_L2)
                        
                            bestAlpha = alphas[np.where(np.array(alpha_R2) == np.max(np.array(alpha_R2)))[0][0]]
                            GLM_R2 = np.max(np.array(alpha_R2)) #R2 of the alpha being selected for the model

                            #now predict test fold using the best alpha
                            ridge = Ridge(alpha=bestAlpha) #choose the best alpha
                            ridge.fit(train_Vel, train_DFF)
                            GLMpredictDFF.append(ridge.predict(test_Vel))
                            GLMpredictIDX.append(test_index)

                        GLMpredictDFF_allFolds = np.concatenate(GLMpredictDFF)
                        GLMpredictIDX_allFolds = np.concatenate(GLMpredictIDX)

                        sorted_indices = np.argsort(GLMpredictIDX_allFolds)
                        GLMpredictDFFsorted = GLMpredictDFF_allFolds[sorted_indices]

                        GLMresiduals = DFF - GLMpredictDFFsorted
                        pixelGLMresiduals[pixelIDX,:] = np.transpose(GLMresiduals)

                        if pixel % resSave == 0:
                            np.save(os.path.join(projectPath,f'{fileNameSession}_pixelGLMresiduals_{version}.npy'),pixelGLMresiduals)
                        #preprocessingOutputs['pixelGLMresiduals'] = pixelGLMresiduals
                np.save(os.path.join(projectPath,f'{fileNameSession}_pixelGLMresiduals_{version}.npy'),pixelGLMresiduals)
                ###########
                ### End GLM

        with open(os.path.join(projectPath,f'{fileNameSession}_preprocessingOutputs_{version}.pickle'), 'wb') as handle:
            pickle.dump(preprocessingOutputs, handle, protocol=pickle.HIGHEST_PROTOCOL)



    if metaFilterCondition and calcTau:
        anti_vasc_mask = np.invert(vasc_mask[0][0].reshape(128,128).T.reshape(-1))

        anti_vascIDX = np.where(anti_vasc_mask)[0]

        with open(os.path.join(projectPath,fileNameSession+'_preprocessingOutputs.pickle'), 'rb') as handle:
            preprocessingOutputs = pickle.load(handle)
            
        print(fileNameSession)
        
        pixelGLMresiduals = np.load(os.path.join(projectPath,fileNameSession+'_pixelGLMresiduals_3.npy'))
        print(f'pixelGLMresiduals_3 shape: {pixelGLMresiduals.shape[0]}')
        
        ABS_LAG = 20
        sampNum = preprocessingOutputs['trialLength']-(2*ABS_LAG)
        noiseFeaturesRaw = preprocessingOutputs['noiseFeaturesRaw']
        corr_t = np.arange(0, correlationWindow, 1/Fs)


        if not justPlot:
            #tau_save = [[] for _ in range(len(areaLabelsSet))]
            tauMatrix = np.zeros((anti_vascIDX.shape[0],16))
            # for SNRidx,SNR in enumerate(SNR_mask):
            #     if SNR:
            startTime = datetime.now()
            for pixelIDX,pixelID in enumerate(anti_vascIDX):
                dFF_snip = pixelGLMresiduals[pixelID,:].reshape(-1)
                if not(sum(dFF_snip) == 0.0):
                    dFF_snip_a = (dFF_snip - np.mean(dFF_snip)) / (np.std(dFF_snip) * len(dFF_snip))
                    dFF_snip_v = (dFF_snip - np.mean(dFF_snip)) /  np.std(dFF_snip)
                    ACF = np.correlate(dFF_snip_a, dFF_snip_v, 'full')
                    ACF_split = ACF[dFF_snip.shape[0]-1:dFF_snip.shape[0]-1+(correlationWindow*Fs)]

                    popt, pcov = curve_fit(decay_func_dual, corr_t, ACF_split, p0_dual, bounds=bounds_2, maxfev=maxfev)
                    np.linalg.cond(pcov)
                    A_fit_0, tau_fit_0, A_fit_1, tau_fit_1, offset_fit = popt

                    popt, pcov = curve_fit(decay_func_mono, corr_t, ACF_split, p0_mono, bounds=bounds_1, maxfev=maxfev)
                    np.linalg.cond(pcov)
                    A_fit_mono, tau_fit_mono, offset_fit_mono = popt
                    #tauMatrix[simNum,2,SNR_IDX] = tau_fit_mono

                    decayFitPointsDual = decay_func_dual(corr_t,A_fit_0,tau_fit_0,A_fit_1,tau_fit_1,offset_fit)
                    R2_Fit_Dual = r2_score(ACF_split, decayFitPointsDual)
                    dualRSS = np.sum((ACF_split-decayFitPointsDual)**2)
                    dualSigmaHat = np.sqrt(dualRSS/ACF_split.shape[0]) #np.std(ACF_split-decayFitPointsMono)
                    dualLL = LL(ACF_split, decayFitPointsDual, dualSigmaHat)
                    dualAIC = AIC(ACF_split.shape[0], dualLL, 5)
                    dualBIC = BIC(ACF_split.shape[0], dualLL, 5)

                    decayFitPointsMono = decay_func_mono(corr_t,A_fit_mono,tau_fit_mono,offset_fit_mono)
                    R2_Fit_Mono = r2_score(ACF_split, decayFitPointsMono)
                    monoRSS = np.sum((ACF_split-decayFitPointsMono)**2)
                    monoSigmaHat = np.sqrt(monoRSS/ACF_split.shape[0]) #np.std(ACF_split-decayFitPointsMono)
                    monoLL = LL(ACF_split, decayFitPointsMono, monoSigmaHat)
                    monoAIC = AIC(ACF_split.shape[0], monoLL, 3)
                    monoBIC = BIC(ACF_split.shape[0], monoLL, 3)

                    deltaAIC = dualAIC - monoAIC
                    akaike_w = np.exp(-0.5 * deltaAIC) / (1 + np.exp(-0.5 * deltaAIC)) # relative probability of the double timescale model being the best fit

                    if dualBIC < monoBIC:
                        bestTau = tau_fit_1
                    else:
                        bestTau = tau_fit_mono
                    tauMatrix[pixelIDX,:] = np.array([bestTau,tau_fit_0,tau_fit_1,R2_Fit_Dual,dualRSS,dualSigmaHat,dualLL,dualAIC,dualBIC,tau_fit_mono,R2_Fit_Mono,monoRSS,monoSigmaHat,monoLL,monoAIC,monoBIC])
                    ########tau_save[areaIDX].append(tau_fit) #tau_save[areaIDX][SNRidx,samp]=tau_fit

                if (pixelIDX%int((anti_vascIDX.shape[0]/20)) == 0) and (pixelIDX != 0):
                    fractionComplete = pixelIDX/anti_vascIDX.shape[0]
                    timeLeft = (startTime-datetime.now()) * ((1-fractionComplete) / fractionComplete)
                    timeDone = startTime + timeLeft
                    print(f'{int(fractionComplete*100)}\u0025 Complete, Time Complete Estimate: {timeDone.hour}:{timeDone.minute} on {timeDone.day}.{timeDone.month}.{timeDone.year}') #% is u0025
            np.save(os.path.join(projectPath,f'{currentSubject}_{currentDate}_{currentSession}_tau_save_{version}.npy'),tauMatrix)

        else:
            tauMatrix = np.load(os.path.join(projectPath,f'{currentSubject}_{currentDate}_{currentSession}_tau_save_{version}.npy'))

            fig, axes = plt.subplots(4,6,figsize=(20,16))
            plt.suptitle(fileNameSession)
            for filterValIDX,filterVal in enumerate([25, 100]):
                filteredTauBest = tauMatrix[:,0].copy()
                filteredTauDual = tauMatrix[:,2].copy()
                filteredTauMono = tauMatrix[:,9].copy()
                for tauIDX in range(tauMatrix.shape[0]):
                    if filteredTauBest[tauIDX] > filterVal:
                        filteredTauBest[tauIDX] = np.zeros(1)*np.nan
                    if filteredTauDual[tauIDX] > filterVal:
                        filteredTauDual[tauIDX] = np.zeros(1)*np.nan
                    if filteredTauMono[tauIDX] > filterVal:
                        filteredTauMono[tauIDX] = np.zeros(1)*np.nan
                for SNRthreshIDX,(title,SNRthresh,filteredTau) in enumerate(zip(['Best Tau Fit (from delta AICc)','Best Tau Fit (from delta AICc)','Dual Tau Fit','Dual Tau Fit','Mono Tau Fit','Mono Tau Fit'],[3,SNRbutterworthThresh,3,SNRbutterworthThresh,3,SNRbutterworthThresh],[filteredTauBest,filteredTauBest,filteredTauDual,filteredTauDual,filteredTauMono,filteredTauMono])):
                    ax = axes[(filterValIDX*2),SNRthreshIDX]
                    passingSNR = np.where(noiseFeaturesRaw[:,3] > SNRthresh)[0]
                    outputsMap = np.zeros(numPixels) * np.nan
                    outputsMap[anti_vascIDX[passingSNR]] = filteredTau[passingSNR]
                    img = ax.imshow(outputsMap.reshape(128,128))
                    ax.axis('off')
                    ax.set_title(f'{title},\nSNR Threshold:{SNRthresh}')
                    cbar = plt.colorbar(img,ax=ax)
                    ax = axes[(filterValIDX*2)+1,SNRthreshIDX]
                    if not np.isnan(filteredTau[passingSNR]).all():
                        ax.hist(filteredTau[passingSNR],bins=50,color='black')
                    if SNRthreshIDX == 0:
                        ax.set_ylabel('Pixel Count')
                    ax.set_xlabel('Tau (s)')
            if savePlots:
                plt.savefig(os.path.join(projectPath,f'{fileNameSession}_TauDistributions_AICc_BIC.pdf'),dpi=600,bbox_inches="tight")
            plt.close()
            
            deltaBIC = tauMatrix[:,8]-tauMatrix[:,15]
            deltaAIC = tauMatrix[:,7]-tauMatrix[:,14]
            fig, axes = plt.subplots(1,3,figsize=(16,5))
            plt.suptitle(fileNameSession)
            for plotIDX,(deltaModel,title) in enumerate(zip([deltaBIC,deltaAIC,deltaBIC-deltaAIC],['delta BIC (dual-mono)','delta AICc (dual-mono)','delta (delta BIC, delta AICc)'])):
                ax = axes[plotIDX]
                outputsMap = np.zeros(numPixels) * np.nan
                outputsMap[anti_vascIDX] = deltaModel #np.exp(-0.5 * deltaBIC) / (1 + np.exp(-0.5 * deltaBIC))
                img = ax.imshow(outputsMap.reshape(128,128))
                ax.axis('off')
                cbar = plt.colorbar(img,ax=ax)
                ax.set_title(f'{title}')
            if savePlots:
                plt.savefig(os.path.join(projectPath,f'{fileNameSession}_delta_AICc_BIC.pdf'),dpi=600,bbox_inches="tight")
            plt.close()

            #plotTauMap()
            #plt.savefig(os.path.join(projectPath,currentSubject+'_'+str(currentDate)+'_'+str(currentSession)+'_LUA'+str(L_A)+'_'+str(U_A)+'_LUT'+str(L_T)+'_'+str(U_T)+'_LUO'+str(L_O)+'_'+str(U_O)+'_tauMap.pdf'),dpi=600,bbox_inches='tight')

        
        
    if metaFilterCondition and processTau:
        anti_vasc_mask = np.invert(vasc_mask[0][0].reshape(128,128).T.reshape(-1))

        anti_vascIDX = np.where(anti_vasc_mask)[0]
        
        # tau_array = np.array([np.array(tau_save_i) for tau_save_i in tau_save])
        tau_save = np.load(os.path.join(projectPath,f'{fileNameSession}_tau_save_{version}.npy'))

        with open(os.path.join(projectPath,fileNameSession+'_preprocessingOutputs.pickle'), 'rb') as handle:
            preprocessingOutputs = pickle.load(handle)

        structureDefinitions = []
        for mergeArea in structListMerge:
            if (mergeArea == 'SS') or (mergeArea == 'RSP'):
                area_list = []
                for subArea in preprocessingOutputs['areaLabelsSet']:
                    if subArea[:len(mergeArea)] == mergeArea:
                        currentAreaIDX = list(set(np.where(preprocessingOutputs['areaLabelsSet'] == subArea)[0]))
                        area_list.append(currentAreaIDX)
                structureDefinitions.append(area_list)
            else:
                structureDefinitions.append(list(np.where(preprocessingOutputs['areaLabelsSet'] == mergeArea)[0]))

        tau_area_processed = np.zeros((structListMerge.shape[0],6))

        
        noiseFeaturesRaw = preprocessingOutputs['noiseFeaturesRaw']
        newSNRthresh = 3
        pixelsPassingSNR = anti_vascIDX[np.array(np.where(noiseFeaturesRaw[:,3] > newSNRthresh))[0]]

        q = (VM['widefield'].ReferenceIm & key)
        tform_allen2mouse = q.fetch('tform_allen2mouse')[0]
        tform_mouse2Tallen = np.linalg.inv(tform_allen2mouse.T)
        MOUSEbregma = q.fetch('bregma')[0][0]
        MOUSElambda = q.fetch('lambda')[0][0]
        Tallen_bregma = (tform_mouse2Tallen @ np.hstack((MOUSEbregma,np.array(1))).reshape(-1,1))[0:-1]
        Tallen_lambda = (tform_mouse2Tallen @ np.hstack((MOUSElambda,np.array(1))).reshape(-1,1))[0:-1]

        Tallen_Grid = tform_mouse2Tallen @ grid[pixelsPassingSNR].T
        CCF_Grid = tform_Tallen2CCF @ Tallen_Grid


        cmap = plt.get_cmap('cool')
        global_min = 0
        global_max = 30
        norm = matplotlib.colors.Normalize(global_min, global_max)
        maskedIDXpassingIDXs = np.asarray([np.where(anti_vascIDX == i)[0][0] for i in pixelsPassingSNR])
        tau_colors = cmap(norm(tau_save[maskedIDXpassingIDXs,0]))
        
        currentMeanMLcoord = np.mean([Tallen_bregma[0],Tallen_lambda[0]])
        MLdepartureTolerance = 15
        #CCF_Grid[0,:] = CCF_Grid[0,:] - (np.ones(pixelsPassingSNR.shape[0]) * trueCenterML)
        if abs(currentMeanMLcoord-trueCenterML) < MLdepartureTolerance:
            allTauCCF_Coords = np.hstack((np.vstack((CCF_Grid[0:2,:],tau_save[maskedIDXpassingIDXs,0])),allTauCCF_Coords))
            #Tallen_bregma_list.append(Tallen_bregma)
            #Tallen_lambda_list.append(Tallen_lambda)
            axCCF[0].scatter(Tallen_Grid[0,:],Tallen_Grid[1,:],s=0.5,color=tau_colors)
            axCCF[0].scatter(Tallen_bregma[0],Tallen_bregma[1],color='green',marker='x')
            axCCF[0].scatter(Tallen_lambda[0],Tallen_lambda[1],color='red',marker='x')
            axCCF[1].scatter(CCF_Grid[0,:],CCF_Grid[1,:],s=0.5,color=tau_colors)
            axCCF[1].scatter(CCF_bregma[0],CCF_bregma[1],color='green',marker='x')
            axCCF[1].scatter(CCF_lambda[0],CCF_lambda[1],color='red',marker='x')
        
        #<- <-
        # mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # mappable.set_array(tau_save[maskedIDXpassingIDXs,0])
        # cbar = plt.colorbar(mappable, ax=axCCF, shrink=0.5, aspect=5)
        # cbar_ticks = np.arange(global_min, global_max, 1)
        # cbar.set_ticks(cbar_ticks)
        # plt.savefig(os.path.join(projectPath,f"{lineSelection}_CCFAlignedCoords.pdf"),dpi=100,bbox_inches="tight")



        maskedAreaLabels = preprocessingOutputs['areaLabels'][anti_vascIDX]
        
        print(f'{fileNameSession}')
        for areaIDX,area in enumerate(structureDefinitions):
            areaIDXs = []
            for subarea in area:
                areaIDXs.append([i for i in maskedAreaLabels.index if maskedAreaLabels[i] == preprocessingOutputs['areaLabelsSet'][subarea]])
            
            areaIDXs = [x for xs in areaIDXs for x in xs]

            passingAreaIDXs = np.array(list(set(pixelsPassingSNR).intersection(set(areaIDXs))))
            areaPassingPixelCounts = []
            if passingAreaIDXs.shape[0] > 5:
                maskedIDXpassingAreaIDXs = [np.where(anti_vascIDX == i)[0][0] for i in passingAreaIDXs]
                currentTaus = tau_save[maskedIDXpassingAreaIDXs,0]
                filteredCurrentTaus = currentTaus[list(set(np.where(currentTaus<longestToleratedTau-0.25)[0]).intersection(set(np.where(currentTaus>0.5)[0])))]
                
                #cross_mouse_region_tau_pop[areaIDX].append(list(filteredCurrentTaus))
                cross_mouse_region_tau_pop[areaIDX] = np.vstack((cross_mouse_region_tau_pop[areaIDX],filteredCurrentTaus.reshape(-1,1)))

                gamma_a,gamma_loc,gamma_scale = gamma.fit(filteredCurrentTaus)
                tau_area_processed[areaIDX,3] = gamma_a
                tau_area_processed[areaIDX,4] = gamma_loc
                tau_area_processed[areaIDX,5] = gamma_scale
                
                tau_area_processed[areaIDX,0] = np.mean(filteredCurrentTaus)
                tau_area_processed[areaIDX,1] = np.std(filteredCurrentTaus)
                areaPassingPixelCounts.append(passingAreaIDXs.shape[0])
                print(f'{structListMerge[areaIDX]}:{passingAreaIDXs.shape[0]}, TauMean:{round(tau_area_processed[areaIDX,0],3)}, TauSD:{round(tau_area_processed[areaIDX,1],3)}')
                # plt.figure()
                # plt.hist(tau_save[passingAreaIDXs],bins=80,color='black')
                # plt.title(f'{area}')
                # plt.xlim(0,50)
                tau_area_processed[areaIDX,2] = filteredCurrentTaus.shape[0]
            else:
                tau_area_processed[areaIDX,0] = 0 * np.nan
                tau_area_processed[areaIDX,1] = 0 * np.nan
                tau_area_processed[areaIDX,2] = np.shape(passingAreaIDXs)[0]
        np.save(os.path.join(projectPath,f'{fileNameSession}_tau_area_processed_{version}'),tau_area_processed)

        fig, axes = plt.subplots(1,2,figsize=(10,5))
        fig.suptitle(f'{fileNameSession}, SNRthresh:{newSNRthresh}')
        for regionIDX in range(structListMerge.shape[0]):
            if not np.isnan(tau_area_processed[regionIDX,0]):
                region_mu = tau_area_processed[regionIDX,0]
                region_sigma = tau_area_processed[regionIDX,1]
                region_pdf_x = np.linspace(region_mu - 4*region_sigma, region_mu + 4*region_sigma)
                region_pdf = (1 / (region_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((region_pdf_x - region_mu) / region_sigma)**2)
                axes[0].plot(region_pdf_x, region_pdf, color=areaColors[regionIDX])
        axes[0].set_xlim(np.mean(tau_area_processed[regionIDX,0])-(7*np.mean(np.mean(tau_area_processed[regionIDX,1]))),np.mean(tau_area_processed[regionIDX,0])+(7*np.mean(np.mean(tau_area_processed[regionIDX,1]))))
        axes[0].set_ylabel('PDF')
        axes[0].set_xlabel('Tau (s)')
        axes[0].set_title('Gaussian Fit')

        for regionIDX in range(structListMerge.shape[0]):
            if not np.isnan(tau_area_processed[regionIDX,0]):
                region_a = tau_area_processed[regionIDX,3]
                region_loc = tau_area_processed[regionIDX,4]
                region_scale = tau_area_processed[regionIDX,5]
                rv = gamma(region_a,region_loc,region_scale)
                pdf_x = np.arange(-10, 50, .1)
                axes[1].plot(pdf_x, rv.pdf(pdf_x),color=areaColors[regionIDX],label=structListMerge[regionIDX])
        #axes[1].set_xlim(0,20)
        axes[1].set_xlim(np.mean(tau_area_processed[regionIDX,0])-(7*np.mean(np.mean(tau_area_processed[regionIDX,1]))),np.mean(tau_area_processed[regionIDX,0])+(7*np.mean(np.mean(tau_area_processed[regionIDX,1]))))
        axes[1].set_title('Gamma Fit')
        axes[1].set_xlabel('Tau (s)')
        axes[1].legend()
        if savePlots:
            plt.savefig(os.path.join(projectPath,f'{fileNameSession}_regionGammaFits.pdf'),dpi=600,bbox_inches="tight")



    if (voidSync_VelXY.shape[0]>0) and (anti_vasc_mask.shape[0]>0) and (len(areaLabelsSet)>0) and summarizeLineTau:
        tau_area_processed = np.load(os.path.join(projectPath,f'{fileNameSession}_tau_area_processed_{version}.npy'))

        for region in range(tau_area_processed.shape[0]):
            crossMouseRegionTotal[region] += tau_area_processed[region,2]
            mean_gamma_params[region,0] += tau_area_processed[region,3]*tau_area_processed[region,2]
            mean_gamma_params[region,1] += tau_area_processed[region,4]*tau_area_processed[region,2]
            mean_gamma_params[region,2] += tau_area_processed[region,5]*tau_area_processed[region,2]
        

print(f'{lineSelection}')
pop_tau_params = np.zeros((structListMerge.shape[0],5))
fig, axes = plt.subplots(1,3,figsize=(22,5))
cross_mouse_gamma_params = np.copy(mean_gamma_params)
for regionIDX in range(structListMerge.shape[0]):
    cross_mouse_gamma_params[regionIDX,:] = mean_gamma_params[regionIDX,:] / crossMouseRegionTotal[regionIDX]

    pop_region_mu = np.mean(cross_mouse_region_tau_pop[regionIDX])
    pop_region_sigma = np.std(cross_mouse_region_tau_pop[regionIDX])
    print(f'X-Mouse {structListMerge[regionIDX]} ({cross_mouse_region_tau_pop[regionIDX].shape[0]}): Tau Mu:{round(pop_region_mu,3)}, SD:{round(pop_region_sigma,3)}')
    pop_region_pdf_x = np.arange(-10, 30, 0.1)
    pop_region_pdf = (1 / (pop_region_sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((pop_region_pdf_x - pop_region_mu) / pop_region_sigma)**2)
    axes[0].plot(pop_region_pdf_x, pop_region_pdf, color=areaColors[regionIDX])

    axes[1].hist(cross_mouse_region_tau_pop[regionIDX],bins=75,color=areaColors[regionIDX],alpha=0.5)

    pdf_x = np.arange(0, 50, 0.1)
    pop_gamma_a,pop_gamma_loc,pop_gamma_scale = gamma.fit(cross_mouse_region_tau_pop[regionIDX])
    rv = gamma(pop_gamma_a,pop_gamma_loc,pop_gamma_scale)
    axes[2].plot(pdf_x, rv.pdf(pdf_x),color=areaColors[regionIDX],label=structListMerge[regionIDX])

    pop_tau_params[regionIDX,:] = np.array((pop_gamma_a,pop_gamma_loc,pop_gamma_scale,pop_region_mu,pop_region_sigma))

#axes.set_xlim(-20,200)
axes[0].set_title('Cross Mouse Population Gaussian')
axes[0].set_xlabel('Tau(s)')
axes[0].set_xlim(-2,20)
axes[1].set_title('Cross Mouse Full Tau Distribution')
axes[1].set_xlabel('Tau (s)')
axes[2].set_title('Cross Mouse Population Gamma Fit')
axes[2].set_xlabel('Tau (s)')
axes[2].set_xlim(0,20)
axes[2].legend()
plt.suptitle(f'{lineSelection}, SNRthresh:{newSNRthresh}')

if savePlots:
    plt.savefig(os.path.join(projectPath,f'crossMouse_regionGammaFits.pdf'),dpi=600,bbox_inches="tight")

np.save(os.path.join(projectPath,f'crossMouse_regionTauFits'),pop_tau_params)
np.save(os.path.join(projectPath,f'{lineSelection}_crossMouse_regionalFullTauDist'),np.array(cross_mouse_region_tau_pop, dtype=object), allow_pickle=True)



np.save(os.path.join(projectPath,f'{lineSelection}_tauCCF'),allTauCCF_Coords)

