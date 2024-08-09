import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from statistics import fmean as mean
import utils.connect_to_dj as connect_to_dj
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from scipy.sparse import csc_matrix
import scipy.io
from scipy import stats
from scipy.stats import gamma
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
import pickle
import platform
import sys
import os
import h5py
import numpy as np
import random
from random import choices
import pandas as pd
import datajoint
from statsmodels.regression.linear_model import WLS
from pypdf import PdfMerger


def layerRegressions(pred_dim,n_splits,highExpressionGeneIDXs,x_data,y_data,layerNames,regressionConditions,cell_region,alphaParams):
    numLayers = len(layerNames)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    GLMpredictDFFsorted = [[] for _ in range(numLayers)]
    alphas = np.power(10, np.linspace(alphaParams[0],alphaParams[1],num=alphaParams[2])) #np.linspace(0.0, 1.0, num=10) #np.power(10, np.linspace(-3, 7, num=30)) #This might be too high, taking too long in processing
    lasso_betas = [np.zeros((alphas.shape[0],pred_dim,x_data[layerIDX].shape[1])) for layerIDX in range(numLayers)]
    alpha_R2 = np.zeros((numLayers,alphas.shape[0]))
    bestAlpha = np.zeros((numLayers,n_splits))
    bestR2 = np.zeros((numLayers,n_splits))
    best_betas = [np.zeros((n_splits,pred_dim,x_data[layerIDX].shape[1])) for layerIDX in range(numLayers)]
    #tauPredictions = [[[] for _ in range(n_splits)] for _ in range(numLayers)]
    tauPredictions = [[np.empty((0,2*pred_dim+1)) for _ in range(n_splits)] for _ in range(numLayers)]
    for layerIDX, layer in enumerate(layerNames):
        print(f'Fitting: {layerNames[layerIDX]}')
        GLMpredictTau = []
        GLMpredictIDX = []
        for foldIDX,(train_index, test_index) in enumerate(kfold.split(y_data[layerIDX])):
            
            train_y = y_data[layerIDX][train_index,:]
            test_y = y_data[layerIDX][test_index,:]
            #train_w = tau_SD_per_cell_H2layerFiltered[layerIDX][train_index,:]**-1
            #test_w = tau_SD_per_cell_H2layerFiltered[layerIDX][test_index,:]**-1
            train_x = np.asarray(x_data[layerIDX][train_index,:])
            test_x = np.asarray(x_data[layerIDX][test_index,:])
            if regressionConditions[3]: #genePredictors condition, exclude low-expression genes
                train_x = train_x[:,highExpressionGeneIDXs[layerIDX]]
                test_x = test_x[:,highExpressionGeneIDXs[layerIDX]]
            

            #GLM (Basic, Identity Linker) with L1 Regularization
            for alphaIDX,alpha in enumerate(alphas):
                lasso = Lasso(alpha=alpha)
                lasso.fit(train_x, train_y)
                pred_y = lasso.predict(test_x)
                R2_GLM_L1 = r2_score(test_y,pred_y)
                lasso_betas[layerIDX][alphaIDX,:,:] = lasso.coef_
                alpha_R2[layerIDX,alphaIDX] = R2_GLM_L1

            bestAlpha[layerIDX,foldIDX] = alphas[np.where(np.array(alpha_R2[layerIDX,:]) == np.max(np.array(alpha_R2[layerIDX,:])))[0][0]]
            bestR2[layerIDX,foldIDX] = np.max(np.array(alpha_R2[layerIDX,:])) #R1 of the alpha being selected for the model
            
            # ### L1 WLS regression, !!!to do: cross validate alpha!!!
            # L1_WLS = WLS(endog=train_y,exog=np.hstack((np.ones((train_x.shape[0],1)),train_x)),weights=train_w).fit_regularized(method='elastic_net', alpha=10e-4, L1_wt=1.0)
            # L1_WLS_pred_y = L1_WLS.predict(np.hstack((np.ones((test_x.shape[0],1)),test_x)))
            # r2_score(test_y,L1_WLS_pred_y)

            #now predict test fold using the best alpha
            lasso = Lasso(alpha=bestAlpha[layerIDX,foldIDX]) #choose the best alpha
            lasso.fit(train_x, train_y)
            best_betas[layerIDX][foldIDX,:,:] = lasso.coef_
            pred_y = lasso.predict(test_x)
            cell_region_IDX = (cell_region[layerIDX][test_index,:]).reshape(-1)
            #tauPredictions[layerIDX][foldIDX].append([test_y,pred_y,cell_region_IDX])
            #print(test_y.shape)
            #print(pred_y.shape)
            #print(cell_region_IDX.shape)
            predCat = np.hstack((test_y.reshape(-1,pred_dim),pred_y.reshape(-1,pred_dim),cell_region_IDX.reshape(-1,1)))
            #print(predCat.shape)
            tauPredictions[layerIDX][foldIDX] = np.vstack((tauPredictions[layerIDX][foldIDX][:,:],predCat))
            GLMpredictTau.append(pred_y)
            GLMpredictIDX.append(test_index)

        #GLMpredictTau_allFolds = np.concatenate(GLMpredictTau)
        #GLMpredictIDX_allFolds = np.concatenate(GLMpredictIDX)
        
        #sorted_indices = np.argsort(GLMpredictIDX_allFolds)
        #GLMpredictDFFsorted[layerIDX] = GLMpredictTau_allFolds[sorted_indices]
    
    return best_betas,lasso_betas,bestAlpha,alphas,tauPredictions,bestR2



def PDFmerger(path,fileBefore,VARS,fileAfter,fileOut):
    pdfs = []
    for currentPDF in VARS:
        pdfs.append(os.path.join(path,fileBefore+str(currentPDF)+fileAfter))
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(os.path.join(path,fileOut))
    merger.close()
    for currentPDF in VARS:
        os.remove(os.path.join(path,fileBefore+str(currentPDF)+fileAfter))
