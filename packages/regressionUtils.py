from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import os
import numpy as np
from random import choices
from statsmodels.regression.linear_model import WLS
from pypdf import PdfMerger
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def layerRegressions(response_dim,n_splits,highMeanPredictorIDXs,x_data,y_data,layerNames,regressionConditions,cell_region,alphaParams,max_iter):
    numLayers = len(layerNames)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    GLMpredictDFFsorted = [[] for _ in range(numLayers)]
    alphas = np.power(10, np.linspace(alphaParams[0],alphaParams[1],num=alphaParams[2])) #np.linspace(0.0, 1.0, num=10) #np.power(10, np.linspace(-3, 7, num=30)) #This might be too high, taking too long in processing
    lasso_betas = [np.zeros((alphas.shape[0],response_dim,highMeanPredictorIDXs[layerIDX].shape[0])) for layerIDX in range(numLayers)]
    alpha_R2 = np.zeros((numLayers,alphas.shape[0]))
    bestAlpha = np.zeros((numLayers,n_splits))
    bestR2 = np.zeros((numLayers,n_splits))
    best_betas = [np.zeros((n_splits,response_dim,highMeanPredictorIDXs[layerIDX].shape[0])) for layerIDX in range(numLayers)]
    loss_history_test_global = [[[] for _ in range(n_splits)] for _ in range(numLayers)]
    loss_history_train_global = [[[] for _ in range(n_splits)] for _ in range(numLayers)]
    dual_gap_history_global = [[[] for _ in range(n_splits)] for _ in range(numLayers)]
    #tauPredictions = [[[] for _ in range(n_splits)] for _ in range(numLayers)]
    if len(cell_region) == 0:
        predAnnotationColumn = 0
    else:
        predAnnotationColumn = 1
    tauPredictions = [[np.empty((0,2*response_dim+predAnnotationColumn)) for _ in range(n_splits)] for _ in range(numLayers)]
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
            #if regressionConditions[3]: #genePredictors condition, exclude low-expression genes
            train_x = train_x[:,highMeanPredictorIDXs[layerIDX]]
            test_x = test_x[:,highMeanPredictorIDXs[layerIDX]]
            

            #GLM (Basic, Identity Linker) with L1 Regularization
            for alphaIDX,alpha in enumerate(alphas):
                lasso = Lasso(alpha=alpha, max_iter=max_iter)
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
            lasso = Lasso(alpha=bestAlpha[layerIDX,foldIDX], warm_start=True, max_iter=1) #choose the best alpha
            
            loss_history_test = []
            loss_history_train = []
            dual_gap_history = []
            for iter in range(max_iter):
                if iter == max_iter-1:
                    print(f'Last iteration of fold {foldIDX}...')

                lasso.fit(train_x, train_y)
                lasso_penalty = lasso.alpha * np.sum(np.abs(lasso.coef_))

                pred_y_test = lasso.predict(test_x)
                mse_test = mean_squared_error(test_y,pred_y_test)
                test_loss = 0.5 * mse_test + lasso_penalty
                loss_history_test.append(test_loss)

                pred_y_train = lasso.predict(train_x)
                mse_train = mean_squared_error(train_y,pred_y_train)
                train_loss = 0.5 * mse_train + lasso_penalty
                loss_history_train.append(train_loss)

                dual_gap_history.append(lasso.dual_gap_ + 10e-17) #add small offset for log plotting later

                #loss_history.append(lasso.loss_)
            loss_history_test_global[layerIDX][foldIDX] = loss_history_test
            loss_history_train_global[layerIDX][foldIDX] = loss_history_train
            dual_gap_history_global[layerIDX][foldIDX] = dual_gap_history
            
            best_betas[layerIDX][foldIDX,:,:] = lasso.coef_
            pred_y = lasso.predict(test_x)
            if predAnnotationColumn == 1:
                cell_region_IDX = (cell_region[layerIDX][test_index,:]).reshape(-1)
            #tauPredictions[layerIDX][foldIDX].append([test_y,pred_y,cell_region_IDX])
            #print(test_y.shape)
            #print(pred_y.shape)
            #print(cell_region_IDX.shape)
            if predAnnotationColumn == 1:
                predCat = np.hstack((test_y.reshape(-1,response_dim),pred_y.reshape(-1,response_dim),cell_region_IDX.reshape(-1,1)))
            else:
                predCat = np.hstack((test_y.reshape(-1,response_dim),pred_y.reshape(-1,response_dim)))
            #print(predCat.shape)
            tauPredictions[layerIDX][foldIDX] = np.vstack((tauPredictions[layerIDX][foldIDX][:,:],predCat))
            GLMpredictTau.append(pred_y)
            GLMpredictIDX.append(test_index)

        #GLMpredictTau_allFolds = np.concatenate(GLMpredictTau)
        #GLMpredictIDX_allFolds = np.concatenate(GLMpredictIDX)
        
        #sorted_indices = np.argsort(GLMpredictIDX_allFolds)
        #GLMpredictDFFsorted[layerIDX] = GLMpredictTau_allFolds[sorted_indices]
    
    return best_betas,lasso_betas,bestAlpha,alphas,tauPredictions,bestR2,loss_history_test_global,loss_history_train_global,dual_gap_history_global



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



def predictor_response_info(x_data,y_data):
    print(f'Predictors Info: [by layer]')
    print(f'rank: {[np.linalg.matrix_rank(x) for x in x_data]}')
    print(f'shape: {[x.shape for x in x_data]}')
    print(f'dtype: {[x.dtype for x in x_data]}')
    print(f'condition number: {[round(np.linalg.cond(x),3) for x in x_data]}')

    print(f'Response Variable(s) Info: [by layer]')
    print(f'rank: {[np.linalg.matrix_rank(y) for y in y_data]}')
    print(f'shape: {[y.shape for y in y_data]}')
    print(f'dtype: {[y.dtype for y in y_data]}')
    print(f'condition number: {[round(np.linalg.cond(y),3) for y in y_data]}')
