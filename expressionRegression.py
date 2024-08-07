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


areaColors = ['#ff0000','#ff704d',                      #MO, reds
            '#4dd2ff','#0066ff','#003cb3','#00ffff',    #VIS, blues
            '#33cc33',                                  #SSp, greens
            '#a366ff']                                  #RSP, purples


# Connects to database and creates virtual modules
VM = connect_to_dj.get_virtual_modules()

standard_scaler = StandardScaler()

%cd C:\Users\lai7370\Projects\CorticalRNN\Organized

##################################
### Script Parameters and Settings
lineSelection = 'Cux2-Ai96'
#lineSelection = 'Rpb4-Ai96'
loadPilot = True
doPlots = True
savePlots = False

structListMerge = np.array(['MOp','MOs','VISa','VISp','VISam','VISpm','SS','RSP'])
structList = structListMerge


tauRegression = True
structNum = structList.shape[0]
applyLayerSpecificityFilter = False #ensure that CCM coordinates are contained within a layer specified in layerAppend
layerAppend = '2/3'
#groupSelector = 12  #12 -> IT_7  -> L2/3 IT
                    #4  -> IT_11 -> L4/5 IT
                    #14 -> IT_9  -> L5 IT
                    #11 -> IT_6  -> L6 IT

calculatingPools = False #first time running for a particular grouping set to True, else the script assumes that pooling has been calculated previously and loads from projectPath
APpool = 1 #Antior-Posterior axis pooling size
MLpool = 1 #Medial-Lateral axis pooling size
bootstrapIterations = 1000

if applyLayerSpecificityFilter:
    structList = [x+layerAppend for x in structList]


################################
### CCF Reference Space Creation
#see link for CCF example scripts from the allen: allensdk.readthedocs.io/en/latest/_static/examples/nb/reference_space.html
output_dir = './CellTypeTestingOut/nrrd25'
reference_space_key = os.path.join('annotation', 'ccf_2017')
resolution = 25
rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=Path(output_dir) / 'manifest.json')
# ID 1 is the adult mouse structure graph
tree = rspc.get_structure_tree(structure_graph_id=1)

annotation, meta = rspc.get_annotation_volume() #in browser navigate to the .nrrd file and download manually,
# The file should be moved to the reference space key directory, only needs to be done once
os.listdir(Path(output_dir) / reference_space_key)
rsp = rspc.get_reference_space()



my_os = platform.system()
if my_os == 'Linux':
    uIn = True
if my_os == 'Windows':
    uIn = False

if uIn:
    lineSelection  = ['Rpb4-Ai96','Cux2-Ai96','C57BL6/J','PV-Ai96'][int(sys.argv[2])]

lineFilter = pd.DataFrame((VM['subject'].Subject * VM['session'].Session * VM['behavior'].BehavioralSession & 'experiment_type="widefield"' & 'task="IntoTheVoid"' & 'line="'+lineSelection+'"').fetch())
if lineSelection == 'C57BL6/J':
    lineFilter = pd.DataFrame((VM['subject'].Subject * VM['session'].Session * VM['behavior'].BehavioralSession & 'experiment_type="widefield"' & 'line="'+lineSelection+'"').fetch())
    lineSelection = 'C57BL6J'

projectFolder = "lineFilter" + lineSelection

my_os = platform.system()
if my_os == 'Linux':
    tauPath = os.path.join(r'/mnt/fsmresfiles/Tau_Processing/',projectFolder+'/')
    savePath = os.path.join(r'/mnt/fsmresfiles/Tau_Processing/H3/')
if my_os == 'Windows':
    tauPath = os.path.join(r'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing',projectFolder)
    savePath = os.path.join(r'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing\H3')
        


if loadPilot:
    projectPath = r'c:\Users\lai7370\OneDrive - Northwestern University\PilotData'

    PilotData = scipy.io.loadmat(os.path.join(projectPath,'filt_neurons_fixedbent_CCF.mat'))

    gene_data = PilotData['filt_neurons']['expmat'][0][0]
    gene_data_dense = csc_matrix.todense(gene_data)
    gene_names = PilotData['filt_neurons']['genes'][0][0]
    geneNames = [item[0] for item in gene_names[:,0]]

    clustid = PilotData['filt_neurons']['clustid'][0][0]
    fn_clustid = [[id[0] for id in cell] for cell in clustid]
    fn_clustid = np.array(fn_clustid).reshape(-1)

    fn_slice = PilotData['filt_neurons']['slice'][0][0]
    fn_slice = np.array(fn_slice).reshape(-1)

    fn_pos = PilotData['filt_neurons']['pos'][0][0]

    fn_CCF = PilotData['filt_neurons']['CCF'][0][0]

total_genes = gene_data_dense.shape[1]

H3_names = set(fn_clustid)
H2_names = []
for curstate in H3_names:
    if (curstate != 'non_Exc') and (curstate != 'qc-filtered'):
        H2_names.append(curstate[:-1])
    else:
        H2_names.append(curstate)
grouping = sorted(list(set(H2_names)))

layerNames  = ['L2_3 IT',   'L4_5 IT',  'L5 IT',    'L6 IT',    'L5 ET']
layerIDs    = [12,          4,          14,         11,         17]
numLayers = len(layerIDs)



cell_region = (np.ones(fn_CCF.shape[0])*-1).astype(int)
for structIDX,structureOfInterest in enumerate(structList):
    print(structIDX)
    structureTree = tree.get_structures_by_acronym([structureOfInterest])
    structureName = structureTree[0]['name']
    structureID = structureTree[0]['id']
    structure_mask = rsp.make_structure_mask([structureID])

    plt.figure()
    plt.title(f'{structureOfInterest}')
    plt.imshow(np.mean(structure_mask,axis=1))

    for cell in range(fn_CCF.shape[0]):
        currentMask = structure_mask[round(fn_CCF[cell,0]),round(fn_CCF[cell,1]),round(fn_CCF[cell,2])]
        if currentMask > 0:
            cell_region[cell] = structIDX


####################################
### Display Specific Gene Expression
geneNameOI = 'Grik1'
geneOI = np.where(gene_names==geneNameOI)[0][0]
geneOI_IDXs = np.where(gene_data_dense[:,geneOI] > 0)[0]
for view in [[0,1],[1,2],[0,2]]:
    plt.figure()
    plt.scatter(fn_CCF[geneOI_IDXs,view[0]],fn_CCF[geneOI_IDXs,view[1]],color='black',s=1,alpha=(gene_data_dense[geneOI_IDXs,geneOI]/np.max(gene_data_dense[geneOI_IDXs,geneOI])))



pop_tau_params = np.load(os.path.join(tauPath,f'crossMouse_regionTauFits.npy'))
fullTauDist = np.load(os.path.join(tauPath,f'{lineSelection}_crossMouse_regionalFullTauDist.npy'), allow_pickle=True)

tau_per_cell = np.zeros(cell_region.shape[0])
for regionIDX in range(len(structList)):
    regionCells = np.where(cell_region == regionIDX)[0]
    pop_gamma_a,pop_gamma_loc,pop_gamma_scale,pop_gaussian_mu,pop_gaussian_sigma = pop_tau_params[regionIDX,:]
    #regionTausFromGammaDist = gamma.rvs(pop_gamma_a, pop_gamma_loc, pop_gamma_scale, size=regionCells.shape[0])
    regionTausFromFullDist = np.array(random.choices(fullTauDist[regionIDX], k=regionCells.shape[0])).reshape(-1)
    tau_per_cell[regionCells] = regionTausFromFullDist #regionTausFromGammaDist

from collections import Counter
occurrences = Counter(cell_region)


H2_all = [s[:-1] for s in fn_clustid]


regionalResample = False
regional_resampling = 3000
tau_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
cell_region_H2layerFiltered = [np.empty((0,1)).astype(int) for _ in range(numLayers)]
#tau_SD_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
gene_data_dense_H2layerFiltered = [np.empty((0,gene_data_dense.shape[1])) for _ in range(numLayers)]
mlCCF_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
apCCF_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
mean_expression = np.zeros((numLayers,len(structList),gene_data_dense.shape[1]))
#sigma_expression = np.zeros((numLayers,len(structList),gene_data_dense.shape[1]))
for layerIDX,(layer,layerName) in enumerate(zip(layerIDs,layerNames)):
    for structIDX,structureOfInterest in enumerate(structList):
        layerIDXs = set([i for i, s in enumerate(H2_all) if s == grouping[layer]])
        regionIDXs = set(np.where(cell_region == structIDX)[0])
        H2layerFilter = list(layerIDXs&regionIDXs)
        print(f'layer:{layerName}, {structureOfInterest}, count: {len(H2layerFilter)}')

        mean_expression[layerIDX,structIDX,:] = np.mean(gene_data_dense[H2layerFilter,:],0)
        #sigma_expression[layerIDX,structIDX,:] = np.std(gene_data_dense[H2layerFilter,:],0)

        if not regionalResample: #equal representation to each cortical region
            if len(H2layerFilter) > 0: #resample with replacement
                H2layerFilter = random.choices(H2layerFilter, k=regional_resampling)
        
        mlCCF_per_cell_H2layerFiltered[layerIDX] = np.vstack((mlCCF_per_cell_H2layerFiltered[layerIDX],fn_CCF[H2layerFilter,2].reshape(-1,1)))
        apCCF_per_cell_H2layerFiltered[layerIDX] = np.vstack((apCCF_per_cell_H2layerFiltered[layerIDX],fn_CCF[H2layerFilter,0].reshape(-1,1)))
        tau_per_cell_H2layerFiltered[layerIDX] = np.vstack((tau_per_cell_H2layerFiltered[layerIDX],tau_per_cell[H2layerFilter].reshape(-1,1)))
        cell_region_H2layerFiltered[layerIDX] = np.vstack((cell_region_H2layerFiltered[layerIDX],cell_region[H2layerFilter].reshape(-1,1)))
        #tau_SD_per_cell_H2layerFiltered[layerIDX] = np.vstack((tau_SD_per_cell_H2layerFiltered[layerIDX],tau_SD_per_cell[H2layerFilter].reshape(-1,1)))
        gene_data_dense_H2layerFiltered[layerIDX] = np.vstack((gene_data_dense_H2layerFiltered[layerIDX],gene_data_dense[H2layerFilter,:]))

if tauRegression:
    geneProfilePresentCount = 0
    possiblePoolsCount = 0
    CCF_ML_Center = 227.53027784753124 #this is hard-coded CCF 'true' center, this comes from the tform_Tallen2CCF of the mean ML coordinates of bregma & lambda from five Cux mice, this should be replaced with a more robust method
    allTauCCF_Coords = np.load(os.path.join(tauPath,f'{lineSelection}_tauCCF.npy'))
    ### pool tau into a grid for bootstrapping the regression ###
    minML_CCF, maxML_CCF, minAP_CCF, maxAP_CCF = np.floor(np.min(allTauCCF_Coords[0,:])), np.ceil(np.max(allTauCCF_Coords[0,:])), np.floor(np.min(allTauCCF_Coords[1,:])), np.ceil(np.max(allTauCCF_Coords[1,:]))
    tauPoolSize = 6
    pooledTauCCF_coords = [np.empty((4,0)) for _ in range(numLayers)]
    pooledTauCCF_coords_noGene = [np.empty((2,0)) for _ in range(numLayers)]
    pooledPixelCount_v_CellCount = [np.empty((2,0)) for _ in range(numLayers)]
    pooledTau_cellAligned = [np.empty((1,0)) for _ in range(numLayers)]
    resampledGenes_aligned = [np.empty((total_genes,0)) for _ in range(numLayers)]
    genePoolSaturation = []
    for layerIDX in range(numLayers):
        print(f'Tau-Gene Alignment Pooling: {layerNames[layerIDX]}')
        for current_tau_ML_pool in np.arange(minML_CCF,np.ceil(CCF_ML_Center),tauPoolSize):
            current_ML_tau_pooling_IDXs = np.where(np.abs(np.abs(allTauCCF_Coords[0,:]-CCF_ML_Center)-np.abs(current_tau_ML_pool-CCF_ML_Center))<(tauPoolSize/2))[0]
            current_ML_cell_pooling_IDXs = np.where(np.abs(mlCCF_per_cell_H2layerFiltered[layerIDX].reshape(-1)-current_tau_ML_pool)<(tauPoolSize/2))[0]
            for current_tau_AP_pool in np.arange(minAP_CCF,maxAP_CCF,tauPoolSize):
                current_tau_pooling_IDXs = np.where(np.abs(allTauCCF_Coords[1,current_ML_tau_pooling_IDXs]-current_tau_AP_pool)<(tauPoolSize/2))
                current_cell_pooling_IDXs = np.where(np.abs(apCCF_per_cell_H2layerFiltered[layerIDX].reshape(-1)[current_ML_cell_pooling_IDXs]-current_tau_AP_pool)<(tauPoolSize/2))[0]
                pooledTaus = allTauCCF_Coords[2,current_ML_tau_pooling_IDXs[current_tau_pooling_IDXs]]
                if pooledTaus.size > 0:
                    #print(mlCCF_per_cell_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs])
                    possiblePoolsCount += 1
                    if current_cell_pooling_IDXs.shape[0] > 0:
                        geneProfilePresentCount += 1

                        pooledTauCCF_coords[layerIDX] = np.hstack((np.array((current_tau_ML_pool,current_tau_AP_pool,np.mean(pooledTaus),np.std(pooledTaus))).reshape(-1,1),pooledTauCCF_coords[layerIDX]))
                        
                        pooledTau_cellAligned[layerIDX] = np.hstack((pooledTaus.reshape(1,-1),pooledTau_cellAligned[layerIDX]))

                        gene_pool_data = gene_data_dense_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                        geneResamplingIDX = random.choices(np.arange(0,gene_pool_data.shape[0]), k=pooledTaus.shape[0])
                        resampledGenes_aligned[layerIDX] = np.hstack((gene_pool_data[geneResamplingIDX,:].reshape(total_genes,-1),resampledGenes_aligned[layerIDX]))

                        pooledPixelCount_v_CellCount[layerIDX] = np.hstack((np.array((pooledTaus.shape[0],gene_pool_data.shape[0])).reshape(2,-1),pooledPixelCount_v_CellCount[layerIDX]))

                        gene_pool_ML_CCF = mlCCF_per_cell_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs]].reshape(-1)
                        gene_pool_AP_CCF = apCCF_per_cell_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs]].reshape(-1)

                        #print(f'CCF ML:{current_tau_ML_pool}, CCF AP:{current_tau_AP_pool}, GeneX ML CCFs:{gene_pool_ML_CCF}, GeneX AP CCFs:{gene_pool_AP_CCF}')
                    else:
                        pooledTauCCF_coords_noGene[layerIDX] = np.hstack((np.array((current_tau_ML_pool,current_tau_AP_pool)).reshape(-1,1),pooledTauCCF_coords_noGene[layerIDX]))
        genePoolSaturation.append(geneProfilePresentCount/possiblePoolsCount)
    
    for layerIDX in range(numLayers):
        plt.figure(), plt.title(f'CCF Pooling:{tauPoolSize}, Fraction of Tau Pooled Points with at least one Gene Profile:{round(genePoolSaturation[layerIDX],3)}\n{layerNames[layerIDX]}')
        plt.scatter(pooledTauCCF_coords[layerIDX][0,:],pooledTauCCF_coords[layerIDX][1,:],color='green',s=0.5)
        plt.scatter(pooledTauCCF_coords_noGene[layerIDX][0,:],pooledTauCCF_coords_noGene[layerIDX][1,:],color='red',s=0.5)
        plt.xlabel('CCF ML'), plt.ylabel('CCF AP'), plt.axis('equal')

        plt.figure(), plt.xlabel('Pool Pixel Count'), plt.ylabel('Pool Cell Count'), plt.title(f'Pool Size:{tauPoolSize}, Pixel & Cell Counts by CCF Pool\n{layerNames[layerIDX]}')
        plt.scatter(pooledPixelCount_v_CellCount[layerIDX][0,:],pooledPixelCount_v_CellCount[layerIDX][1,:],color='black')
        plt.axis('equal')

plt.plot(np.mean(standard_scaler.fit_transform(np.asarray(gene_data_dense_H2layerFiltered[layerIDX][:,:]).T).T,axis=0))



expressionPercentiles = [np.zeros((5,len(geneNames))) for _ in range(numLayers)]
for layerIDX in range(numLayers):
    expressionPercentiles[layerIDX][:,:] = np.percentile(np.asarray(gene_data_dense_H2layerFiltered[layerIDX][:,:]),[0,25,50,75,100],axis=0)

plt.figure()
for per in range(5):
    plt.plot(expressionPercentiles[layerIDX][per,:])
plt.yscale('log')
plt.ylim(1e-1,)

#############################################################################################
### visualization of and calculation of high expression genes are combined here, separate ###
highExpressionGeneIDXs = [[] for _ in range(numLayers)]
expressionThresh = 0.1
for layerIDX in range(numLayers):
    layerMeanExpressions = np.mean(np.asarray(gene_data_dense_H2layerFiltered[layerIDX][:,:]),axis=0)
    highExpressionGeneIDXs[layerIDX] = (np.where(layerMeanExpressions > expressionThresh)[0]).astype(int)
    print(highExpressionGeneIDXs[layerIDX].shape[0])

    sortedExpressions = np.argsort(layerMeanExpressions)
    expressionCutoffIDX = np.argmin(np.abs(layerMeanExpressions[sortedExpressions]-expressionThresh))

    #lowExpressionGeneIDXs = np.where(layerMeanExpressions < 0.1)[0]
    #print(f'{layerNames[layerIDX]}: \n{np.array(geneNames)[lowExpressionGeneIDXs]}\n\n')
    fig, ax = plt.subplots(1,1,figsize=(15,10))
    ax.plot(layerMeanExpressions[sortedExpressions],color='black')
    ax.vlines(x=expressionCutoffIDX, ymin=0, ymax=np.max(layerMeanExpressions),color='black',alpha=0.5,linestyles='dashed')
    ax.hlines(y=expressionThresh, xmin=0, xmax=total_genes,color='black',alpha=0.5,linestyles='dashed')
    ax.set_xticks(np.arange(0, total_genes, 1))
    ax.set_xticklabels(np.array(geneNames)[sortedExpressions], rotation=90)
    ax.set_ylabel('Mean Gene Expression')
    ax.set_yscale('log')
    ax.set_title(f'{layerNames[layerIDX]}, Num Excluded Genes:{expressionCutoffIDX}, Mean Expression Cutoff:{expressionThresh}')
    plt.savefig(os.path.join(savePath,f'{lineSelection}_excludedExpressions_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')



spatialReconstruction = False
######################################################################################
### Standard Scaler transform the gene expression and tau data prior to regression ###
if spatialReconstruction:
    gene_data_dense_H2layerFiltered_standard = [np.zeros_like(gene_data_dense_H2layerFiltered[layerIDX]) for layerIDX in range(numLayers)]
    tau_per_cell_H2layerFiltered_standard = [np.zeros_like(tau_per_cell_H2layerFiltered[layerIDX]) for layerIDX in range(numLayers)]
    mlCCF_per_cell_H2layerFiltered_standard = [np.zeros_like(mlCCF_per_cell_H2layerFiltered[layerIDX]) for layerIDX in range(numLayers)]
    apCCF_per_cell_H2layerFiltered_standard = [np.zeros_like(apCCF_per_cell_H2layerFiltered[layerIDX]) for layerIDX in range(numLayers)]
    for layerIDX in range(numLayers):
        gene_data_dense_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(gene_data_dense_H2layerFiltered[layerIDX][:,:]))
        tau_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(tau_per_cell_H2layerFiltered[layerIDX][:,:]))
        mlCCF_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(mlCCF_per_cell_H2layerFiltered[layerIDX][:,:]))
        apCCF_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(apCCF_per_cell_H2layerFiltered[layerIDX][:,:]))
if tauRegression:
    pooledTau_cellAligned_standard = [np.zeros_like(pooledTau_cellAligned[layerIDX].T) for layerIDX in range(numLayers)]
    resampledGenes_aligned_H2layerFiltered_standard = [np.zeros_like(resampledGenes_aligned[layerIDX].T) for layerIDX in range(numLayers)]
    for layerIDX in range(numLayers):
        pooledTau_cellAligned_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(pooledTau_cellAligned[layerIDX][:,:]).T)
        resampledGenes_aligned_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(resampledGenes_aligned[layerIDX][:,:]).T)

print(np.mean(resampledGenes_aligned_H2layerFiltered_standard[0][:,:],axis=0)) #just to see that the means are zero after standardizing


if spatialReconstruction:
    pred_dim = 2
else:
    pred_dim = 1
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
GLMpredictDFFsorted = [[] for _ in range(numLayers)]
alphas = np.power(10, np.linspace(-4,0,num=18)) #np.linspace(0.0, 1.0, num=10) #np.power(10, np.linspace(-3, 7, num=30)) #This might be too high, taking too long in processing
lasso_weight = [np.zeros((alphas.shape[0],pred_dim,highExpressionGeneIDXs[layerIDX].shape[0])) for layerIDX in range(numLayers)]
alpha_R2 = np.zeros((numLayers,alphas.shape[0]))
bestAlpha = np.zeros((numLayers,n_splits))
bestR2 = np.zeros((numLayers,n_splits))
best_coef = [np.zeros((n_splits,pred_dim,highExpressionGeneIDXs[layerIDX].shape[0])) for layerIDX in range(numLayers)]
tauPredictions = [[[] for _ in range(n_splits)] for _ in range(numLayers)]
print(f'Starting Regressions: \n(spatial={spatialReconstruction}, tauRecon={tauRegression}, regionResamp={regionalResample})')
for layerIDX, layer in enumerate(layerIDs):
    print(f'Fitting: {layerNames[layerIDX]}')
    GLMpredictTau = []
    GLMpredictIDX = []
    for foldIDX,(train_index, test_index) in enumerate(kfold.split(tau_per_cell_H2layerFiltered_standard[layerIDX])):
        if spatialReconstruction:
            train_y = np.hstack((apCCF_per_cell_H2layerFiltered_standard[layerIDX][train_index,:],mlCCF_per_cell_H2layerFiltered_standard[layerIDX][train_index,:]))
            test_y = np.hstack((apCCF_per_cell_H2layerFiltered_standard[layerIDX][test_index,:],mlCCF_per_cell_H2layerFiltered_standard[layerIDX][test_index,:]))
        if tauRegression:
            train_y = pooledTau_cellAligned_standard[layerIDX][train_index,:]
            test_y = pooledTau_cellAligned_standard[layerIDX][test_index,:]
        else:
            train_y = tau_per_cell_H2layerFiltered_standard[layerIDX][train_index,:]
            test_y = tau_per_cell_H2layerFiltered_standard[layerIDX][test_index,:]
        #train_w = tau_SD_per_cell_H2layerFiltered[layerIDX][train_index,:]**-1
        #test_w = tau_SD_per_cell_H2layerFiltered[layerIDX][test_index,:]**-1
        if tauRegression:
            train_x = np.asarray(resampledGenes_aligned_H2layerFiltered_standard[layerIDX][train_index,:])
            test_x  = np.asarray(resampledGenes_aligned_H2layerFiltered_standard[layerIDX][test_index,:])
        else:
            train_x = np.asarray(gene_data_dense_H2layerFiltered_standard[layerIDX][train_index,:])
            test_x  = np.asarray(gene_data_dense_H2layerFiltered_standard[layerIDX][test_index,:])
        
        train_x = train_x[:,highExpressionGeneIDXs[layerIDX]]
        test_x = test_x[:,highExpressionGeneIDXs[layerIDX]]
        

        #GLM (Basic, Identity Linker) with L1 Regularization
        for alphaIDX,alpha in enumerate(alphas):
            lasso = Lasso(alpha=alpha)
            lasso.fit(train_x, train_y)
            pred_y = lasso.predict(test_x)
            R2_GLM_L1 = r2_score(test_y,pred_y)
            lasso_weight[layerIDX][alphaIDX,:,:] = lasso.coef_
            alpha_R2[layerIDX,alphaIDX] = R2_GLM_L1

        bestAlpha[layerIDX,foldIDX] = alphas[np.where(np.array(alpha_R2[layerIDX,:]) == np.max(np.array(alpha_R2[layerIDX,:])))[0][0]]
        bestR2[layerIDX,foldIDX] = np.max(np.array(alpha_R2[layerIDX,:])) #R1 of the alpha being selected for the model

        #now predict test fold using the best alpha
        lasso = Lasso(alpha=bestAlpha[layerIDX,foldIDX]) #choose the best alpha
        lasso.fit(train_x, train_y)
        best_coef[layerIDX][foldIDX,:,:] = lasso.coef_
        pred_y = lasso.predict(test_x)
        cell_region_IDX = (cell_region_H2layerFiltered[layerIDX][test_index,:]).reshape(-1)
        tauPredictions[layerIDX][foldIDX].append([test_y,pred_y,cell_region_IDX])
        GLMpredictTau.append(pred_y)
        GLMpredictIDX.append(test_index)

    GLMpredictTau_allFolds = np.concatenate(GLMpredictTau)
    GLMpredictIDX_allFolds = np.concatenate(GLMpredictIDX)
    
    sorted_indices = np.argsort(GLMpredictIDX_allFolds)
    GLMpredictDFFsorted[layerIDX] = GLMpredictTau_allFolds[sorted_indices]


###############################################################################################################################
### just visualize the tau reconstruction across folds, not important for calculating any variables used in later processes ###
if regionalResample:
    resampTitle = 'Regional Resampling'
else:
    resampTitle = 'Het Regional'
if spatialReconstruction:
    plottingTitles = ["A-P CCF","M-L CCF"]
    titleAppend = f'{resampTitle} Spatial Reconstruction'
else:
    plottingTitles = ["tau"]
    titleAppend = f'{resampTitle} {lineSelection} Tau Reconstruction'
for dim in range(pred_dim):
    currentPlottingTitle = plottingTitles[dim]
    for layerIDX,(layer,layerName) in enumerate(zip(layerIDs,layerNames)):
        fig, axes = plt.subplots(1,n_splits,figsize=(20,6))
        plt.suptitle(f'{titleAppend}, {layerName}')
        for foldIDX,ax in enumerate(axes):
            #cell_region_IDX = (cell_region_H2layerFiltered[layerIDX][test_index,:]).astype(int).reshape(-1)
            test_y = tauPredictions[layerIDX][foldIDX][0][0][:,dim]
            pred_y = tauPredictions[layerIDX][foldIDX][0][1][:,dim]
            cell_region_IDX = tauPredictions[layerIDX][foldIDX][0][2].astype(int)
            cell_region_colors = np.asarray(areaColors)[cell_region_IDX]

            for regionIDX,region in enumerate(structList):
                regionR2IDXs = np.where(cell_region_IDX == regionIDX)
                label = f'{region}' #R2: {round(r2_score(test_y[regionR2IDXs],pred_y[regionR2IDXs]),3)}'
                ax.scatter(test_y[regionR2IDXs],pred_y[regionR2IDXs],color=areaColors[regionIDX],label=label,s=1)

            #ax.scatter(test_y,pred_y,color=cell_region_colors,s=1)
            #ax.axis('equal')
            ax.set_title(f'Fold: {foldIDX}')
            ax.set_xlabel(f'True {currentPlottingTitle} (standardized)')
            ax.set_ylabel(f'Predicted {currentPlottingTitle} (standardized)')
            if foldIDX==0:
                ax.legend()
        plt.savefig(os.path.join(savePath,f'{titleAppend}_predicted{currentPlottingTitle}_{layerName}.pdf'),dpi=600,bbox_inches='tight')



if spatialReconstruction:
    for layerIDX,(layer,layerName) in enumerate(zip(layerIDs,layerNames)):
        crossFoldsTrue = np.concatenate([np.asarray(tauPredictions[layerIDX][foldIDX][0][0]) for foldIDX in range(len(tauPredictions[layerIDX]))],axis=0)
        crossFoldsPred = np.concatenate([np.asarray(tauPredictions[layerIDX][foldIDX][0][1]) for foldIDX in range(len(tauPredictions[layerIDX]))],axis=0)
        crossFoldsRegionIDX = np.concatenate([np.asarray(tauPredictions[layerIDX][foldIDX][0][2]) for foldIDX in range(len(tauPredictions[layerIDX]))],axis=0)
        apR2 = r2_score(crossFoldsTrue[:,0],crossFoldsPred[:,0])
        mlR2 = r2_score(crossFoldsTrue[:,1],crossFoldsPred[:,1])
        fig, axes = plt.subplots(1,2,figsize=(20,10))
        plt.suptitle(f'{layerName} Cross-Fold Spatial Reconstructions from Standardized Gene Expressions')
        axes[0].set_xlabel(f'True Standardized A-P CCF'), axes[0].set_ylabel('True Standardized M-L CCF')
        axes[1].set_xlabel(f'Predicted Standardized A-P CCF\n$R^2$={round(apR2,6)}'), axes[1].set_ylabel(f'Predicted Standardized M-L CCF\n$R^2$={round(mlR2,6)}')
        for regionIDX,region in enumerate(structList):
            regionR2IDXs = np.where(crossFoldsRegionIDX == regionIDX)
            axes[0].scatter(crossFoldsTrue[regionR2IDXs,0],crossFoldsTrue[regionR2IDXs,1],color=areaColors[regionIDX],s=0.25)
            axes[1].scatter(crossFoldsPred[regionR2IDXs,0],crossFoldsPred[regionR2IDXs,1],color=areaColors[regionIDX],s=0.25)
            axes[0].axis('equal')
            axes[1].axis('equal')
        plt.savefig(os.path.join(savePath,f'spatialReconstruction_{layerName}.pdf'),dpi=600,bbox_inches='tight')




mean_expression_scaler = StandardScaler()
#gene_subset = 105
with open(os.path.join(savePath,f'{titleAppend}_H3regression.txt'), "w") as file:
    file.write(f'{titleAppend}\n\n')
plotting = True
foldIDX = 0
mean_expression_standard = np.zeros_like(mean_expression)
for layerIDX,(layer,layerName) in enumerate(zip(layerIDs,layerNames)):
    mean_fold_coef = np.mean(best_coef[layerIDX][:,:,:],axis=0)
    sd_fold_coef = np.std(best_coef[layerIDX][:,:,:],axis=0)
    sorted_coef = np.argsort(mean_fold_coef)
    mean_expression_standard[layerIDX,:,:] = mean_expression_scaler.fit_transform(mean_expression[layerIDX,:,:])

    with open(os.path.join(savePath,f'{titleAppend}_H3regression.txt'), "a") as file:
        file.write(f'{layerName}, Best R2:{round(bestR2[layerIDX,foldIDX],3)} (at alpha={round(bestAlpha[layerIDX,foldIDX],10)})\n')
        for dim in range(pred_dim):
            if spatialReconstruction:
                file.write(f'####### {plottingTitles[dim]} #######\n')
            file.write(f'Highest + Predictors:{np.array(geneNames)[highExpressionGeneIDXs[layerIDX]][sorted_coef[dim,:]][-5:]}\n')
            file.write(f'Predictor Weights:{np.round(mean_fold_coef[dim,sorted_coef[dim,:]][-5:],3)}\n')
            file.write(f'Lowest - Predictors:{np.array(geneNames)[highExpressionGeneIDXs[layerIDX]][sorted_coef[dim,:]][:5]}\n')
            file.write(f'Predictor Weights:{np.round(mean_fold_coef[dim,sorted_coef[dim,:]][:5],3)}\n')
        file.write(f'\n')

    if plotting:
        fig, ax = plt.subplots(pred_dim,1,figsize=(10,10))
        plt.suptitle(f'{titleAppend}, {layerName}, best R2: {round(bestR2[layerIDX,foldIDX],3)} ($\\alpha$: {round(bestAlpha[layerIDX,foldIDX],6)}), fold:{foldIDX}\nRegional Resampling with Replacement:{regional_resampling}')
        for dim in range(pred_dim):
            ax[dim].set_title(f'{plottingTitles[dim]}')
            ax[dim].plot(alphas,lasso_weight[layerIDX][:,dim,:])
            ax[dim].vlines(x=bestAlpha[layerIDX,foldIDX],ymin=-10**5,ymax=10**5,color='black')
            ax[dim].set_xscale('log')
            ax[dim].set_ylim(np.min(lasso_weight[layerIDX][:,:]),np.max(lasso_weight[layerIDX][:,:]))
            ax[dim].set_xlabel(f'$\\alpha$')
            ax[dim].set_ylabel(f'$\\beta$')
        plt.savefig(os.path.join(savePath,f'{titleAppend}_GeneLassoWeightsAll_{layerName}.pdf'),dpi=600,bbox_inches='tight')
        plt.close()

        if spatialReconstruction:
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax.set_title(f'{titleAppend} from Gene Expression\nA-P vs M-L $\\beta$ Values\n{layerName}, $\\alpha$: {round(bestAlpha[layerIDX,foldIDX],6)}, R2: {round(bestR2[layerIDX,foldIDX],3)}, error:5-fold SD\nRegional Resampling with Replacement:{regional_resampling}')
            ax.scatter(mean_fold_coef[0,:],mean_fold_coef[1,:],color='black',s=0.5)
            plt.errorbar(mean_fold_coef[0,:], mean_fold_coef[1,:], xerr=sd_fold_coef[0,:], yerr=sd_fold_coef[1,:], fmt="o", color='black')
            for i, geneText in enumerate(np.asarray(geneNames)[highExpressionGeneIDXs[layerIDX]]):
                ax.annotate(geneText, (mean_fold_coef[0,i], mean_fold_coef[1,i]))
            ax.set_xlabel(f'A-P $\\beta$')
            ax.set_ylabel(f'M-L $\\beta$')
            plt.savefig(os.path.join(savePath,f'{titleAppend}_APvsML_GeneBetaWeights_{layerName}.pdf'),dpi=600,bbox_inches='tight')
            plt.close()

        fig, ax = plt.subplots(pred_dim,1,figsize=(16,13))
        plt.suptitle(f'{titleAppend} from Gene Expression\n{layerName}, $\\alpha$: {round(bestAlpha[layerIDX,foldIDX],6)}, R2: {round(bestR2[layerIDX,foldIDX],3)}, error:5-fold SD\nRegional Resampling with Replacement:{regional_resampling}')
        for dim in range(pred_dim):
            ax[dim].set_title(f'{plottingTitles[dim]}')
            #ax.scatter(np.arange(0,gene_data_dense.shape[1],1),mean_fold_coef[sorted_coef],color='black')
            ax[dim].errorbar(np.arange(0,highExpressionGeneIDXs[layerIDX].shape[0],1), mean_fold_coef[dim,sorted_coef[dim,:]], yerr=sd_fold_coef[dim,sorted_coef[dim,:]], fmt="o",color='black')
            ax[dim].hlines(y=0, xmin=0, xmax=highExpressionGeneIDXs[layerIDX].shape[0],color='black',alpha=0.5,linestyles='dashed')
            ax[dim].set_xticks(np.arange(0, highExpressionGeneIDXs[layerIDX].shape[0], 1))
            ax[dim].set_xticklabels(np.array(geneNames)[highExpressionGeneIDXs[layerIDX]][sorted_coef[dim,:]], rotation=90)
            ax[dim].set_ylabel(f'$\\beta$')
        plt.savefig(os.path.join(savePath,f'{titleAppend}_GeneLassoWeights_{layerName}.pdf'),dpi=600,bbox_inches='tight')
        plt.close()

        fig1, ax1 = plt.subplots(1,1,figsize=(15,8))
        #fig2, ax2 = plt.subplots(1,1,figsize=(15,8))
        for structIDX,structureOfInterest in enumerate(structList):
            ax1.plot(np.arange(0,highExpressionGeneIDXs[layerIDX].shape[0],1),mean_expression_standard[layerIDX,structIDX,highExpressionGeneIDXs[layerIDX]].reshape(-1,1)[sorted_coef[dim,:]],label=structureOfInterest,color=areaColors[structIDX])
            #ax2.plot(np.arange(0,total_genes,1),gaussian_filter(mean_expression_standard[structIDX,:].reshape(-1,1)[sorted_coef],sigma=2),label=structureOfInterest,color=areaColors[structIDX])
        ax1.set_xticks(np.arange(0, highExpressionGeneIDXs[layerIDX].shape[0], 1))
        ax1.set_xticklabels(np.asarray(geneNames)[highExpressionGeneIDXs[layerIDX]][sorted_coef[dim,:]], rotation=90)
        ax1.legend()
        ax1.set_ylabel('Standardized Gene Expression')
        ax1.set_title(f'{titleAppend}, {layerName}, $\\alpha$: {round(bestAlpha[layerIDX,foldIDX],6)}, R2: {round(bestR2[layerIDX,foldIDX],3)}')
        plt.savefig(os.path.join(savePath,f'{titleAppend}_regionalGeneExpressions_{layerName}.pdf'),dpi=600,bbox_inches='tight')
        plt.close()


##############################################
### Layer-specific expression correlations ###
fig, axes = plt.subplots(numLayers,numLayers,figsize=(15,15))
plt.suptitle('Cross-Layer Standardized Gene Expression Correlations')
for layerIDX0 in range(numLayers):
    for layerIDX1 in range(numLayers):
        L2L_r2 = r2_score(mean_expression_standard[layerIDX0,:,:].reshape(-1),mean_expression_standard[layerIDX1,:,:].reshape(-1))
        axes[layerIDX0,layerIDX1].set_title(f'$R^2$={round(L2L_r2,3)}')
        for currentRegion in range(structNum):
            axes[layerIDX0,layerIDX1].scatter(mean_expression_standard[layerIDX0,currentRegion,:],mean_expression_standard[layerIDX1,currentRegion,:],label=structListMerge[currentRegion],color=areaColors[currentRegion],s=0.25)
            if layerIDX1 == 0:
                axes[layerIDX0,layerIDX1].set_ylabel(f"{layerNames[layerIDX0]}\nGene Expressions")
            if layerIDX0 == numLayers-1:
                axes[layerIDX0,layerIDX1].set_xlabel(f"{layerNames[layerIDX1]}\nGene Expressions")
            if (layerIDX0 == numLayers-1) and (layerIDX1 == numLayers-1):
                axes[layerIDX0,layerIDX1].legend()
plt.savefig(os.path.join(savePath,f'{lineSelection}_crossLayerGeneExpressionCorrelations.pdf'),dpi=600,bbox_inches='tight')






##############################
### Plotting and Visualization
areaColors = ['#4dd2ff','#0066ff','#003cb3','#00ffff','#99ccff',          #VIS, blues
            '#ff0000','#ff704d',                                          #MO, reds
            '#33cc33','#339933','#8cd98c','#336600','#8cff1a','#00cc7a',  #SSp, greens
            '#a366ff','#8a00e6']                                          #RSP, purples

applyStructureMask = True

for groupSelector in layerIDs:
    current_group_name = grouping[groupSelector]

    maxCluster = sum(1 for s in H3_names if s.startswith(current_group_name))

    ##################################################################################################
    ### A-P & M-L pooling with D-V collapse, first step of boostrapping, these will be resampled later
    if calculatingPools:
        CCF_AP_Slices = np.arange(0,420,APpool)
        CCF_ML_Slices = np.arange(0,250,MLpool)
        pooledClusterCounts = np.zeros((structNum,CCF_AP_Slices.shape[0],CCF_ML_Slices.shape[0],maxCluster))
        unique_CCF_coordinates = [[] for _ in range(structNum)]

        for structIDX,structureOfInterest in enumerate(structList):
            structureTree = tree.get_structures_by_acronym([structureOfInterest])
            structureName = structureTree[0]['name']
            structureID = structureTree[0]['id']
            structure_mask = rsp.make_structure_mask([structureID])

            print(structIDX)

            for AP_IDX,CCF_AP in enumerate(CCF_AP_Slices):
                condition = ((fn_CCF[:,0]>=CCF_AP-(.5*APpool)) & (fn_CCF[:,0]<CCF_AP+(.5*APpool)))
                currentSliceIDX = np.where(condition)
                
                subClassSlice = fn_clustid[currentSliceIDX]
                CCF_Slice = fn_CCF[currentSliceIDX,:]

                CCF_AP = round(CCF_AP)
                currentMask = np.flipud(np.rot90(structure_mask[CCF_AP,:,:]))
                
                for ML_IDX,CCF_ML in enumerate(CCF_ML_Slices):
                    CCF_ML = round(CCF_ML)
                    
                    for H3type in np.arange(1,maxCluster+1,1):
                        clusterName = current_group_name+str(H3type)
                        
                        groupIDX = np.where(subClassSlice == clusterName)
                        for current_groupIDX in groupIDX[0]:
                            CCF_1 = CCF_Slice[0,current_groupIDX,1]
                            CCF_2 = CCF_Slice[0,current_groupIDX,2]
                            if applyStructureMask:
                                if (currentMask[round(CCF_2),round(CCF_1)] == 1) & ((CCF_2>=(CCF_ML-(.5*MLpool))) & (CCF_2<(CCF_ML+(.5*MLpool)))):
                                    pooledClusterCounts[structIDX,AP_IDX,ML_IDX,H3type-1] += 1
                                    unique_CCF_coordinates[structIDX].append((AP_IDX,ML_IDX))

        unique_CCF_coordinates_array = np.array([np.array(coorList) for coorList in unique_CCF_coordinates])
        np.save(os.path.join(projectPath,current_group_name+'pooledClusterCounts.npy'),pooledClusterCounts)
        np.save(os.path.join(projectPath,current_group_name+'unique_CCF_coordinates_array.npy'),unique_CCF_coordinates_array)


if not calculatingPools:
    for lineFilterIDX in np.arange(0,lineFilter.shape[0],1):
        currentSubject = lineFilter['subject_fullname'][lineFilterIDX]
        currentDate = lineFilter['session_date'][lineFilterIDX]
        currentSession = lineFilter['session_number'][lineFilterIDX]
        currentMouse = currentSubject +'_'+ str(currentDate) +'_'+ str(currentSession)
        layerIDX = -1
        layer_reconstruction_rSquared = np.zeros((len(layerNames),bootstrapIterations))
        for groupSelector,titleName in zip(layerIDs,layerNames):
            layerIDX += 1
            current_group_name = grouping[groupSelector]

            maxCluster = sum(1 for s in H3_names if s.startswith(current_group_name))

            pooledClusterCounts = np.load(os.path.join(projectPath,current_group_name+'pooledClusterCounts.npy'))
            unique_CCF_coordinates_array = np.load(os.path.join(projectPath,current_group_name+'unique_CCF_coordinates_array.npy'),allow_pickle=True)

            unique_CCF_coordinates = [[] for _ in range(structNum)]
            for structIDX,structCoor in enumerate(unique_CCF_coordinates_array):
                for currentCoor in structCoor:
                    unique_CCF_coordinates[structIDX].append(tuple(currentCoor))

            ############################################################
            ### Bootstrapping, resample the counts above with replaceent
            BootstrapH3means = [[] for _ in range(structNum)]
            BootstrapH3CI = [[] for _ in range(structNum)]
            pooled_coordinates = [[] for _ in range(structNum)]
            selectedPools = [np.zeros((len(set(unique_CCF_coordinates[structIDX])),maxCluster)) for structIDX in range(structNum)]
            savedShuffledResampledMeans = np.zeros((structNum,maxCluster,bootstrapIterations))
            for structIDX in range(structNum):
                pooled_coordinates[structIDX] = list(set(unique_CCF_coordinates[structIDX]))
                for pooledIDX,pooledCoord in enumerate(pooled_coordinates[structIDX]):
                    #plt.scatter(pooledCoord[0],pooledCoord[1])
                    selectedPools[structIDX][pooledIDX,:] = pooledClusterCounts[structIDX,pooledCoord[0],pooledCoord[1],:]
                for clustIDX in range(maxCluster):
                    dataXXX = selectedPools[structIDX][:,clustIDX]
                    meansXXX = sorted(mean(choices(dataXXX, k=len(dataXXX))) for i in range(bootstrapIterations))
                    BootstrapH3means[structIDX].append(mean(dataXXX))
                    BootstrapCI = ((mean(dataXXX)-meansXXX[round(bootstrapIterations*.025)]) + (meansXXX[round(bootstrapIterations*.975)]-mean(dataXXX))) / 2
                    BootstrapH3CI[structIDX].append(BootstrapCI)
                    random.shuffle(meansXXX)
                    savedShuffledResampledMeans[structIDX,clustIDX,:] = np.array(meansXXX)


            if doPlots:
                fig,axes=plt.subplots(1,1,figsize=(8,4))
                ax=axes
                H3IDX = 5
                for area in range(pooledClusterCounts.shape[0]):
                    for AP_IDX in range(pooledClusterCounts.shape[1]):
                        for ML_IDX in range(pooledClusterCounts.shape[2]):
                            if pooledClusterCounts[area,AP_IDX,ML_IDX,H3IDX] != 0:
                                ax.scatter(AP_IDX,ML_IDX,color=areaColors[area],s=.5)

                ax.set_xlabel('CCF A-P Axis')
                ax.set_ylabel('CCF L-M Axis')

                all_patches = []
                for colorScat,labelScat in zip(areaColors,structList):
                    current_patch = mpatches.Patch(color=colorScat,label=labelScat)
                    all_patches.append(current_patch)

                ax.legend(handles=all_patches,bbox_to_anchor=(1.04,.5),loc='center left')
                plt.title(titleName+' CCF DV Axis Collapse, AP & ML Pooling of '+str(APpool)+' CCF, H3id='+str(H3IDX))
                if savePlots:
                    plt.savefig(os.path.join(projectPath,titleName+'_CCF_Bootstrap_Viz_POOL'+str(APpool)+'_H3_'+str(H3IDX)+'.pdf'),bbox_inches="tight")



                fig,axes=plt.subplots(structNum,1,figsize=(5,13))
                for axIDX,ax in enumerate(axes):
                    ax.bar(np.arange(0,maxCluster,1),BootstrapH3means[axIDX][:maxCluster],color=areaColors[axIDX])
                    ax.errorbar(np.arange(0,maxCluster,1),BootstrapH3means[axIDX][:maxCluster],BootstrapH3CI[axIDX][:maxCluster],fmt=' ',color='black')
                    ax.set_ylabel(structList[axIDX],fontsize=8)
                ax.set_xlabel('H3 Index')
                axes[0].set_title(titleName+' DV Axis Collapse, AP & ML CCF Pooling='+str(APpool)+',\nMean H3 Counts with Bootstrapped 95%CI\n(1000 iterations)')
                if savePlots:
                    plt.savefig(os.path.join(projectPath,titleName+'_Bootstrapped_DistByRegion_pool'+str(APpool)+'.pdf'))


            bootstrappedMeansArray = np.array([np.array(xi) for xi in BootstrapH3means])
            np.save(os.path.join(projectPath,titleName+'_bootstrappedMeansArray.npy'),bootstrappedMeansArray)


            #######
            ### Tau
            tau_area_processed = np.load(os.path.join(tauPath,currentMouse+'_tau_area_processed.npy'))
            areaLabelsSet = np.load(os.path.join(tauPath,currentMouse+'_areaLabelsSet.npy'))
            
            currentAreas = list(areaLabelsSet)
            areaSortingIDX = [currentAreas.index(struct) for struct in list(structList) if struct in currentAreas]
            #areaSortingIDX = [5,6,0,4,7,2,11,10,13,1,18,15,16,3,9] #resolve difference between timescale and H3 count arrays
            tau_area_sort = tau_area_processed[areaSortingIDX,:]

            ###
            scaler_y = StandardScaler()
            trueTau = (tau_area_sort[:,0]/10).reshape(-1, 1)
            for iterationIDX in range(bootstrapIterations):
                currentMeans = savedShuffledResampledMeans[:,:,iterationIDX]

                zscoredBootstrappedMeans = scipy.stats.zscore(currentMeans,axis=0)
                y_standardized = scaler_y.fit_transform(trueTau)
                clusterWeights = np.linalg.lstsq(zscoredBootstrappedMeans,y_standardized)
                tau_area_reconstructed = zscoredBootstrappedMeans @ clusterWeights[0]
                scaledTauReconstruction = scaler_y.inverse_transform(tau_area_reconstructed)

                reconstructionFit = stats.linregress(scaledTauReconstruction.transpose(),trueTau.transpose())
                layer_reconstruction_rSquared[layerIDX,iterationIDX] = reconstructionFit.rvalue ** 2
            ###

            y_standardized = scaler_y.fit_transform((tau_area_sort[:,0]/10).reshape(-1, 1))

            zscoredBootstrappedMeans = scipy.stats.zscore(bootstrappedMeansArray[:,:],axis=0)
            clusterWeights = np.linalg.lstsq(zscoredBootstrappedMeans,y_standardized)
            tau_area_reconstructed = zscoredBootstrappedMeans @ clusterWeights[0]
            scaledTauReconstruction = scaler_y.inverse_transform(tau_area_reconstructed)

            all_patches = []
            for colorScat,labelScat in zip(areaColors,areaLabelsSet[areaSortingIDX]):
                current_patch = mpatches.Patch(color=colorScat,label=labelScat)
                all_patches.append(current_patch)

            fig,axes=plt.subplots(1,1,figsize=(4,4))
            ax=axes
            ax.errorbar(tau_area_sort[:,0]/10,scaledTauReconstruction, xerr=tau_area_sort[:,1]/10, fmt=" ",color='black')
            ax.scatter(tau_area_sort[:,0]/10,scaledTauReconstruction,c=areaColors)
            ax.set_xlabel('True Tau (sec \u00B1 SD)')
            ax.set_ylabel('Reconstructed Tau (s)')
            ax.legend(handles=all_patches,bbox_to_anchor=(1.04,.5),loc='center left')
            plt.title('One Bootstrap Iteration,\nTrue vs Reconstructed Tau (from Z-Scored '+titleName+' H3 distribution)\n'+currentMouse)
            if savePlots:
                plt.savefig(os.path.join(projectPath,currentMouse+'_'+titleName+'_TauReconstructionByArea.pdf'),dpi=600,bbox_inches="tight")

        numLayer = len(layerNames)
        layer_rSquare_mean = np.zeros(numLayer)
        Bootstrap_rSquare_CI = np.zeros(numLayer)
        for layerIDX in range(numLayer):
            sorted_layer_rSquared = sorted(layer_reconstruction_rSquared[layerIDX,:])
            layer_rSquare_mean[layerIDX] = mean(sorted_layer_rSquared)
            Bootstrap_rSquare_CI[layerIDX] = ((mean(sorted_layer_rSquared)-sorted_layer_rSquared[round(bootstrapIterations*.025)]) + (sorted_layer_rSquared[round(bootstrapIterations*.975)]-mean(sorted_layer_rSquared))) / 2

        fig,axes=plt.subplots(1,1,figsize=(5,5))
        ax = axes
        ax.bar(np.arange(0,numLayer,1),layer_rSquare_mean,color="grey")
        ax.errorbar(np.arange(0,numLayer,1),layer_rSquare_mean,Bootstrap_rSquare_CI,fmt=' ',color='black')
        ax.set_xlabel('Layer')
        ax.set_ylabel('R Squared of Tau Reconstruction')
        ax.set_xticks(np.arange(0,numLayer,1),layerNames)
        plt.title(str(bootstrapIterations)+' Bootstrap Iterations,\nTau Reconstruction By Cortical Layer, Bootstrapped 95% CI,\n Lagged Velocity GLM, SNRthresh=200\n'+currentMouse)
        if savePlots:
            plt.savefig(os.path.join(projectPath,currentMouse+'_RsquaredLayerReconstruction.pdf'),bbox_inches="tight")


