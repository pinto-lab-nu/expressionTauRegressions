import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from pathlib import Path
from statistics import fmean as mean
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from sklearn.linear_model import LinearRegression
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
import sys
import os
import h5py
import numpy as np
import random
from random import choices
import pandas as pd
from statsmodels.regression.linear_model import WLS
from sklearn.preprocessing import OneHotEncoder
from packages.regressionUtils import *
from packages.dataloading import *
from collections import Counter
import datetime


time_start = datetime.datetime.now()

standard_scaler = StandardScaler()
hotencoder = OneHotEncoder(sparse_output=False)


##################################
### Script Parameters and Settings
lineSelection = 'Cux2-Ai96'
#lineSelection = 'Rpb4-Ai96'
geneLimit = -1 #for testing purposes only, remove later
loadData = True
lineSelection, my_os, tauPath, savePath, download_base = pathSetter(lineSelection)
plotting = True
numPrecision, alphaPrecision = 3, 5 #just for display in plotting and regression text files


structListMerge = np.array(['MOp','MOs','VISa','VISp','VISam','VISpm','SS','RSP'])
structList = structListMerge


structNum = structList.shape[0]
#applyLayerSpecificityFilter = False #ensure that CCM coordinates are contained within a layer specified in layerAppend
#layerAppend = '2/3'
#groupSelector = 12  #12 -> IT_7  -> L2/3 IT
                    #4  -> IT_11 -> L4/5 IT
                    #14 -> IT_9  -> L5 IT
                    #11 -> IT_6  -> L6 IT


#if applyLayerSpecificityFilter:
#    structList = [x+layerAppend for x in structList]


areaColors = ['#ff0000','#ff704d',                      #MO, reds
            '#4dd2ff','#0066ff','#003cb3','#00ffff',    #VIS, blues
            '#33cc33',                                  #SSp, greens
            '#a366ff']                                  #RSP, purples



################################
### CCF Reference Space Creation
#see link for CCF example scripts from the allen: allensdk.readthedocs.io/en/latest/_static/examples/nb/reference_space.html
tree = {}
rsp = {}
for resolution in [10,25,100]:
    output_dir = os.path.join(savePath,'Data',f'nrrd{resolution}')
    reference_space_key = os.path.join('annotation','ccf_2017')
    rspc = ReferenceSpaceCache(resolution, 'annotation/ccf_2017', manifest=Path(output_dir) / 'manifest.json') #reference_space_key replaced by 'annotation/ccf_2017'
    # ID 1 is the adult mouse structure graph
    tree[f'{resolution}'] = rspc.get_structure_tree(structure_graph_id=1)

    annotation, meta = rspc.get_annotation_volume() #in browser navigate to the .nrrd file and download manually, not working automatically for some reason
    # The file should be moved to the reference space key directory, only needs to be done once
    os.listdir(Path(output_dir) / reference_space_key)
    rsp[f'{resolution}'] = rspc.get_reference_space()


if loadData:
    gene_data_dense, pilotGeneNames, fn_clustid, fn_CCF = pilotLoader(savePath)
    merfish_CCF_Genes, allMerfishGeneNames = merfishLoader(savePath,download_base,pilotGeneNames,geneLimit)

time_load_data = datetime.datetime.now()
print(f'Time to load data: {time_load_data - time_start}')

#standardMerfish_CCF_Genes = standard_scaler.fit_transform(merfish_CCF_Genes)
#standardMerfish_CCF_Genes = pd.DataFrame(standardMerfish_CCF_Genes, columns=merfish_CCF_Genes.columns)

enrichedGeneNames = list(merfish_CCF_Genes.drop(columns=['x_ccf','y_ccf','z_ccf']).columns)
total_genes = gene_data_dense.shape[1]

raw_merfish_genes = np.array(merfish_CCF_Genes.drop(columns=['x_ccf','y_ccf','z_ccf']))
numMerfishCells = merfish_CCF_Genes.shape[0]
raw_merfish_CCF = np.array(merfish_CCF_Genes.loc[:,['x_ccf','y_ccf','z_ccf']])
del merfish_CCF_Genes

print(f'Memory usage of raw_merfish_genes: {round(sys.getsizeof(raw_merfish_genes)/1024/1024,1)}GB')
print(f'Memory usage of raw_merfish_CCF: {round(sys.getsizeof(raw_merfish_CCF)/1024/1024,1)}GB') #even though there are only three coordinate axes, the precision of these values is higher than the gene expressions (uses up more memory than might be expected)



# with open(os.path.join(savePath,f'Chen_merfishImputed_geneOverlap.txt'), "w") as file:
#     file.write('Gene Overlap (Chen & Merfish-Imputed Datasets):\n\n')

#     for currentGene in pilotGeneNames:
#         geneIDX_Text = f'Merfish Imputed IDX of {currentGene}: {np.where(np.array(allMerfishGeneNames)==currentGene)[0]}'
#         print(geneIDX_Text)
#         file.write(geneIDX_Text+'\n')



#fn_clustid = np.load(os.path.join(projectPath,'fn_clustid.npy'))

H2_all = [s[:-1] for s in fn_clustid]
H3_names = set(fn_clustid)
H2_names = []
for curstate in H3_names:
    if (curstate != 'non_Exc') and (curstate != 'qc-filtered'):
        H2_names.append(curstate[:-1])
    else:
        H2_names.append(curstate)
grouping = sorted(list(set(H2_names)))

#For layers, it's important that the first layer indexed is L2/3, since a Cux2 expression filter is applied later
merfishLayerNames = ['L2_3 IT_ET'] # 'L4_5 IT_ET', 'L5 IT_ET', 'L6 IT_ET'] #['CTX IT, ET']
pilotLayerNames  =  ['L2_3 IT',   'L4_5 IT',  'L5 IT',    'L6 IT',    'L5 ET']
layerIDs    =       [12,          4,          14,         11,         17]
#numLayers = len(layerIDs)


# ### Testing ###
# for structIDX,structureOfInterest in enumerate(structList):
#     if structIDX > -1:
#         structureOfInterestAppend = structureOfInterest + 'agl6a'
#         print(tree.get_structures_by_acronym([structureOfInterestAppend]))


def plotMask(structureOfInterest,structure_mask,savePath,resolution):
    plt.figure()
    plt.title(f'{structureOfInterest}')
    plt.imshow(np.mean(structure_mask[:,:,:],axis=1)) #restrict to ~40 voxels along axis 1 to get dorsal cortex (for 25 rez), this is just for viz
    plt.savefig(os.path.join(savePath,'Masks',f'{structureOfInterest}_{resolution}.pdf'),dpi=600,bbox_inches='tight')
    plt.close()

def CellRegion(cell_region,resolution,structure_mask,CCFvalues,CCFindexOrder,CCFmultiplier,structIDX):
    for cell in range(cell_region[resolution].shape[0]):
        currentMask = structure_mask[round(CCFvalues[cell,CCFindexOrder[0]]*CCFmultiplier),round(CCFvalues[cell,CCFindexOrder[1]]*CCFmultiplier),round(CCFvalues[cell,CCFindexOrder[2]]*CCFmultiplier)]
        if currentMask > 0:
            cell_region[resolution][cell] = structIDX

    return cell_region

if not os.path.exists(os.path.join(savePath,'Masks')):
    os.makedirs(os.path.join(savePath,'Masks'))

cell_region = {}
cell_region['10'] = (np.ones(numMerfishCells)*-1).astype(int)
cell_region['25'] = (np.ones(fn_CCF.shape[0])*-1).astype(int)
cell_layer = [(np.ones(numMerfishCells)*-1).astype(int)]

for resolution,datasetName in zip(['10','25'],['Merfish-Imputed','Pilot']):
    if resolution == '10':
        CCFvalues = raw_merfish_CCF
        CCFmultiplier = 100
        CCFindexOrder = [0,1,2]
    if resolution == '25':
        CCFvalues = fn_CCF
        CCFmultiplier = 1
        CCFindexOrder = [0,1,2]

    maskDim0,maskDim1,maskDim2 = rsp[f'{resolution}'].make_structure_mask([1]).shape
    
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    for ccfIDX,axes in enumerate(ax):
        axes.hist(CCFvalues[:,ccfIDX], color='black', bins=500)
    plt.suptitle(f'CCF Distributions (resolution={resolution})')
    plt.savefig(os.path.join(savePath,'Masks',f'CCFdistributions_resolution{resolution}.pdf'), dpi=600, bbox_inches='tight')
    plt.close()

    #fig, ax = plt.subplots(1,1,figsize=(8,8))
    if resolution == '10':
        for structIDX,structureOfInterest in enumerate(structList):
            print(f'Making {structureOfInterest} CCF mask (resolution={resolution})...')

            for layerAppend in ['2/3']:
                
                if structureOfInterest == 'RSP':
                    structure_mask = np.zeros((maskDim0,maskDim1,maskDim2))
                    for subRSP in ['v','d','agl']:
                        structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest+subRSP+layerAppend])
                        structureName = structureTree[0]['name']
                        structureID = structureTree[0]['id']
                        structure_mask += rsp[f'{resolution}'].make_structure_mask([structureID])
                
                if structureOfInterest == 'SS':
                    structure_mask = np.zeros((maskDim0,maskDim1,maskDim2))
                    for subSS in ['p-n','p-bfd','p-ll','p-m','p-ul','p-tr','p-un','s']:
                        structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest+subSS+layerAppend])
                        structureName = structureTree[0]['name']
                        structureID = structureTree[0]['id']
                        structure_mask += rsp[f'{resolution}'].make_structure_mask([structureID])

                if not(structureOfInterest == 'SS') and not(structureOfInterest == 'RSP'):
                    structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest+layerAppend])
                    structureName = structureTree[0]['name']
                    structureID = structureTree[0]['id']
                    structure_mask = rsp[f'{resolution}'].make_structure_mask([structureID])

                plotMask(structureOfInterest,structure_mask,savePath,resolution)
                cell_region = CellRegion(cell_region,resolution,structure_mask,CCFvalues,CCFindexOrder,CCFmultiplier,structIDX)
    
    if resolution == '25':
        for structIDX,structureOfInterest in enumerate(structList):
            print(f'Making {structureOfInterest} CCF mask (resolution={resolution})...')

            structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest])
            structureName = structureTree[0]['name']
            structureID = structureTree[0]['id']
            structure_mask = rsp[f'{resolution}'].make_structure_mask([structureID])

            plotMask(structureOfInterest,structure_mask,savePath,resolution)
            cell_region = CellRegion(cell_region,resolution,structure_mask,CCFvalues,CCFindexOrder,CCFmultiplier,structIDX)

    regionalCounts = Counter(cell_region[resolution])
    print(f'Regional Cell Counts, {datasetName}:')
    for structIDX in range(len(structList)):
        print(f'{structList[structIDX]}:{regionalCounts[structIDX]}')
    print('\n')


# ####################################
# ### Display Specific Gene Expression
# geneNameOI = 'Grik1'
# geneOI = np.where(gene_names==geneNameOI)[0][0]
# geneOI_IDXs = np.where(gene_data_dense[:,geneOI] > 0)[0]
# for view in [[0,1],[1,2],[0,2]]:
#     plt.figure()
#     plt.scatter(fn_CCF[geneOI_IDXs,view[0]],fn_CCF[geneOI_IDXs,view[1]],color='black',s=1,alpha=(gene_data_dense[geneOI_IDXs,geneOI]/np.max(gene_data_dense[geneOI_IDXs,geneOI])))


#cell_region = cell_region.astype(int)


# index_map = {value: index for index, value in enumerate(areaLabelsSet)}
# mapping = []
# for value in structList:
#     if value in index_map:
#         mapping.append(index_map[value])
#     else:
#         # Handle the case where the string in arr2 is not found in arr1
#         mapping.append(None)

# tau_region_mapped = tau_area_processed[:,0,0][mapping]
# tau_SD_region_mapped = tau_area_processed[:,1,0][mapping]
# tau_per_cell = tau_region_mapped[cell_region]
# tau_SD_per_cell = tau_SD_region_mapped[cell_region]

# pop_tau_params = np.load(os.path.join(tauPath,f'crossMouse_regionTauFits.npy'))
# fullTauDist = np.load(os.path.join(tauPath,f'{lineSelection}_crossMouse_regionalFullTauDist.npy'), allow_pickle=True)

# tau_per_cell = np.zeros(cell_region.shape[0])
# for regionIDX in range(len(structList)):
#     regionCells = np.where(cell_region == regionIDX)[0]
#     pop_gamma_a,pop_gamma_loc,pop_gamma_scale,pop_gaussian_mu,pop_gaussian_sigma = pop_tau_params[regionIDX,:]
#     #regionTausFromGammaDist = gamma.rvs(pop_gamma_a, pop_gamma_loc, pop_gamma_scale, size=regionCells.shape[0])
#     regionTausFromFullDist = np.array(random.choices(fullTauDist[regionIDX], k=regionCells.shape[0])).reshape(-1)
#     tau_per_cell[regionCells] = regionTausFromFullDist #regionTausFromGammaDist


# occurrences = Counter(cell_region)



regionalResample = False #resample each cortical region such that it's represented equally, in practice this tends to over-represent smaller regions
regional_resampling = 3000
cell_region_H2layerFiltered, gene_data_dense_H2layerFiltered, mlCCF_per_cell_H2layerFiltered, apCCF_per_cell_H2layerFiltered, H3_per_cell_H2layerFiltered, mean_expression = {},{},{},{},{},{}
for layerNames,numLayers,resolution in zip([pilotLayerNames,merfishLayerNames],[len(pilotLayerNames),len(merfishLayerNames)],['25','10']):
    if resolution == '10':
        CCFvalues = raw_merfish_CCF
        gene_data = raw_merfish_genes
    if resolution == '25':
        CCFvalues = fn_CCF
        gene_data = gene_data_dense
    
    #tau_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
    cell_region_H2layerFiltered[resolution] = [np.empty((0,1)).astype(int) for _ in range(numLayers)]
    #tau_SD_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
    gene_data_dense_H2layerFiltered[resolution] = [np.empty((0,gene_data.shape[1])) for _ in range(numLayers)]
    mlCCF_per_cell_H2layerFiltered[resolution] = [np.empty((0,1)) for _ in range(numLayers)]
    apCCF_per_cell_H2layerFiltered[resolution] = [np.empty((0,1)) for _ in range(numLayers)]
    H3_per_cell_H2layerFiltered[resolution] = [np.empty((0,1)) for _ in range(numLayers)]
    mean_expression[resolution] = np.zeros((numLayers,len(structList),gene_data.shape[1]))
    #sigma_expression = np.zeros((numLayers,len(structList),gene_data_dense.shape[1]))

    for layerIDX,(layer,layerName) in enumerate(zip(layerIDs,layerNames)):

        if resolution == '25':
            layerIDXs = set([i for i, s in enumerate(H2_all) if s == grouping[layer]])
        else:
            layerIDXs = set(np.where(cell_region['10'] > -1)[0])
        
        if layerIDX == 0:
            if resolution == '25':
                cux2columnIDX = np.where(np.array(pilotGeneNames)=='Cux2')[0][0]
            if resolution == '10':
                cux2columnIDX = np.where(np.array(enrichedGeneNames)=='Cux2')[0][0]
            cux2IDXs = set(np.where(gene_data[:,cux2columnIDX]>0)[0]) #for layer 2/3 filter out cells not expressing Cux2 (helps to align population to the functional dataset)

        for structIDX,structureOfInterest in enumerate(structList):
            regionIDXs = set(np.where(cell_region[resolution] == structIDX)[0])
            
            H2layerFilter = layerIDXs&regionIDXs
            if layerIDX == 0:
                H2layerFilter = H2layerFilter&cux2IDXs
            H2layerFilter = list(H2layerFilter)
        
            print(f'layer:{layerName}, {structureOfInterest}, count: {len(H2layerFilter)}')
            
            if resolution == '25':
                print(np.unique(fn_clustid[H2layerFilter]))
                H3_values = np.array([int(item[-1]) for item in fn_clustid[H2layerFilter]])
                H3_per_cell_H2layerFiltered[resolution][layerIDX] = np.vstack((H3_per_cell_H2layerFiltered[resolution][layerIDX],H3_values.reshape(-1,1)))

            mean_expression[resolution][layerIDX,structIDX,:] = np.mean(gene_data[H2layerFilter,:],0)
            #sigma_expression[layerIDX,structIDX,:] = np.std(gene_data_dense[H2layerFilter,:],0)

            # if regionalResample: #equal representation to each cortical region
            #     if len(H2layerFilter) > 0: #resample with replacement
            #         H2layerFilter = random.choices(H2layerFilter, k=regional_resampling)
            
            mlCCF_per_cell_H2layerFiltered[resolution][layerIDX] = np.vstack((mlCCF_per_cell_H2layerFiltered[resolution][layerIDX],CCFvalues[H2layerFilter,2].reshape(-1,1)))
            apCCF_per_cell_H2layerFiltered[resolution][layerIDX] = np.vstack((apCCF_per_cell_H2layerFiltered[resolution][layerIDX],CCFvalues[H2layerFilter,0].reshape(-1,1)))
            #tau_per_cell_H2layerFiltered[layerIDX] = np.vstack((tau_per_cell_H2layerFiltered[layerIDX],tau_per_cell[H2layerFilter].reshape(-1,1)))
            cell_region_H2layerFiltered[resolution][layerIDX] = np.vstack((cell_region_H2layerFiltered[resolution][layerIDX],cell_region[resolution][H2layerFilter].reshape(-1,1)))
            #tau_SD_per_cell_H2layerFiltered[layerIDX] = np.vstack((tau_SD_per_cell_H2layerFiltered[layerIDX],tau_SD_per_cell[H2layerFilter].reshape(-1,1)))
            gene_data_dense_H2layerFiltered[resolution][layerIDX] = np.vstack((gene_data_dense_H2layerFiltered[resolution][layerIDX],gene_data[H2layerFilter,:]))



CCF_ML_Center = 227.53027784753124 #this is hard-coded CCF 'true' center (in CCF 25 resolution), this comes from the tform_Tallen2CCF of the mean ML coordinates of bregma & lambda from five Cux mice, this should be replaced with a more robust method!!!
CCF_ML_Center_mm = CCF_ML_Center * 0.025
# if resolution == '10':
#     apCCF_per_cell_H2layerFiltered[resolution][layerIDX] = apCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1) * ((10/25)*100)
#     mlCCF_per_cell_H2layerFiltered[resolution][layerIDX] = np.abs((mlCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1) * ((10/25)*100)) - CCF_ML_Center)

for layerIDX,_ in enumerate(pilotLayerNames):
    apCCF_per_cell_H2layerFiltered['25'][layerIDX] = apCCF_per_cell_H2layerFiltered['25'][layerIDX] * 0.025
    mlCCF_per_cell_H2layerFiltered['25'][layerIDX] = mlCCF_per_cell_H2layerFiltered['25'][layerIDX] * 0.025

for layerIDX,_ in enumerate(merfishLayerNames):
    mlCCF_per_cell_H2layerFiltered['10'][layerIDX] = -1 * np.abs(mlCCF_per_cell_H2layerFiltered['10'][layerIDX] - CCF_ML_Center_mm) + CCF_ML_Center_mm #must collapse resolution 10um CCF across midline since it is a bilateral coordinate system

#############################################################################################
### visualization of and calculation of high expression genes are combined here, separate ###
allTauCCF_Coords = np.load(os.path.join(tauPath,f'{lineSelection}_tauCCF.npy'))
allTauCCF_Coords[0,:] *= 0.025 #convert 25um resolution functional-registered coordinates to mm
allTauCCF_Coords[1,:] *= 0.025 #convert "..."

#meanExpressionThreshArrayFull = [0] #[0.4,0.2,0.1,0]
#meanH3ThreshArrayFull = [0] #[0.1,0.05,0.025,0]
tauPoolSizeArrayFull = [2,4,8]
if my_os == 'Linux':
    #meanExpressionThreshArray = [meanExpressionThreshArrayFull[int(sys.argv[1])]] #batch job will distribute parameter instances among jobs run in parallel
    #meanH3ThreshArray = [meanH3ThreshArrayFull[int(sys.argv[1])]]                 #same for these regressions, just with a different parameter range
    tauPoolSizeArray = [tauPoolSizeArrayFull[int(sys.argv[1])]]
if my_os == 'Windows':
    #meanExpressionThreshArray = meanExpressionThreshArrayFull
    #meanH3ThreshArray = meanH3ThreshArrayFull
    tauPoolSizeArray = tauPoolSizeArrayFull

meanExpressionThresh,meanH3Thresh = 0,0

for layerNames,numLayers,resolution,datasetName in zip([merfishLayerNames,pilotLayerNames],[len(merfishLayerNames),len(pilotLayerNames)],['10','25'],['Merfish-Imputed','Pilot']):

    #for meanExpressionThresh,meanH3Thresh in zip(meanExpressionThreshArray,meanH3ThreshArray):
    
    poolIndex = 0
    for tauPoolSize in tauPoolSizeArray:
        tauPoolSize *= 0.025 #convert 25um resolution functional-registered coordinates to mm
        poolIndex += 1
        tauSortedPath = os.path.join(savePath,lineSelection,f'pooling{tauPoolSize}mm')
        if not os.path.exists(tauSortedPath):
            os.makedirs(tauSortedPath)
        
        ### pool tau into a grid for bootstrapping the regression ###
        minML_CCF, maxML_CCF, minAP_CCF, maxAP_CCF = np.min(allTauCCF_Coords[0,:]), np.max(allTauCCF_Coords[0,:]), np.min(allTauCCF_Coords[1,:]), np.max(allTauCCF_Coords[1,:])
        pooledTauCCF_coords = [np.empty((4,0)) for _ in range(numLayers)]
        pooledTauCCF_coords_noGene = [np.empty((4,0)) for _ in range(numLayers)]
        pooledPixelCount_v_CellCount = [np.empty((2,0)) for _ in range(numLayers)]
        pooledTau_cellAligned = [np.empty((1,0)) for _ in range(numLayers)]
        pooled_cell_region_H2layerFiltered = [np.empty((0,1)).astype(int) for _ in range(numLayers)]
        total_genes = gene_data_dense_H2layerFiltered[resolution][0].shape[1]
        resampledGenes_aligned = [np.empty((total_genes,0)) for _ in range(numLayers)]
        resampledH3_aligned_H2layerFiltered = [np.empty((1,0)) for _ in range(numLayers)] #[np.empty((1,0)) for _ in range(numLayers)]
        resampledH3_aligned_H2layerFiltered_OneHot = []
        H3_per_cell_H2layerFiltered_OneHot = []
        genePoolSaturation = []
        for layerIDX in range(numLayers):
            geneProfilePresentCount = 0
            possiblePoolsCount = 0
            print(f'Tau-Gene Alignment Pooling (size {tauPoolSize}mm): {layerNames[layerIDX]}')
            for current_tau_ML_pool in np.arange(minML_CCF,CCF_ML_Center_mm,tauPoolSize):
                current_ML_tau_pooling_IDXs = np.where(np.abs(np.abs(allTauCCF_Coords[0,:]-CCF_ML_Center_mm)-np.abs(current_tau_ML_pool-CCF_ML_Center_mm))<(tauPoolSize/2))[0] #our pixel space extents bilaterally, but CCF is unilateral, so 'CCF' coordinates from pixel space need to reflected over the ML center axis (CCF_ML_center)
                
                # if resolution == '25':
                #     cellwise_ML_CCF_25 = mlCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1)
                # if resolution == '10':
                #     cellwise_ML_CCF_25 = np.abs((mlCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1) * ((10/25)*100)) - CCF_ML_Center)

                current_ML_cell_pooling_IDXs = np.where(np.abs(mlCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1)-current_tau_ML_pool)<(tauPoolSize/2))[0]
                for current_tau_AP_pool in np.arange(minAP_CCF,maxAP_CCF,tauPoolSize):
                    current_tau_pooling_IDXs = np.where(np.abs(allTauCCF_Coords[1,current_ML_tau_pooling_IDXs]-current_tau_AP_pool)<(tauPoolSize/2))

                    # if resolution == '25':
                    #     cellwise_AP_CCF_25 = apCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1)
                    # if resolution == '10':
                    #     cellwise_AP_CCF_25 = apCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1) * ((10/25)*100)

                    current_cell_pooling_IDXs = np.where(np.abs(apCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1)[current_ML_cell_pooling_IDXs]-current_tau_AP_pool)<(tauPoolSize/2))[0]
                    pooledTaus = allTauCCF_Coords[2,current_ML_tau_pooling_IDXs[current_tau_pooling_IDXs]]
                    if pooledTaus.size > 0:
                        #print(mlCCF_per_cell_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs])
                        possiblePoolsCount += 1
                        if current_cell_pooling_IDXs.shape[0] > 0:
                            #print(current_tau_ML_pool,current_tau_AP_pool)

                            geneProfilePresentCount += 1

                            pooledTauCCF_coords[layerIDX] = np.hstack((pooledTauCCF_coords[layerIDX], np.array((current_tau_ML_pool,current_tau_AP_pool,np.mean(pooledTaus),np.std(pooledTaus))).reshape(-1,1))) #switched order
                            
                            pooledTau_cellAligned[layerIDX] = np.hstack((pooledTau_cellAligned[layerIDX],pooledTaus.reshape(1,-1)))
                            
                            gene_pool_data = gene_data_dense_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                            geneResamplingIDX = random.choices(np.arange(0,gene_pool_data.shape[0]), k=pooledTaus.shape[0])
                            resampledGenes_aligned[layerIDX] = np.hstack((resampledGenes_aligned[layerIDX],gene_pool_data[geneResamplingIDX,:].reshape(total_genes,-1)))

                            if resolution == '25':
                                H3_pool_data = H3_per_cell_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                                
                                data_flattened = H3_pool_data.flatten()
                                categories = np.arange(1, 10)
                                counts = np.array([np.sum(data_flattened == category) for category in categories])
                                normalized_counts = (counts / counts.sum()).reshape(9,-1)

                                resampledH3_aligned_H2layerFiltered[layerIDX] = np.hstack((resampledH3_aligned_H2layerFiltered[layerIDX],H3_pool_data[geneResamplingIDX,:].reshape(1,-1)))
                                #resampledH3_aligned_H2layerFiltered[layerIDX] = np.hstack((resampledH3_aligned_H2layerFiltered[layerIDX],normalized_counts[geneResamplingIDX,:].reshape(1,-1)))

                            pooledPixelCount_v_CellCount[layerIDX] = np.hstack((pooledPixelCount_v_CellCount[layerIDX],np.array((pooledTaus.shape[0],gene_pool_data.shape[0])).reshape(2,-1)))

                            cell_region_pool_data = cell_region_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                            pooled_cell_region_H2layerFiltered[layerIDX] = np.vstack((pooled_cell_region_H2layerFiltered[layerIDX],cell_region_pool_data[geneResamplingIDX,:].reshape(-1,1)))

                            #gene_pool_ML_CCF = mlCCF_per_cell_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs]].reshape(-1)
                            #gene_pool_AP_CCF = apCCF_per_cell_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs]].reshape(-1)

                            #print(f'CCF ML:{current_tau_ML_pool}, CCF AP:{current_tau_AP_pool}, GeneX ML CCFs:{gene_pool_ML_CCF}, GeneX AP CCFs:{gene_pool_AP_CCF}')
                        else:
                            pooledTauCCF_coords_noGene[layerIDX] = np.hstack((np.array((current_tau_ML_pool,current_tau_AP_pool,np.mean(pooledTaus),np.std(pooledTaus))).reshape(-1,1),pooledTauCCF_coords_noGene[layerIDX]))
            genePoolSaturation.append(geneProfilePresentCount/possiblePoolsCount)
            if resolution == '25':
                resampledH3_aligned_H2layerFiltered_OneHot.append(hotencoder.fit_transform(resampledH3_aligned_H2layerFiltered[layerIDX].T))
                H3_per_cell_H2layerFiltered_OneHot.append(hotencoder.fit_transform(H3_per_cell_H2layerFiltered[resolution][layerIDX]))

        for layerIDX in range(numLayers):
            plt.figure(), plt.title(f'CCF Pooling:{tauPoolSize}mm, Fraction of Tau Pooled Points with at least one Gene Profile:{round(genePoolSaturation[layerIDX],3)}\n{lineSelection}, {layerNames[layerIDX]}')
            plt.scatter(pooledTauCCF_coords_noGene[layerIDX][1,:],pooledTauCCF_coords_noGene[layerIDX][0,:],color='red',s=0.5)
            plt.scatter(pooledTauCCF_coords[layerIDX][1,:],pooledTauCCF_coords[layerIDX][0,:],color='green',s=0.5)
            plt.xlabel(r'A$\leftrightarrow$P (mm)'), plt.ylabel(r'L$\leftrightarrow$M (mm)'), plt.axis('equal')
            plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{lineSelection}_tauExpressionPooling{tauPoolSize}mm_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(1,1,figsize=(8,8))
            plt.suptitle(f'{lineSelection} Tau, CCF Pooling:{tauPoolSize}mm\n{lineSelection}, {layerNames[layerIDX]}')
            cmap = plt.get_cmap('cool')
            global_min,global_max = np.log10(1),np.log10(30)
            norm = matplotlib.colors.Normalize(global_min, global_max)
            tau_colors = cmap(norm(np.log10(pooledTauCCF_coords[layerIDX][2,:])))
            ax.scatter(pooledTauCCF_coords[layerIDX][1,:],pooledTauCCF_coords[layerIDX][0,:],color=tau_colors,s=6)
            ax.set_xlabel(r'A$\leftrightarrow$P (mm)'), ax.set_ylabel(r'L$\leftrightarrow$M (mm)'), ax.axis('equal')
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array(tau_colors)
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=35, pad=0.1, orientation='horizontal')
            cbar_ticks = np.arange(global_min, global_max, 1)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(10**(cbar_ticks),fontsize=6,rotation=45)
            cbar.set_label('Tau', rotation=0)
            plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{lineSelection}_tauPooling{tauPoolSize}mm_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
            plt.close()

            plt.figure(), plt.xlabel('Pool Pixel Count'), plt.ylabel('Pool Cell Count')
            plt.title(f'Pool Size:{tauPoolSize}mm, Pixel & Cell Counts by CCF Pool\n{layerNames[layerIDX]}\n{lineSelection}, {layerNames[layerIDX]}')
            plt.scatter(pooledPixelCount_v_CellCount[layerIDX][0,:],pooledPixelCount_v_CellCount[layerIDX][1,:],color='black',s=1)
            plt.axis('equal')
            plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{lineSelection}_pooling{tauPoolSize}mm_CellPixelCounts_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
            plt.close()

        #rename = os.path.join(tauSortedPath,f'{lineSelection}_tauExpressionPooling{tauPoolSize}.pdf')
        #PDFmerger(tauSortedPath,f'{lineSelection}_tauExpressionPooling{tauPoolSize}_',layerNames,'.pdf',rename)

        #rename = os.path.join(tauSortedPath,f'{lineSelection}_tauPooling{tauPoolSize}.pdf')
        #PDFmerger(tauSortedPath,f'{lineSelection}_tauPooling{tauPoolSize}_',layerNames,'.pdf',rename)

        #rename = os.path.join(tauSortedPath,f'{lineSelection}_pooling{tauPoolSize}_CellPixelCounts.pdf')
        #PDFmerger(tauSortedPath,f'{lineSelection}_pooling{tauPoolSize}_CellPixelCounts_',layerNames,'.pdf',rename)


        linearmodel = LinearRegression()
        for layerIDX in range(numLayers):
            standardized_CCF_Tau = standard_scaler.fit_transform(pooledTauCCF_coords[layerIDX].T).T
            
            r_squared_regression = []
            for dim in range(2):
                linearmodel.fit(standardized_CCF_Tau[dim,:].reshape(-1,1),standardized_CCF_Tau[2,:].reshape(-1))
                tau_pred = linearmodel.predict(standardized_CCF_Tau[dim,:].reshape(-1,1))
                r_squared_regression.append(r2_score(standardized_CCF_Tau[2,:].reshape(-1,1), tau_pred))

            fig, ax = plt.subplots(1,2,figsize=(8,4))
            ax[0].scatter(standardized_CCF_Tau[0,:],standardized_CCF_Tau[2,:],color='black',s=1)
            ax[1].scatter(standardized_CCF_Tau[1,:],standardized_CCF_Tau[2,:],color='black',s=1)
            ax[0].set_title(f'$R^2$={round(r_squared_regression[0],3)}'), ax[1].set_title(f'$R^2$={round(r_squared_regression[1],3)}')
            ax[0].set_xlabel('Standardized ML CCF'), ax[1].set_xlabel('Standardized AP CCF')
            ax[0].set_ylabel('Standardized Tau'), ax[1].set_ylabel('Standardized Tau')
            plt.suptitle(f'Pooling={tauPoolSize}mm, Standardized {lineSelection} Tau & AP, ML CCF Correlation\n{layerNames[layerIDX]}')
            plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{lineSelection}_pooling{tauPoolSize}mm_TauCCFcorr_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
            plt.close()
        #rename = os.path.join(tauSortedPath,f'{lineSelection}_pooling{tauPoolSize}mm_TauCCFcorr.pdf')
        #PDFmerger(tauSortedPath,f'{lineSelection}_pooling{tauPoolSize}_TauCCFcorr_',layerNames,'.pdf',rename)

        # plt.plot(np.mean(standard_scaler.fit_transform(np.asarray(gene_data_dense_H2layerFiltered[layerIDX][:,:]).T).T,axis=0))

        # expressionPercentiles = [np.zeros((5,len(geneNames))) for _ in range(numLayers)]
        # for layerIDX in range(numLayers):
        #     expressionPercentiles[layerIDX][:,:] = np.percentile(np.asarray(gene_data_dense_H2layerFiltered[layerIDX][:,:]),[0,25,50,75,100],axis=0)

        # plt.figure()
        # for per in range(5):
        #     plt.plot(expressionPercentiles[layerIDX][per,:])
        # plt.yscale('log')
        # plt.ylim(1e-1,)



        ######################################################################################
        ### Standard Scaler transform the gene expression and tau data prior to regression ###
        mean_expression_standard = np.zeros_like(mean_expression[resolution])
        # Tau Regressions #
        pooledTau_cellAligned_standard = [np.zeros_like(pooledTau_cellAligned[layerIDX].T) for layerIDX in range(numLayers)]
        resampledGenes_aligned_H2layerFiltered_standard = [np.zeros_like(resampledGenes_aligned[layerIDX].T) for layerIDX in range(numLayers)]
        # CCF Regressions #
        gene_data_dense_H2layerFiltered_standard = [np.zeros_like(gene_data_dense_H2layerFiltered[resolution][layerIDX]) for layerIDX in range(numLayers)]
        #tau_per_cell_H2layerFiltered_standard = [np.zeros_like(tau_per_cell_H2layerFiltered[layerIDX]) for layerIDX in range(numLayers)]
        mlCCF_per_cell_H2layerFiltered_standard = [np.zeros_like(mlCCF_per_cell_H2layerFiltered[resolution][layerIDX]) for layerIDX in range(numLayers)]
        apCCF_per_cell_H2layerFiltered_standard = [np.zeros_like(apCCF_per_cell_H2layerFiltered[resolution][layerIDX]) for layerIDX in range(numLayers)]
        for layerIDX in range(numLayers):
            ## Tau ##
            pooledTau_cellAligned_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(pooledTau_cellAligned[layerIDX][:,:]).T)
            #tau_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(tau_per_cell_H2layerFiltered[layerIDX][:,:]))
            ## Genes ##
            mean_expression_standard[layerIDX,:,:] = standard_scaler.fit_transform(mean_expression[resolution][layerIDX,:,:])
            resampledGenes_aligned_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(resampledGenes_aligned[layerIDX][:,:]).T)
            gene_data_dense_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(gene_data_dense_H2layerFiltered[resolution][layerIDX][:,:]))
            # CCF #
            mlCCF_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(mlCCF_per_cell_H2layerFiltered[resolution][layerIDX][:,:]))
            apCCF_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(apCCF_per_cell_H2layerFiltered[resolution][layerIDX][:,:]))
                
        #print(np.mean(resampledGenes_aligned_H2layerFiltered_standard[0][:,:],axis=0)) #just to see that the means are zero after standardizing


        if poolIndex == 1:
            regressionsToStart = [0,1]
            plottingConditions = [False,True] #plot spatial reconstruction?
        else:
            regressionsToStart = [0,1]#[0] #no need to run spatial regression multiple times across pooling sizes, just when the meanPredictionThresh changes
            plottingConditions = [False]#[False] #no need to plot spatial regression plots across "..."


        for namePredictors,predictorTitle,predictorEncodeType,predictorPathSuffix in zip(['Gene Predictors', 'H3 Predictors'],
                                                                                        ['Gene Expression',  'H3 Level'],
                                                                                        ['Standardized',     'OneHot'],
                                                                                        ['GenePredictors',   'H3Predictors']):
            
            if (predictorPathSuffix == 'H3Predictors') and (datasetName == 'Merfish-Imputed'):
                break
            
            if not os.path.exists(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'{datasetName}')):
                os.makedirs(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'{datasetName}'))

            if predictorPathSuffix == 'GenePredictors':
                predictorDataRaw = gene_data_dense_H2layerFiltered[resolution]
                meanPredictionThresh = meanExpressionThresh
                if resolution == '25':
                    #predictorDataRaw = gene_data_dense_H2layerFiltered[resolution]
                    #meanPredictionThresh = meanExpressionThresh
                    predictorNamesArray = np.array(pilotGeneNames)
                    #numLayers = len(pilotLayerNames)
                    #layerNames = pilotLayerNames
                if resolution == '10':
                    #predictorDataRaw = [raw_merfish_genes]
                    #meanPredictionThresh = meanExpressionThresh
                    predictorNamesArray = np.array(enrichedGeneNames)
                    #numLayers = len(merfishLayerNames)
                    #layerNames = merfishLayerNames
                
            if predictorPathSuffix == 'H3Predictors':
                predictorDataRaw = H3_per_cell_H2layerFiltered_OneHot
                meanPredictionThresh = meanH3Thresh
                predictorNamesArray = np.arange(0, predictorDataRaw[layerIDX].shape[1], 1)
                numLayers = len(layerIDs)
                layerNames = pilotLayerNames
            
            # if predictorPathSuffix == 'merfishImputedGenePredictors':
            #     predictorDataRaw = [raw_merfish_genes]
            #     meanPredictionThresh = 0.1
            #     predictorNamesArray = np.array(enrichedGeneNames)
            #     numLayers = 1
            #     layerNames = merfishLayerNames

            if predictorPathSuffix == 'merfishImputedGenePredictors':
                figWidth = 20
            else:
                figWidth = 15

            highMeanPredictorIDXs = [[] for _ in range(numLayers)]
            for layerIDX in range(numLayers):
                layerMeanPredictors = np.mean(np.asarray(predictorDataRaw[layerIDX][:,:]),axis=0)
                highMeanPredictorIDXs[layerIDX] = (np.where(layerMeanPredictors > meanPredictionThresh)[0]).astype(int)
                #print(highMeanPredictorIDXs[layerIDX].shape[0])

                sortedMeanPredictor = np.argsort(layerMeanPredictors)
                meanPredictorCutoffIDX = np.argmin(np.abs(layerMeanPredictors[sortedMeanPredictor]-meanPredictionThresh))

                # geneNameOI = 'Cux2'
                # geneOI = np.where(gene_names==geneNameOI)[0][0]
                # plt.figure()
                # plt.hist(gene_data_dense_H2layerFiltered[layerIDX,:,geneOI],bins=10,color='black')
                # #plt.yscale('log')
                # plt.title(f'Cux2 Expression, Layer {layerNames[layerIDX]}')
                # plt.xlim(0,10)
                
                #lowExpressionGeneIDXs = np.where(layerMeanPredictors < 0.1)[0]
                #print(f'{layerNames[layerIDX]}: \n{np.array(geneNames)[lowExpressionGeneIDXs]}\n\n')
                fig, ax = plt.subplots(1,1,figsize=(figWidth,10))
                ax.plot(layerMeanPredictors[sortedMeanPredictor],color='black')
                ax.vlines(x=meanPredictorCutoffIDX, ymin=0, ymax=np.max(layerMeanPredictors),color='black',alpha=0.5,linestyles='dashed')
                ax.hlines(y=meanPredictionThresh, xmin=0, xmax=predictorDataRaw[layerIDX].shape[1],color='black',alpha=0.5,linestyles='dashed')
                ax.set_xticks(np.arange(0, predictorDataRaw[layerIDX].shape[1], 1))
                if (predictorPathSuffix == 'pilotGenePredictors') or (predictorPathSuffix == 'merfishImputedGenePredictors'):
                    ax.set_xticklabels(predictorNamesArray[sortedMeanPredictor], rotation=90)
                else:
                    ax.set_xticklabels(predictorNamesArray[sortedMeanPredictor])
                ax.set_ylabel(f'Mean {predictorTitle}')
                ax.set_yscale('log')
                ax.set_title(f'{layerNames[layerIDX]}, Num Excluded {namePredictors}:{meanPredictorCutoffIDX}, Mean {predictorTitle} Cutoff:{meanPredictionThresh}')
                plt.savefig(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'{datasetName}',f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                plt.close()

            rename = os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}.pdf')
            #PDFmerger(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}'),f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}_',layerNames,'.pdf',rename)


            for regressionType in regressionsToStart:
                if regressionType == 0: # geneX, H3 -> Tau
                    spatialReconstruction = False
                    tauRegression = True
                    cell_region_filtered = pooled_cell_region_H2layerFiltered
                    y_data = pooledTau_cellAligned_standard
                    pred_dim = 1
                    if predictorPathSuffix == 'GenePredictors':
                        x_data = resampledGenes_aligned_H2layerFiltered_standard
                    if predictorPathSuffix == 'H3Predictors':
                        x_data = resampledH3_aligned_H2layerFiltered_OneHot

                if regressionType == 1: # geneX, H3 -> CCF
                    spatialReconstruction = True
                    tauRegression = False
                    cell_region_filtered = cell_region_H2layerFiltered[resolution]
                    y_data = [np.hstack((apCCF_per_cell_H2layerFiltered_standard[layerIDX],mlCCF_per_cell_H2layerFiltered_standard[layerIDX])) for layerIDX in range(numLayers)]
                    pred_dim = 2
                    if predictorPathSuffix == 'GenePredictors':
                        x_data = gene_data_dense_H2layerFiltered_standard
                    if predictorPathSuffix == 'H3Predictors':
                        x_data = H3_per_cell_H2layerFiltered_OneHot

                # if regressionType == 2: # H3 -> Tau
                #     spatialReconstruction = False
                #     tauRegression = True
                #     genePredictors = False
                #     cell_region = pooled_cell_region_H2layerFiltered
                #     y_data = pooledTau_cellAligned_standard
                #     pred_dim = 1
                #     x_data = resampledH3_aligned_H2layerFiltered

                # if regressionType == 3: # H3 -> CCF
                #     spatialReconstruction = True
                #     tauRegression = False
                #     genePredictors = False
                #     cell_region = cell_region_H2layerFiltered
                #     y_data = [np.hstack((apCCF_per_cell_H2layerFiltered_standard[layerIDX],mlCCF_per_cell_H2layerFiltered_standard[layerIDX])) for layerIDX in range(numLayers)]
                #     pred_dim = 2
                #     x_data = H3_per_cell_H2layerFiltered

                # if regressionType == 2: #Merfish-Imputed -> Tau
                #     spatialReconstruction = False
                #     tauRegression = True
                #     cell_region_filtered = pooled_cell_region_H2layerFiltered
                #     y_data = pooledTau_cellAligned_standard
                #     pred_dim = 1
                #     x_data = resampledGenes_aligned_H2layerFiltered_standard

                # if regressionType == 3: # Merfish-Imputed -> CCF
                #     spatialReconstruction = True
                #     tauRegression = False
                #     cell_region_filtered = cell_region_H2layerFiltered[resolution]
                #     x_data = [np.array(standardMerfish_CCF_Genes.loc[:,enrichedGeneNames])]
                #     y_data = [np.array(standardMerfish_CCF_Genes.loc[:,['x_ccf','y_ccf']])]
                #     pred_dim = 2
                #     highMeanPredictorIDXs = [[] for _ in range(numLayers)]
                #     for layerIDX in range(numLayers):
                #         layerMeanPredictors = np.mean(np.asarray(predictorDataRaw[layerIDX][:,:]),axis=0)
                #         highMeanPredictorIDXs[layerIDX] = (np.where(layerMeanPredictors > meanPredictionThresh)[0]).astype(int)

                if regressionType == 4: # CCF <-> Tau
                    print('To Do: make regression type 4')
                    break
                
                # if ???:
                #     y_data = tau_per_cell_H2layerFiltered_standard

                regressionConditions = [spatialReconstruction,tauRegression,regionalResample] #genePredictors (relic term at index 3)
                n_splits = 5
                alphaParams = [-4,0,25] #Alpha Lower (10**x), Alpha Upper (10**x), Steps

                print(f'Starting Regressions, Type {regressionType}:')
                if (regressionType == 0):
                    print(f'{datasetName} {predictorTitle} -> {lineSelection} Tau (CCF Pooling={tauPoolSize}mm, predThresh={meanPredictionThresh}, regionResamp={regressionConditions[2]})')
                    best_coef_tau,lasso_weight_tau,bestAlpha_tau,alphas_tau,tauPredictions_tau,bestR2_tau = layerRegressions(pred_dim,n_splits,highMeanPredictorIDXs,x_data,y_data,layerNames,regressionConditions,cell_region_filtered,alphaParams)
                
                if (regressionType == 1):
                    print(f'{datasetName} {predictorTitle} -> CCF (predThresh={meanPredictionThresh}, regionResamp={regressionConditions[2]})')
                    best_coef_spatial,lasso_weight_spatial,bestAlpha_spatial,alphas_spatial,tauPredictions_spatial,bestR2_spatial = layerRegressions(pred_dim,n_splits,highMeanPredictorIDXs,x_data,y_data,layerNames,regressionConditions,cell_region_filtered,alphaParams)

                # if regressionType == 2:
                #     print(f'{datasetName} {predictorTitle} -> {lineSelection} Tau (CCF Pooling={tauPoolSize}, predThresh={meanPredictionThresh}, regionResamp={regressionConditions[2]})')
                #     best_coef_spatial,lasso_weight_spatial,bestAlpha_spatial,alphas_spatial,tauPredictions_spatial,bestR2_spatial = layerRegressions(pred_dim,n_splits,highMeanPredictorIDXs,x_data,y_data,layerNames,regressionConditions,[],alphaParams)

                # if regressionType == 3:
                #     print(f'{datasetName} {predictorTitle} -> CCF (predThresh={meanPredictionThresh}, regionResamp={regressionConditions[2]})')
                #     best_coef_spatial,lasso_weight_spatial,bestAlpha_spatial,alphas_spatial,tauPredictions_spatial,bestR2_spatial = layerRegressions(pred_dim,n_splits,highMeanPredictorIDXs,x_data,y_data,layerNames,regressionConditions,[],alphaParams)



            mean_fold_coef_tau = [np.mean(best_coef_tau[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
            sd_fold_coef_tau = [np.std(best_coef_tau[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
            sorted_coef_tau = [np.argsort(mean_fold_coef_tau[layerIDX]) for layerIDX in range(numLayers)]
            mean_fold_coef_spatial = [np.mean(best_coef_spatial[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
            sd_fold_coef_spatial = [np.std(best_coef_spatial[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
            sorted_coef_spatial = [np.argsort(mean_fold_coef_spatial[layerIDX]) for layerIDX in range(numLayers)]







            ###################################################################
            ################### Regression Outputs Plotting ###################
            resampTitle = f'predThresh={meanPredictionThresh}'
            for spatialReconstruction in plottingConditions: #[True,False]
                if spatialReconstruction:
                    plottingTitles = ["A-P CCF","M-L CCF"]
                    titleAppend = f'Spatial Reconstruction from {datasetName} {predictorTitle}, {resampTitle}'
                    tauPredictions = tauPredictions_spatial
                    bestR2 = bestR2_spatial
                    mean_fold_coef = mean_fold_coef_spatial
                    sorted_coef = sorted_coef_spatial
                    bestAlpha = bestAlpha_spatial
                    alphas = alphas_spatial
                    lasso_weight = lasso_weight_spatial
                    sd_fold_coef = sd_fold_coef_spatial
                    pred_dim = 2
                    plottingDir = os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'{datasetName}')
                else:
                    plottingTitles = ["tau"]
                    titleAppend = f'{lineSelection} Tau Reconstruction from {datasetName} {predictorTitle} (pooling={tauPoolSize}mm, {resampTitle})'
                    tauPredictions = tauPredictions_tau
                    bestR2 = bestR2_tau
                    mean_fold_coef = mean_fold_coef_tau
                    sorted_coef = sorted_coef_tau
                    bestAlpha = bestAlpha_tau
                    alphas = alphas_tau
                    lasso_weight = lasso_weight_tau
                    sd_fold_coef = sd_fold_coef_tau
                    pred_dim = 1
                    plottingDir = os.path.join(tauSortedPath,f'{predictorPathSuffix}',f'{datasetName}')
                    if not os.path.exists(plottingDir):
                        os.makedirs(plottingDir)


                beta_dict = {}
                beta_dict['mean_fold_coef'] = mean_fold_coef
                beta_dict['sd_fold_coef'] = sd_fold_coef
                beta_dict['sorted_coef'] = sorted_coef
                beta_dict['layerNames'] = layerNames
                beta_dict['regressionTitle'] = titleAppend
                beta_dict['spatialReconstruction'] = spatialReconstruction
                with open(os.path.join(plottingDir,f'betaDictionary.txt'), 'wb+') as f:
                    pickle.dump(beta_dict, f)

                # if spatialReconstruction and (predictorPathSuffix != 'H3Predictors') and (meanExpressionThresh == 0):
                #     fig, axes = plt.subplots(numLayers,numLayers,figsize=(15,15))
                #     axes = np.atleast_1d(axes)
                #     plt.suptitle(f'Cross-Layer A-P $\\beta$ Correlations, {datasetName} Transcripts')
                #     for layerIDX0 in range(numLayers):
                #         for layerIDX1 in range(numLayers):

                #             linearmodel.fit(mean_fold_coef_spatial[layerIDX0][0].reshape(-1,1),mean_fold_coef_spatial[layerIDX1][0].reshape(-1,1))
                #             beta_pred = linearmodel.predict(mean_fold_coef_spatial[layerIDX0][0].reshape(-1,1))
                #             L2L_r2 = r2_score(mean_fold_coef_spatial[layerIDX1][0].reshape(-1,1), beta_pred)
                            
                #             axes[layerIDX0,layerIDX1].set_title(f'$R^2$={round(L2L_r2,3)}')
                #             axes[layerIDX0,layerIDX1].scatter(mean_fold_coef_spatial[layerIDX0][0],mean_fold_coef_spatial[layerIDX1][0],color='black',s=0.25)
                #             axes[layerIDX0,layerIDX1].errorbar(mean_fold_coef_spatial[layerIDX0][0],mean_fold_coef_spatial[layerIDX1][0], xerr=sd_fold_coef_spatial[layerIDX0][0], yerr=sd_fold_coef_spatial[layerIDX1][0], fmt="o", color='black', markersize=0.25)
                #             for i, predictorText in enumerate(predictorNamesArray):
                #                 axes[layerIDX0,layerIDX1].annotate(predictorText, (mean_fold_coef_spatial[layerIDX0][0][i], mean_fold_coef_spatial[layerIDX1][0][i]))
                #             if layerIDX1 == 0:
                #                 axes[layerIDX0,layerIDX1].set_ylabel(f"{layerNames[layerIDX0]} A-P$\\beta$")
                #             if layerIDX0 == numLayers-1:
                #                 axes[layerIDX0,layerIDX1].set_xlabel(f"{layerNames[layerIDX1]} A-P$\\beta$")
                #     plt.savefig(os.path.join(savePath,'Spatial',f'crossLayer_{datasetName}APbeta_Correlations.pdf'),dpi=600,bbox_inches='tight')
                #     plt.close()


                with open(os.path.join(plottingDir,f'regression_{titleAppend}.txt'), "w") as file:
                    file.write(f'{titleAppend}\n\n')

                for layerIDX,layerName in enumerate(layerNames):
                    
                    bestR2_mean, bestR2_SD = round(np.mean(bestR2[layerIDX,:]),numPrecision), round(np.std(bestR2[layerIDX,:]),numPrecision)
                    bestAlpha_mean, bestAlpha_SD = np.mean(bestAlpha[layerIDX,:]), np.std(bestAlpha[layerIDX,:])
                    
                    with open(os.path.join(plottingDir,f'regression_{titleAppend}.txt'), "a") as file:
                        file.write(f'{layerName}, Best R2+-SD:{bestR2_mean}+-{bestR2_SD} (at alpha+-SD={round(bestAlpha_mean,alphaPrecision)}+-{round(bestAlpha_SD,alphaPrecision)})\n')
                        for dim in range(pred_dim):
                            if spatialReconstruction:
                                file.write(f'####### {plottingTitles[dim]} #######\n')
                            file.write(f'Highest + Predictors:{predictorNamesArray[highMeanPredictorIDXs[layerIDX]][sorted_coef[layerIDX][dim,:]][-5:]}\n')
                            file.write(f'Predictor Weights:{np.round(mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]][-5:],3)}\n')
                            file.write(f'Lowest - Predictors:{predictorNamesArray[highMeanPredictorIDXs[layerIDX]][sorted_coef[layerIDX][dim,:]][:5]}\n')
                            file.write(f'Predictor Weights:{np.round(mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]][:5],3)}\n')
                        file.write(f'\n')

                    fig, axes = plt.subplots(pred_dim,1,figsize=(10,10))
                    plt.suptitle(f'{titleAppend}, {layerName}, best $R^2\pm$SD: {bestR2_mean}$\pm${bestR2_SD} ($\\alpha\pm$SD: {round(bestAlpha_mean,alphaPrecision)}$\pm${round(bestAlpha_SD,alphaPrecision)})')
                    for dim,ax in enumerate(np.atleast_1d(axes)):
                        ax.set_title(f'{plottingTitles[dim]}')
                        ax.plot(alphas, lasso_weight[layerIDX][:,dim,:])
                        ax.vlines(x=bestAlpha_mean, ymin=-10**5, ymax=10**5, color='black')
                        ax.fill_betweenx(y=np.array([-10**5,10**5]), x1=bestAlpha_mean-bestAlpha_SD, x2=bestAlpha_mean+bestAlpha_SD, color='gray', alpha=0.3)
                        ax.set_xscale('log')
                        ax.set_ylim(np.min(lasso_weight[layerIDX][:,:]), np.max(lasso_weight[layerIDX][:,:]))
                        ax.set_xlabel(f'$\\alpha$')
                        ax.set_ylabel(f'$\\beta$')
                    if plotting:
                        plt.savefig(os.path.join(plottingDir,f'{predictorPathSuffix}LassoWeightsAll_{layerName}_{titleAppend}.pdf'),dpi=600,bbox_inches='tight')
                        plt.close()
                    
                    if spatialReconstruction:
                        fig, ax = plt.subplots(1,1,figsize=(10,10))
                        ax.set_title(f'{titleAppend}\nA-P vs M-L $\\beta$ Values\n{layerName}, $\\alpha\pm$SD={round(bestAlpha_mean,alphaPrecision)}$\pm${round(bestAlpha_SD,alphaPrecision)}, $R^2\pm$SD={bestR2_mean}$\pm${bestR2_SD}, error:5-fold SD')
                        ax.scatter(mean_fold_coef[layerIDX][0,:],mean_fold_coef[layerIDX][1,:],color='black',s=0.5)
                        plt.errorbar(mean_fold_coef[layerIDX][0,:], mean_fold_coef[layerIDX][1,:], xerr=sd_fold_coef[layerIDX][0,:], yerr=sd_fold_coef[layerIDX][1,:], fmt="o", color='black')
                        for i, predictorText in enumerate(predictorNamesArray[highMeanPredictorIDXs[layerIDX]]):
                            ax.annotate(predictorText, (mean_fold_coef[layerIDX][0,i], mean_fold_coef[layerIDX][1,i]))
                        ax.set_xlabel(f'A-P $\\beta$')
                        ax.set_ylabel(f'M-L $\\beta$')
                        if plotting:
                            plt.savefig(os.path.join(plottingDir,f'APvsML_{predictorPathSuffix}BetaWeights_{layerName}_{titleAppend}.pdf'),dpi=600,bbox_inches='tight')
                            plt.close()

                    fig, axes = plt.subplots(pred_dim,1,figsize=(16,13))
                    plt.suptitle(f'{titleAppend}\n{layerName}, $\\alpha\pm$SD={round(bestAlpha_mean,alphaPrecision)}$\pm${round(bestAlpha_SD,alphaPrecision)}, $R^2\pm$SD={bestR2_mean}$\pm${bestR2_SD}, error:5-fold SD')
                    for dim,ax in enumerate(np.atleast_1d(axes)):
                        ax.set_title(f'{plottingTitles[dim]}')
                        #ax.scatter(np.arange(0,gene_data_dense.shape[1],1),mean_fold_coef[sorted_coef],color='black')
                        ax.errorbar(np.arange(0,highMeanPredictorIDXs[layerIDX].shape[0],1), mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]], yerr=sd_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]], fmt="o",color='black')
                        ax.hlines(y=0, xmin=0, xmax=highMeanPredictorIDXs[layerIDX].shape[0],color='black',alpha=0.5,linestyles='dashed')
                        ax.set_xticks(np.arange(0, highMeanPredictorIDXs[layerIDX].shape[0], 1))
                        ax.set_xticklabels(predictorNamesArray[highMeanPredictorIDXs[layerIDX]][sorted_coef[layerIDX][dim,:]], rotation=90)
                        ax.set_ylabel(f'$\\beta$')
                    if plotting:
                        plt.savefig(os.path.join(plottingDir,f'{predictorPathSuffix}LassoWeights_{layerName}_{titleAppend}.pdf'),dpi=600,bbox_inches='tight')
                        plt.close()
                    
                    fig1, axes = plt.subplots(pred_dim,1,figsize=(12,12))
                    #fig2, ax2 = plt.subplots(1,1,figsize=(15,8))
                    for dim,ax in enumerate(np.atleast_1d(axes)):
                        for structIDX,structureOfInterest in enumerate(structList):
                            ax.plot(np.arange(0,highMeanPredictorIDXs[layerIDX].shape[0],1),mean_expression_standard[layerIDX,structIDX,highMeanPredictorIDXs[layerIDX]].reshape(-1,1)[sorted_coef[layerIDX][dim,:]],label=structureOfInterest,color=areaColors[structIDX])
                            #ax2.plot(np.arange(0,total_genes,1),gaussian_filter(mean_expression_standard[structIDX,:].reshape(-1,1)[sorted_coef],sigma=2),label=structureOfInterest,color=areaColors[structIDX])
                        ax.set_xticks(np.arange(0, highMeanPredictorIDXs[layerIDX].shape[0], 1))
                        ax.set_xticklabels(predictorNamesArray[highMeanPredictorIDXs[layerIDX]][sorted_coef[layerIDX][dim,:]], rotation=90)
                        ax.legend()
                        ax.set_ylabel(f'{predictorEncodeType} {predictorTitle}')
                        ax.set_title(f'{titleAppend}, {layerName}, $\\alpha\pm$SD={round(bestAlpha_mean,alphaPrecision)}$\pm${round(bestAlpha_SD,alphaPrecision)}, $R^2\pm$SD={bestR2_mean}$\pm${bestR2_SD}')
                    if plotting:
                        plt.savefig(os.path.join(plottingDir,f'regional{predictorPathSuffix}_{layerName}_{titleAppend}.pdf'),dpi=600,bbox_inches='tight')
                        plt.close()

                    for dim in range(pred_dim):
                        currentPlottingTitle = plottingTitles[dim]
                        fig, axes = plt.subplots(1,n_splits,figsize=(20,6))
                        plt.suptitle(f'{titleAppend}, {layerName}')
                        for foldIDX,ax in enumerate(axes):
                            #cell_region_IDX = (cell_region_H2layerFiltered[layerIDX][test_index,:]).astype(int).reshape(-1)
                            test_y = tauPredictions[layerIDX][foldIDX][:,dim]
                            pred_y = tauPredictions[layerIDX][foldIDX][:,dim+pred_dim]
                            cell_region_IDX = tauPredictions[layerIDX][foldIDX][:,-1].astype(int)
                            cell_region_colors = np.asarray(areaColors)[cell_region_IDX]

                            for regionIDX,region in enumerate(structList):
                                regionR2IDXs = np.where(cell_region_IDX == regionIDX)
                                label = f'{region}' #R2: {round(r2_score(test_y[regionR2IDXs],pred_y[regionR2IDXs]),3)}'
                                ax.scatter(test_y[regionR2IDXs],pred_y[regionR2IDXs],color=areaColors[regionIDX],label=label,s=1)

                            #ax.scatter(test_y,pred_y,color=cell_region_colors,s=1)
                            #ax.axis('equal')
                            #ax.set_xlim(4,10)
                            #ax.set_ylim(4,10)
                            ax.set_title(f'Fold: {foldIDX}')
                            ax.set_xlabel(f'True {currentPlottingTitle} (standardized)')
                            ax.set_ylabel(f'Predicted {currentPlottingTitle} (standardized)')
                            if foldIDX==0:
                                ax.legend()
                        if plotting:
                            plt.savefig(os.path.join(plottingDir,f'predicted{currentPlottingTitle}_{layerName}_{titleAppend}.pdf'),dpi=600,bbox_inches='tight')
                            plt.close()

                # if plotting:
                #     rename = os.path.join(plottingDir,f'{predictorPathSuffix}LassoWeightsAll_{titleAppend}.pdf')
                #     #PDFmerger(plottingDir,f'{predictorPathSuffix}LassoWeightsAll_',layerNames,f'_{titleAppend}.pdf',rename)

                #     if spatialReconstruction:
                #         rename = os.path.join(plottingDir,f'APvsML_{predictorPathSuffix}BetaWeights_{titleAppend}.pdf')
                #         #PDFmerger(plottingDir,f'APvsML_{predictorPathSuffix}BetaWeights_',layerNames,f'_{titleAppend}.pdf',rename)
                    
                #     rename = os.path.join(plottingDir,f'{predictorPathSuffix}LassoWeights_{titleAppend}.pdf')
                #     #PDFmerger(plottingDir,f'{predictorPathSuffix}LassoWeights_',layerNames,f'_{titleAppend}.pdf',rename)

                #     rename = os.path.join(plottingDir,f'regional{predictorPathSuffix}_{titleAppend}.pdf')
                #     #PDFmerger(plottingDir,f'regional{predictorPathSuffix}_',layerNames,f'_{titleAppend}.pdf',rename)

                #     for dim in range(pred_dim):
                #         currentPlottingTitle = plottingTitles[dim]
                #         rename = os.path.join(plottingDir,f'predicted{currentPlottingTitle}_{titleAppend}.pdf')
                #         #PDFmerger(plottingDir,f'predicted{currentPlottingTitle}_',layerNames,f'_{titleAppend}.pdf',rename)



            for layerIDX,layerName in enumerate(layerNames):
                fig, ax = plt.subplots(1,2,figsize=(15,7))
                plt.suptitle(f'{resampTitle} Spatial and {lineSelection} Tau Reconstruction from {datasetName} {predictorTitle}\n{lineSelection} Tau vs A-P, M-L $\\beta$ Values\n{layerName}, error:5-fold SD')
                ax[0].scatter(mean_fold_coef_tau[layerIDX].reshape(-1), mean_fold_coef_spatial[layerIDX][0,:].reshape(-1),color='black',s=0.5)
                ax[0].errorbar(mean_fold_coef_tau[layerIDX].reshape(-1), mean_fold_coef_spatial[layerIDX][0,:].reshape(-1), xerr=sd_fold_coef_tau[layerIDX][0,:], yerr=sd_fold_coef_spatial[layerIDX][0,:], fmt="o", color='black')
                ax[1].scatter(mean_fold_coef_tau[layerIDX].reshape(-1), mean_fold_coef_spatial[layerIDX][1,:].reshape(-1),color='black',s=0.5)
                ax[1].errorbar(mean_fold_coef_tau[layerIDX].reshape(-1), mean_fold_coef_spatial[layerIDX][1,:].reshape(-1), xerr=sd_fold_coef_tau[layerIDX][0,:], yerr=sd_fold_coef_spatial[layerIDX][1,:], fmt="o", color='black')
                for i, predictorText in enumerate(predictorNamesArray[highMeanPredictorIDXs[layerIDX]]):
                    ax[0].annotate(predictorText, (mean_fold_coef_tau[layerIDX][0,i], mean_fold_coef_spatial[layerIDX][0,i]))
                    ax[1].annotate(predictorText, (mean_fold_coef_tau[layerIDX][0,i], mean_fold_coef_spatial[layerIDX][1,i]))
                ax[0].set_xlabel(f'Tau $\\beta$')
                ax[0].set_ylabel(f'A-P $\\beta$')
                ax[1].set_xlabel(f'Tau $\\beta$')
                ax[1].set_ylabel(f'M-L $\\beta$')
                #plt.savefig(os.path.join(plottingDir,f'{datasetName}_{resampTitle}_{lineSelection}Tau_vs_AP&ML_Betas_{layerName}.pdf'),dpi=600,bbox_inches='tight')
                plt.savefig(os.path.join(tauSortedPath,f'{predictorPathSuffix}',f'{datasetName}',f'{datasetName}_{resampTitle}_{lineSelection}Tau_vs_AP&ML_Betas_{layerName}.pdf'),dpi=600,bbox_inches='tight')
                plt.close()
            #rename = os.path.join(tauSortedPath,f'{predictorPathSuffix}',f'{datasetName}_{resampTitle}_{lineSelection}Tau_vs_AP&ML_Betas.pdf')
            #PDFmerger(os.path.join(tauSortedPath,f'{predictorPathSuffix}'),f'{resampTitle}_{lineSelection}Tau_vs_AP&ML_Betas_',layerNames,'.pdf',rename)


            if poolIndex == 1:
                for layerIDX,layerName in enumerate(layerNames):
                    apR2,mlR2 = [],[]
                    for foldIDX in range(n_splits):
                        apR2.append(r2_score(tauPredictions_spatial[layerIDX][foldIDX][:,0],tauPredictions_spatial[layerIDX][foldIDX][:,2]))
                        mlR2.append(r2_score(tauPredictions_spatial[layerIDX][foldIDX][:,1],tauPredictions_spatial[layerIDX][foldIDX][:,3]))
                    fig, axes = plt.subplots(1,2,figsize=(20,10))
                    plt.suptitle(f'{layerName} Cross-Fold Spatial Reconstructions from {predictorEncodeType} {predictorTitle}')
                    axes[0].set_xlabel(f'True Standardized A-P CCF'), axes[0].set_ylabel('True Standardized M-L CCF')
                    axes[1].set_xlabel(f'Predicted Standardized A-P CCF\n$R^2$={round(np.mean(apR2),3)}'), axes[1].set_ylabel(f'Predicted Standardized M-L CCF\n$R^2$={round(np.mean(mlR2),3)}')
                    for regionIDX,region in enumerate(structList):
                        for foldIDX in range(n_splits):
                            regionR2IDXs = np.where(tauPredictions_spatial[layerIDX][foldIDX][:,-1] == regionIDX)
                            axes[0].scatter(tauPredictions_spatial[layerIDX][foldIDX][regionR2IDXs,0],tauPredictions_spatial[layerIDX][foldIDX][regionR2IDXs,1],color=areaColors[regionIDX],s=0.15)
                            axes[1].scatter(tauPredictions_spatial[layerIDX][foldIDX][regionR2IDXs,2],tauPredictions_spatial[layerIDX][foldIDX][regionR2IDXs,3],color=areaColors[regionIDX],s=0.15)
                            #axes[0].axis('equal')
                            #axes[1].axis('equal')
                            for axnum in range(2):
                                axes[axnum].set_xlim(-2.5,2.5), axes[axnum].set_ylim(-2.5,2.5)
                    plt.savefig(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'{predictorPathSuffix}Thresh{meanPredictionThresh}_spatialReconstruction_{layerName}.pdf'),dpi=600,bbox_inches='tight')
                    plt.close()
                #rename = os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'{predictorPathSuffix}Thresh{meanPredictionThresh}_spatialReconstruction.pdf')
                #PDFmerger(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}'),f'{predictorPathSuffix}Thresh{meanPredictionThresh}_spatialReconstruction_',layerNames,'.pdf',rename)



# sorted_coef = np.argsort(best_coef[layerIDX,foldIDX,:])
# fig, ax = plt.subplots(1,1,figsize=(15,8))
# ax.plot(np.arange(0,gene_data_dense.shape[1],1),best_coef[layerIDX,foldIDX,sorted_coef])
# ax.set_xticks(np.arange(0, gene_data_dense.shape[1], 1))
# ax.set_xticklabels(np.array(geneNames)[sorted_coef], rotation=90)

##############################################
### Layer-specific expression correlations ###
fig, axes = plt.subplots(numLayers,numLayers,figsize=(15,15))
plt.suptitle('Cross-Layer Standardized Gene Expression Correlations')
for layerIDX0 in range(numLayers):
    for layerIDX1 in range(numLayers):

        linearmodel.fit(mean_expression_standard[layerIDX0,:,:].reshape(-1,1),mean_expression_standard[layerIDX1,:,:].reshape(-1,1))
        expr_pred = linearmodel.predict(mean_expression_standard[layerIDX0,:,:].reshape(-1,1))
        L2L_r2 = r2_score(mean_expression_standard[layerIDX1,:,:].reshape(-1,1), expr_pred)
        #L2L_r2 = r2_score(mean_expression_standard[layerIDX0,:,:].reshape(-1),mean_expression_standard[layerIDX1,:,:].reshape(-1))
        
        axes[layerIDX0,layerIDX1].set_title(f'$R^2$={round(L2L_r2,3)}')
        for currentRegion in range(structNum):
            axes[layerIDX0,layerIDX1].scatter(mean_expression_standard[layerIDX0,currentRegion,:],mean_expression_standard[layerIDX1,currentRegion,:],label=structListMerge[currentRegion],color=areaColors[currentRegion],s=0.25)
            if layerIDX1 == 0:
                axes[layerIDX0,layerIDX1].set_ylabel(f"{layerNames[layerIDX0]}\nGene Expressions")
            if layerIDX0 == numLayers-1:
                axes[layerIDX0,layerIDX1].set_xlabel(f"{layerNames[layerIDX1]}\nGene Expressions")
            if (layerIDX0 == numLayers-1) and (layerIDX1 == numLayers-1):
                axes[layerIDX0,layerIDX1].legend()
plt.savefig(os.path.join(savePath,'Spatial',f'crossLayerGeneExpressionCorrelations.pdf'),dpi=600,bbox_inches='tight')
plt.close()


time_end = datetime.datetime.now()
print(f'Time to run analysis: {time_end - time_load_data}')