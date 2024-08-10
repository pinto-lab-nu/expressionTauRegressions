import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from pathlib import Path
from statistics import fmean as mean
import packages.connect_to_dj as connect_to_dj
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
from sklearn.preprocessing import OneHotEncoder
from packages.regressionUtils import *



# Connects to database and creates virtual modules
VM = connect_to_dj.get_virtual_modules()

standard_scaler = StandardScaler()
hotencoder = OneHotEncoder(sparse_output=False)


##################################
### Script Parameters and Settings
lineSelection = 'Cux2-Ai96'
#lineSelection = 'Rpb4-Ai96'
loadPilot = True

structListMerge = np.array(['MOp','MOs','VISa','VISp','VISam','VISpm','SS','RSP'])
structList = structListMerge


structNum = structList.shape[0]
applyLayerSpecificityFilter = False #ensure that CCM coordinates are contained within a layer specified in layerAppend
layerAppend = '2/3'
#groupSelector = 12  #12 -> IT_7  -> L2/3 IT
                    #4  -> IT_11 -> L4/5 IT
                    #14 -> IT_9  -> L5 IT
                    #11 -> IT_6  -> L6 IT


if applyLayerSpecificityFilter:
    structList = [x+layerAppend for x in structList]


areaColors = ['#ff0000','#ff704d',                      #MO, reds
            '#4dd2ff','#0066ff','#003cb3','#00ffff',    #VIS, blues
            '#33cc33',                                  #SSp, greens
            '#a366ff']                                  #RSP, purples



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



################################
### CCF Reference Space Creation
#see link for CCF example scripts from the allen: allensdk.readthedocs.io/en/latest/_static/examples/nb/reference_space.html
output_dir = os.path.join(savePath,'Data','nrrd25')
reference_space_key = os.path.join('annotation', 'ccf_2017')
resolution = 25
rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=Path(output_dir) / 'manifest.json')
# ID 1 is the adult mouse structure graph
tree = rspc.get_structure_tree(structure_graph_id=1)

annotation, meta = rspc.get_annotation_volume() #in browser navigate to the .nrrd file and download manually, not working automatically for some reason
# The file should be moved to the reference space key directory, only needs to be done once
os.listdir(Path(output_dir) / reference_space_key)
rsp = rspc.get_reference_space()
        


if loadPilot:
    #projectPath = r'c:\Users\lai7370\OneDrive - Northwestern University\PilotData'
    #PilotData = h5py.File(os.path.join(projectPath,'filt_neurons_fixedbent_CCF.mat'))

    PilotData = scipy.io.loadmat(os.path.join(savePath,'Data','filt_neurons_fixedbent_CCF.mat'))

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

#fn_clustid = np.load(os.path.join(projectPath,'fn_clustid.npy'))

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
    print(f'Making {structureOfInterest} mask...')
    structureTree = tree.get_structures_by_acronym([structureOfInterest])
    structureName = structureTree[0]['name']
    structureID = structureTree[0]['id']
    structure_mask = rsp.make_structure_mask([structureID])

    plt.figure()
    plt.title(f'{structureOfInterest}')
    plt.imshow(np.mean(structure_mask,axis=1))
    plt.savefig(os.path.join(savePath,f'{structureOfInterest}.pdf'),dpi=600,bbox_inches='tight')
    plt.close()

    for cell in range(fn_CCF.shape[0]):
        currentMask = structure_mask[round(fn_CCF[cell,0]),round(fn_CCF[cell,1]),round(fn_CCF[cell,2])]
        if currentMask > 0:
            cell_region[cell] = structIDX


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

# from collections import Counter
# occurrences = Counter(cell_region)


H2_all = [s[:-1] for s in fn_clustid]


regionalResample = False #resample each cortical region such that it's represented equally, in practice this tends to over-represent smaller regions
regional_resampling = 3000
#tau_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
cell_region_H2layerFiltered = [np.empty((0,1)).astype(int) for _ in range(numLayers)]
#tau_SD_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
gene_data_dense_H2layerFiltered = [np.empty((0,gene_data_dense.shape[1])) for _ in range(numLayers)]
mlCCF_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
apCCF_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
H3_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
mean_expression = np.zeros((numLayers,len(structList),gene_data_dense.shape[1]))
#sigma_expression = np.zeros((numLayers,len(structList),gene_data_dense.shape[1]))
for layerIDX,(layer,layerName) in enumerate(zip(layerIDs,layerNames)):
    for structIDX,structureOfInterest in enumerate(structList):
        layerIDXs = set([i for i, s in enumerate(H2_all) if s == grouping[layer]])
        regionIDXs = set(np.where(cell_region == structIDX)[0])
        H2layerFilter = list(layerIDXs&regionIDXs)

        print(f'layer:{layerName}, {structureOfInterest}, count: {len(H2layerFilter)}')
        print(np.unique(fn_clustid[H2layerFilter]))

        H3_values = np.array([int(item[-1]) for item in fn_clustid[H2layerFilter]])

        mean_expression[layerIDX,structIDX,:] = np.mean(gene_data_dense[H2layerFilter,:],0)
        #sigma_expression[layerIDX,structIDX,:] = np.std(gene_data_dense[H2layerFilter,:],0)

        if regionalResample: #equal representation to each cortical region
            if len(H2layerFilter) > 0: #resample with replacement
                H2layerFilter = random.choices(H2layerFilter, k=regional_resampling)
        
        mlCCF_per_cell_H2layerFiltered[layerIDX] = np.vstack((mlCCF_per_cell_H2layerFiltered[layerIDX],fn_CCF[H2layerFilter,2].reshape(-1,1)))
        apCCF_per_cell_H2layerFiltered[layerIDX] = np.vstack((apCCF_per_cell_H2layerFiltered[layerIDX],fn_CCF[H2layerFilter,0].reshape(-1,1)))
        H3_per_cell_H2layerFiltered[layerIDX] = np.vstack((H3_per_cell_H2layerFiltered[layerIDX],H3_values.reshape(-1,1)))
        #tau_per_cell_H2layerFiltered[layerIDX] = np.vstack((tau_per_cell_H2layerFiltered[layerIDX],tau_per_cell[H2layerFilter].reshape(-1,1)))
        cell_region_H2layerFiltered[layerIDX] = np.vstack((cell_region_H2layerFiltered[layerIDX],cell_region[H2layerFilter].reshape(-1,1)))
        #tau_SD_per_cell_H2layerFiltered[layerIDX] = np.vstack((tau_SD_per_cell_H2layerFiltered[layerIDX],tau_SD_per_cell[H2layerFilter].reshape(-1,1)))
        gene_data_dense_H2layerFiltered[layerIDX] = np.vstack((gene_data_dense_H2layerFiltered[layerIDX],gene_data_dense[H2layerFilter,:]))



#############################################################################################
### visualization of and calculation of high expression genes are combined here, separate ###
if my_os == 'Linux':
    meanExpressionThreshArray = [[0.4,0.2,0.1,0][int(sys.argv[1])]]
    meanH3ThreshArray = [[0.1,0.05,0.025,0][int(sys.argv[1])]]
if my_os == 'Windows':
    meanExpressionThreshArray = [0.4,0.2,0.1,0]
    meanH3ThreshArray = [0.1,0.05,0.025,0]

for meanExpressionThresh,meanH3Thresh in zip(meanExpressionThreshArray,meanH3ThreshArray):
    
    poolIndex = 0
    for tauPoolSize in [1,2,4,8,16]:
        poolIndex += 1
        tauSortedPath = os.path.join(savePath,lineSelection,f'pooling{tauPoolSize}')
        if not os.path.exists(tauSortedPath):
            os.makedirs(tauSortedPath)
        CCF_ML_Center = 227.53027784753124 #this is hard-coded CCF 'true' center, this comes from the tform_Tallen2CCF of the mean ML coordinates of bregma & lambda from five Cux mice, this should be replaced with a more robust method!!!
        allTauCCF_Coords = np.load(os.path.join(tauPath,f'{lineSelection}_tauCCF.npy'))
        ### pool tau into a grid for bootstrapping the regression ###
        minML_CCF, maxML_CCF, minAP_CCF, maxAP_CCF = np.floor(np.min(allTauCCF_Coords[0,:])), np.ceil(np.max(allTauCCF_Coords[0,:])), np.floor(np.min(allTauCCF_Coords[1,:])), np.ceil(np.max(allTauCCF_Coords[1,:]))
        pooledTauCCF_coords = [np.empty((4,0)) for _ in range(numLayers)]
        pooledTauCCF_coords_noGene = [np.empty((4,0)) for _ in range(numLayers)]
        pooledPixelCount_v_CellCount = [np.empty((2,0)) for _ in range(numLayers)]
        pooledTau_cellAligned = [np.empty((1,0)) for _ in range(numLayers)]
        pooled_cell_region_H2layerFiltered = [np.empty((0,1)).astype(int) for _ in range(numLayers)]
        resampledGenes_aligned = [np.empty((total_genes,0)) for _ in range(numLayers)]
        resampledH3_aligned_H2layerFiltered = [np.empty((1,0)) for _ in range(numLayers)]
        resampledH3_aligned_H2layerFiltered_OneHot = []
        H3_per_cell_H2layerFiltered_OneHot = []
        genePoolSaturation = []
        for layerIDX in range(numLayers):
            geneProfilePresentCount = 0
            possiblePoolsCount = 0
            print(f'Tau-Gene Alignment Pooling: {layerNames[layerIDX]}')
            for current_tau_ML_pool in np.arange(minML_CCF,CCF_ML_Center,tauPoolSize):
                current_ML_tau_pooling_IDXs = np.where(np.abs(np.abs(allTauCCF_Coords[0,:]-CCF_ML_Center)-np.abs(current_tau_ML_pool-CCF_ML_Center))<(tauPoolSize/2))[0] #our pixel space extents bilaterally, but CCF is unilateral, so 'CCF' coordinates from pixel space need to reflected over the ML center axis (CCF_ML_center)
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
                            
                            pooledTau_cellAligned[layerIDX] = np.hstack((pooledTau_cellAligned[layerIDX],pooledTaus.reshape(1,-1)))
                            
                            gene_pool_data = gene_data_dense_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                            geneResamplingIDX = random.choices(np.arange(0,gene_pool_data.shape[0]), k=pooledTaus.shape[0])
                            resampledGenes_aligned[layerIDX] = np.hstack((resampledGenes_aligned[layerIDX],gene_pool_data[geneResamplingIDX,:].reshape(total_genes,-1)))

                            H3_pool_data = H3_per_cell_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                            resampledH3_aligned_H2layerFiltered[layerIDX] = np.hstack((resampledH3_aligned_H2layerFiltered[layerIDX],H3_pool_data[geneResamplingIDX,:].reshape(1,-1)))

                            pooledPixelCount_v_CellCount[layerIDX] = np.hstack((pooledPixelCount_v_CellCount[layerIDX],np.array((pooledTaus.shape[0],gene_pool_data.shape[0])).reshape(2,-1)))

                            cell_region_pool_data = cell_region_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                            pooled_cell_region_H2layerFiltered[layerIDX] = np.vstack((pooled_cell_region_H2layerFiltered[layerIDX],cell_region_pool_data[geneResamplingIDX,:].reshape(-1,1)))

                            gene_pool_ML_CCF = mlCCF_per_cell_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs]].reshape(-1)
                            gene_pool_AP_CCF = apCCF_per_cell_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs]].reshape(-1)

                            #print(f'CCF ML:{current_tau_ML_pool}, CCF AP:{current_tau_AP_pool}, GeneX ML CCFs:{gene_pool_ML_CCF}, GeneX AP CCFs:{gene_pool_AP_CCF}')
                        else:
                            pooledTauCCF_coords_noGene[layerIDX] = np.hstack((np.array((current_tau_ML_pool,current_tau_AP_pool,np.mean(pooledTaus),np.std(pooledTaus))).reshape(-1,1),pooledTauCCF_coords_noGene[layerIDX]))
            genePoolSaturation.append(geneProfilePresentCount/possiblePoolsCount)
            resampledH3_aligned_H2layerFiltered_OneHot.append(hotencoder.fit_transform(resampledH3_aligned_H2layerFiltered[layerIDX].T))
            H3_per_cell_H2layerFiltered_OneHot.append(hotencoder.fit_transform(H3_per_cell_H2layerFiltered[layerIDX]))

        for layerIDX in range(numLayers):
            plt.figure(), plt.title(f'CCF Pooling:{tauPoolSize}, Fraction of Tau Pooled Points with at least one Gene Profile:{round(genePoolSaturation[layerIDX],3)}\n{lineSelection}, {layerNames[layerIDX]}')
            plt.scatter(pooledTauCCF_coords_noGene[layerIDX][1,:],pooledTauCCF_coords_noGene[layerIDX][0,:],color='red',s=0.5)
            plt.scatter(pooledTauCCF_coords[layerIDX][1,:],pooledTauCCF_coords[layerIDX][0,:],color='green',s=0.5)
            plt.xlabel('CCF AP'), plt.ylabel('CCF ML'), plt.axis('equal')
            plt.savefig(os.path.join(tauSortedPath,f'{lineSelection}_tauExpressionPooling{tauPoolSize}_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(1,1,figsize=(8,8))
            plt.suptitle(f'{lineSelection} Tau, CCF Pooling:{tauPoolSize}\n{lineSelection}, {layerNames[layerIDX]}')
            cmap = plt.get_cmap('cool')
            global_min,global_max = np.log10(1),np.log10(30)
            norm = matplotlib.colors.Normalize(global_min, global_max)
            tau_colors = cmap(norm(np.log10(pooledTauCCF_coords[layerIDX][2,:])))
            ax.scatter(pooledTauCCF_coords[layerIDX][1,:],pooledTauCCF_coords[layerIDX][0,:],color=tau_colors,s=6)
            ax.set_xlabel('CCF AP'), ax.set_ylabel('CCF ML'), ax.axis('equal')
            mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array(tau_colors)
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=35, pad=0.1, orientation='horizontal')
            cbar_ticks = np.arange(global_min, global_max, 1)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(10**(cbar_ticks),fontsize=6,rotation=45)
            cbar.set_label('Tau', rotation=0)
            plt.savefig(os.path.join(tauSortedPath,f'{lineSelection}_tauPooling{tauPoolSize}_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
            plt.close()

            plt.figure(), plt.xlabel('Pool Pixel Count'), plt.ylabel('Pool Cell Count')
            plt.title(f'Pool Size:{tauPoolSize}, Pixel & Cell Counts by CCF Pool\n{layerNames[layerIDX]}\n{lineSelection}, {layerNames[layerIDX]}')
            plt.scatter(pooledPixelCount_v_CellCount[layerIDX][0,:],pooledPixelCount_v_CellCount[layerIDX][1,:],color='black',s=1)
            plt.axis('equal')
            plt.savefig(os.path.join(tauSortedPath,f'{lineSelection}_pooling{tauPoolSize}_CellPixelCounts_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
            plt.close()
        
        rename = os.path.join(tauSortedPath,f'{lineSelection}_tauExpressionPooling{tauPoolSize}.pdf')
        #PDFmerger(tauSortedPath,f'{lineSelection}_tauExpressionPooling{tauPoolSize}_',layerNames,'.pdf',rename)

        rename = os.path.join(tauSortedPath,f'{lineSelection}_tauPooling{tauPoolSize}.pdf')
        #PDFmerger(tauSortedPath,f'{lineSelection}_tauPooling{tauPoolSize}_',layerNames,'.pdf',rename)

        rename = os.path.join(tauSortedPath,f'{lineSelection}_pooling{tauPoolSize}_CellPixelCounts.pdf')
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
            plt.suptitle(f'Pooling={tauPoolSize}, Standardized {lineSelection} Tau & AP, ML CCF Correlation\n{layerNames[layerIDX]}')
            plt.savefig(os.path.join(tauSortedPath,f'{lineSelection}_pooling{tauPoolSize}_TauCCFcorr_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
            plt.close()
        rename = os.path.join(tauSortedPath,f'{lineSelection}_pooling{tauPoolSize}_TauCCFcorr.pdf')
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



        mean_expression_scaler = StandardScaler()
        ######################################################################################
        ### Standard Scaler transform the gene expression and tau data prior to regression ###
        mean_expression_standard = np.zeros_like(mean_expression)
        # Tau Regressions #
        pooledTau_cellAligned_standard = [np.zeros_like(pooledTau_cellAligned[layerIDX].T) for layerIDX in range(numLayers)]
        resampledGenes_aligned_H2layerFiltered_standard = [np.zeros_like(resampledGenes_aligned[layerIDX].T) for layerIDX in range(numLayers)]
        # CCF Regressions #
        gene_data_dense_H2layerFiltered_standard = [np.zeros_like(gene_data_dense_H2layerFiltered[layerIDX]) for layerIDX in range(numLayers)]
        #tau_per_cell_H2layerFiltered_standard = [np.zeros_like(tau_per_cell_H2layerFiltered[layerIDX]) for layerIDX in range(numLayers)]
        mlCCF_per_cell_H2layerFiltered_standard = [np.zeros_like(mlCCF_per_cell_H2layerFiltered[layerIDX]) for layerIDX in range(numLayers)]
        apCCF_per_cell_H2layerFiltered_standard = [np.zeros_like(apCCF_per_cell_H2layerFiltered[layerIDX]) for layerIDX in range(numLayers)]
        for layerIDX in range(numLayers):
            mean_expression_standard[layerIDX,:,:] = mean_expression_scaler.fit_transform(mean_expression[layerIDX,:,:])
            # Tau Regressions #
            pooledTau_cellAligned_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(pooledTau_cellAligned[layerIDX][:,:]).T)
            resampledGenes_aligned_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(resampledGenes_aligned[layerIDX][:,:]).T)
            # CCF Regressions #
            gene_data_dense_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(gene_data_dense_H2layerFiltered[layerIDX][:,:]))
            #tau_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(tau_per_cell_H2layerFiltered[layerIDX][:,:]))
            mlCCF_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(mlCCF_per_cell_H2layerFiltered[layerIDX][:,:]))
            apCCF_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(apCCF_per_cell_H2layerFiltered[layerIDX][:,:]))
                
        #print(np.mean(resampledGenes_aligned_H2layerFiltered_standard[0][:,:],axis=0)) #just to see that the means are zero after standardizing



        if poolIndex == 1:
            regressionsToStart = [0,1]
            plottingConditions = [False,True]
        else:
            regressionsToStart = [0] #no need to run spatial regression multiple times across pooling sizes, just when the meanPredictionThresh changes
            plottingConditions = [False] #no need to plot spatial regression plots across "..."

        for namePredictors,predictorTitle,predictorEncodeType,predictorPathSuffix,genePredictorsCondition in zip(['Gene Predictors','H3 Predictors'],['Gene Expression','H3 Level'],['Standardized','OneHot'],['genePredictors','H3Predictors'],[True,False]):

            if not os.path.exists(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}')):
                os.makedirs(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}'))

            if genePredictorsCondition:
                predictorDataRaw = gene_data_dense_H2layerFiltered
                meanPredictionThresh = meanExpressionThresh
                predictorNamesArray = np.array(geneNames)
            else:
                predictorDataRaw = H3_per_cell_H2layerFiltered
                meanPredictionThresh = meanH3Thresh
                predictorNamesArray = np.arange(0, predictorDataRaw[layerIDX].shape[1], 1)

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
                fig, ax = plt.subplots(1,1,figsize=(15,10))
                ax.plot(layerMeanPredictors[sortedMeanPredictor],color='black')
                ax.vlines(x=meanPredictorCutoffIDX, ymin=0, ymax=np.max(layerMeanPredictors),color='black',alpha=0.5,linestyles='dashed')
                ax.hlines(y=meanPredictionThresh, xmin=0, xmax=predictorDataRaw[layerIDX].shape[1],color='black',alpha=0.5,linestyles='dashed')
                ax.set_xticks(np.arange(0, predictorDataRaw[layerIDX].shape[1], 1))
                if genePredictorsCondition:
                    ax.set_xticklabels(predictorNamesArray[sortedMeanPredictor], rotation=90)
                else:
                    ax.set_xticklabels(predictorNamesArray[sortedMeanPredictor])
                ax.set_ylabel(f'Mean {predictorTitle}')
                ax.set_yscale('log')
                ax.set_title(f'{layerNames[layerIDX]}, Num Excluded {namePredictors}:{meanPredictorCutoffIDX}, Mean {predictorTitle} Cutoff:{meanPredictionThresh}')
                plt.savefig(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                plt.close()

            rename = os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}.pdf')
            #PDFmerger(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}'),f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}_',layerNames,'.pdf',rename)


            for regressionType in regressionsToStart:
                if regressionType == 0: # geneX, H3 -> Tau
                    spatialReconstruction = False
                    tauRegression = True
                    genePredictors = genePredictorsCondition #are genes the predictor variables? If so this is handled later by selecting only high-expression genes (determined separately for each cortical layer) for the regression
                    cell_region_filtered = pooled_cell_region_H2layerFiltered
                    y_data = pooledTau_cellAligned_standard
                    pred_dim = 1
                    if genePredictors:
                        x_data = resampledGenes_aligned_H2layerFiltered_standard
                    else:
                        x_data = resampledH3_aligned_H2layerFiltered_OneHot

                if regressionType == 1: # geneX, H3 -> CCF
                    spatialReconstruction = True
                    tauRegression = False
                    genePredictors = genePredictorsCondition
                    cell_region_filtered = cell_region_H2layerFiltered
                    y_data = [np.hstack((apCCF_per_cell_H2layerFiltered_standard[layerIDX],mlCCF_per_cell_H2layerFiltered_standard[layerIDX])) for layerIDX in range(numLayers)]
                    pred_dim = 2
                    if genePredictors:
                        x_data = gene_data_dense_H2layerFiltered_standard
                    else:
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

                if regressionType == 4: # CCF <-> Tau
                    print('To Do: make regression type 2')
                    break
                
                # if ???:
                #     y_data = tau_per_cell_H2layerFiltered_standard

                regressionConditions = [spatialReconstruction,tauRegression,regionalResample,genePredictors]
                n_splits = 5
                alphaParams = [-4,0,25] #Alpha Lower (10**x), Alpha Upper (10**x), Steps

                print(f'Starting Regressions, Type {regressionType}:')
                if regressionType == 0:
                    print(f'{predictorTitle} -> {lineSelection} Tau (CCF Pooling={tauPoolSize}, predThresh={meanPredictionThresh}, regionResamp={regressionConditions[2]})')
                    best_coef_0,lasso_weight_0,bestAlpha_0,alphas_0,tauPredictions_0,bestR2_0 = layerRegressions(pred_dim,n_splits,highMeanPredictorIDXs,x_data,y_data,layerNames,regressionConditions,cell_region_filtered,alphaParams)
                
                if regressionType == 1:
                    print(f'{predictorTitle} -> CCF (predThresh={meanPredictionThresh}, regionResamp={regressionConditions[2]})')
                    best_coef_1,lasso_weight_1,bestAlpha_1,alphas_1,tauPredictions_1,bestR2_1 = layerRegressions(pred_dim,n_splits,highMeanPredictorIDXs,x_data,y_data,layerNames,regressionConditions,cell_region_filtered,alphaParams)



            mean_fold_coef_0 = [np.mean(best_coef_0[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
            sd_fold_coef_0 = [np.std(best_coef_0[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
            sorted_coef_0 = [np.argsort(mean_fold_coef_0[layerIDX]) for layerIDX in range(numLayers)]
            mean_fold_coef_1 = [np.mean(best_coef_1[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
            sd_fold_coef_1 = [np.std(best_coef_1[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
            sorted_coef_1 = [np.argsort(mean_fold_coef_1[layerIDX]) for layerIDX in range(numLayers)]







            ###################################################################
            ################### Regression Outputs Plotting ###################
            plotting = True
            numPrecision, alphaPrecision = 3, 5 #just for display
            resampTitle = f'predThresh={meanPredictionThresh}'
            for spatialReconstruction in plottingConditions: #[True,False]
                if spatialReconstruction:
                    plottingTitles = ["A-P CCF","M-L CCF"]
                    titleAppend = f'Spatial Reconstruction from {predictorTitle}, {resampTitle}'
                    tauPredictions = tauPredictions_1
                    bestR2 = bestR2_1
                    mean_fold_coef = mean_fold_coef_1
                    sorted_coef = sorted_coef_1
                    bestAlpha = bestAlpha_1
                    alphas = alphas_1
                    lasso_weight = lasso_weight_1
                    sd_fold_coef = sd_fold_coef_1
                    pred_dim = 2
                    plottingDir = os.path.join(savePath,'Spatial',f'{predictorPathSuffix}')
                else:
                    plottingTitles = ["tau"]
                    titleAppend = f'{lineSelection} Tau Reconstruction from {predictorTitle} (pooling={tauPoolSize}, {resampTitle})'
                    tauPredictions = tauPredictions_0
                    bestR2 = bestR2_0
                    mean_fold_coef = mean_fold_coef_0
                    sorted_coef = sorted_coef_0
                    bestAlpha = bestAlpha_0
                    alphas = alphas_0
                    lasso_weight = lasso_weight_0
                    sd_fold_coef = sd_fold_coef_0
                    pred_dim = 1
                    plottingDir = os.path.join(tauSortedPath,f'{predictorPathSuffix}')
                    if not os.path.exists(plottingDir):
                        os.makedirs(plottingDir)

                with open(os.path.join(plottingDir,f'regression_{titleAppend}.txt'), "w") as file:
                    file.write(f'{titleAppend}\n\n')

                for layerIDX,(layer,layerName) in enumerate(zip(layerIDs,layerNames)):

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
                        ax.set_title(f'{titleAppend} from {predictorTitle}\nA-P vs M-L $\\beta$ Values\n{layerName}, $\\alpha\pm$SD={round(bestAlpha_mean,alphaPrecision)}$\pm${round(bestAlpha_SD,alphaPrecision)}, $R^2\pm$SD={bestR2_mean}$\pm${bestR2_SD}, error:5-fold SD')
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
                    plt.suptitle(f'{titleAppend} from {predictorTitle}\n{layerName}, $\\alpha\pm$SD={round(bestAlpha_mean,alphaPrecision)}$\pm${round(bestAlpha_SD,alphaPrecision)}, $R^2\pm$SD={bestR2_mean}$\pm${bestR2_SD}, error:5-fold SD')
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

                if plotting:
                    rename = os.path.join(plottingDir,f'{predictorPathSuffix}LassoWeightsAll_{titleAppend}.pdf')
                    #PDFmerger(plottingDir,f'{predictorPathSuffix}LassoWeightsAll_',layerNames,f'_{titleAppend}.pdf',rename)

                    if spatialReconstruction:
                        rename = os.path.join(plottingDir,f'APvsML_{predictorPathSuffix}BetaWeights_{titleAppend}.pdf')
                        #PDFmerger(plottingDir,f'APvsML_{predictorPathSuffix}BetaWeights_',layerNames,f'_{titleAppend}.pdf',rename)
                    
                    rename = os.path.join(plottingDir,f'{predictorPathSuffix}LassoWeights_{titleAppend}.pdf')
                    #PDFmerger(plottingDir,f'{predictorPathSuffix}LassoWeights_',layerNames,f'_{titleAppend}.pdf',rename)

                    rename = os.path.join(plottingDir,f'regional{predictorPathSuffix}_{titleAppend}.pdf')
                    #PDFmerger(plottingDir,f'regional{predictorPathSuffix}_',layerNames,f'_{titleAppend}.pdf',rename)

                    for dim in range(pred_dim):
                        currentPlottingTitle = plottingTitles[dim]
                        rename = os.path.join(plottingDir,f'predicted{currentPlottingTitle}_{titleAppend}.pdf')
                        #PDFmerger(plottingDir,f'predicted{currentPlottingTitle}_',layerNames,f'_{titleAppend}.pdf',rename)



            for layerIDX,(layer,layerName) in enumerate(zip(layerIDs,layerNames)):
                fig, ax = plt.subplots(1,2,figsize=(15,7))
                plt.suptitle(f'{resampTitle} Spatial and {lineSelection} Tau Reconstruction from {predictorTitle}\n{lineSelection} Tau vs A-P, M-L $\\beta$ Values\n{layerName}, error:5-fold SD')
                ax[0].scatter(mean_fold_coef_0[layerIDX].reshape(-1), mean_fold_coef_1[layerIDX][0,:].reshape(-1),color='black',s=0.5)
                ax[0].errorbar(mean_fold_coef_0[layerIDX].reshape(-1), mean_fold_coef_1[layerIDX][0,:].reshape(-1), xerr=sd_fold_coef_0[layerIDX][0,:], yerr=sd_fold_coef_1[layerIDX][0,:], fmt="o", color='black')
                ax[1].scatter(mean_fold_coef_0[layerIDX].reshape(-1), mean_fold_coef_1[layerIDX][1,:].reshape(-1),color='black',s=0.5)
                ax[1].errorbar(mean_fold_coef_0[layerIDX].reshape(-1), mean_fold_coef_1[layerIDX][1,:].reshape(-1), xerr=sd_fold_coef_0[layerIDX][0,:], yerr=sd_fold_coef_1[layerIDX][1,:], fmt="o", color='black')
                for i, predictorText in enumerate(predictorNamesArray[highMeanPredictorIDXs[layerIDX]]):
                    ax[0].annotate(predictorText, (mean_fold_coef_0[layerIDX][0,i], mean_fold_coef_1[layerIDX][0,i]))
                    ax[1].annotate(predictorText, (mean_fold_coef_0[layerIDX][0,i], mean_fold_coef_1[layerIDX][1,i]))
                ax[0].set_xlabel(f'Tau $\\beta$')
                ax[0].set_ylabel(f'A-P $\\beta$')
                ax[1].set_xlabel(f'Tau $\\beta$')
                ax[1].set_ylabel(f'M-L $\\beta$')
                plt.savefig(os.path.join(tauSortedPath,f'{predictorPathSuffix}',f'{resampTitle}_{lineSelection}Tau_vs_AP&ML_Betas_{layerName}.pdf'),dpi=600,bbox_inches='tight')
                plt.close()
            rename = os.path.join(tauSortedPath,f'{predictorPathSuffix}',f'{resampTitle}_{lineSelection}Tau_vs_AP&ML_Betas.pdf')
            #PDFmerger(os.path.join(tauSortedPath,f'{predictorPathSuffix}'),f'{resampTitle}_{lineSelection}Tau_vs_AP&ML_Betas_',layerNames,'.pdf',rename)


            if poolIndex == 1:
                for layerIDX,(layer,layerName) in enumerate(zip(layerIDs,layerNames)):
                    apR2,mlR2 = [],[]
                    for foldIDX in range(n_splits):
                        apR2.append(r2_score(tauPredictions_1[layerIDX][foldIDX][:,0],tauPredictions_1[layerIDX][foldIDX][:,2]))
                        mlR2.append(r2_score(tauPredictions_1[layerIDX][foldIDX][:,1],tauPredictions_1[layerIDX][foldIDX][:,3]))
                    fig, axes = plt.subplots(1,2,figsize=(20,10))
                    plt.suptitle(f'{layerName} Cross-Fold Spatial Reconstructions from {predictorEncodeType} {predictorTitle}')
                    axes[0].set_xlabel(f'True Standardized A-P CCF'), axes[0].set_ylabel('True Standardized M-L CCF')
                    axes[1].set_xlabel(f'Predicted Standardized A-P CCF\n$R^2$={round(np.mean(apR2),3)}'), axes[1].set_ylabel(f'Predicted Standardized M-L CCF\n$R^2$={round(np.mean(mlR2),3)}')
                    for regionIDX,region in enumerate(structList):
                        for foldIDX in range(n_splits):
                            regionR2IDXs = np.where(tauPredictions_1[layerIDX][foldIDX][:,-1] == regionIDX)
                            axes[0].scatter(tauPredictions_1[layerIDX][foldIDX][regionR2IDXs,0],tauPredictions_1[layerIDX][foldIDX][regionR2IDXs,1],color=areaColors[regionIDX],s=0.15)
                            axes[1].scatter(tauPredictions_1[layerIDX][foldIDX][regionR2IDXs,2],tauPredictions_1[layerIDX][foldIDX][regionR2IDXs,3],color=areaColors[regionIDX],s=0.15)
                            #axes[0].axis('equal')
                            #axes[1].axis('equal')
                            for axnum in range(2):
                                axes[axnum].set_xlim(-2.5,2.5), axes[axnum].set_ylim(-2.5,2.5)
                    plt.savefig(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'{predictorPathSuffix}Thresh{meanPredictionThresh}_spatialReconstruction_{layerName}.pdf'),dpi=600,bbox_inches='tight')
                    plt.close()
                rename = os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'{predictorPathSuffix}Thresh{meanPredictionThresh}_spatialReconstruction.pdf')
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


