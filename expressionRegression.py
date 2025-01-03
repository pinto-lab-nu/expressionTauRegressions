import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import sys
import os
import numpy as np
import random
from random import choices
import pandas as pd
import pickle
#from statsmodels.regression.linear_model import WLS
#from sklearn.preprocessing import OneHotEncoder
from packages.regressionUtils import *
from packages.dataloading import *
from packages.functional_dataset import *
from packages.plotting_util import *
from collections import Counter
import datetime
import re
import argparse


def string_sanitizer(input_string):
        sanitized_string = re.sub(r'\/','_',input_string)
        return sanitized_string

def plot_mask(structureOfInterest,structure_mask,savePath,resolution):
    plt.figure()
    plt.title(f'{structureOfInterest}')
    plt.imshow(np.mean(structure_mask[:,:,:],axis=1)) #restrict to ~40 voxels along axis 1 to get dorsal cortex (for 25 rez), this is just for viz
    plt.savefig(os.path.join(savePath,'Masks',f'{structureOfInterest}_{resolution}.pdf'),dpi=600,bbox_inches='tight')
    plt.close()

def cell_region_function(cell_region,cell_layer,resolution,structure_mask,CCFvalues,CCFindexOrder,CCFmultiplier,structIDX,layerIDX):
    for cell in range(cell_region[resolution].shape[0]):
        currentMask = structure_mask[round(CCFvalues[cell,CCFindexOrder[0]]*CCFmultiplier),round(CCFvalues[cell,CCFindexOrder[1]]*CCFmultiplier),round(CCFvalues[cell,CCFindexOrder[2]]*CCFmultiplier)]
        if currentMask > 0:
            cell_region[resolution][cell] = structIDX
            if resolution == '10':
                cell_layer[resolution][cell] = layerIDX

    return cell_region,cell_layer

def region_count_printer(cell_region,resolution,datasetName,structList):
    regionalCounts = Counter(cell_region[resolution])
    print(f'Regional Cell Counts, {datasetName}:')
    for structIDX in range(len(structList)):
        print(f'{structList[structIDX]}:{regionalCounts[structIDX]}')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lineSelection", choices=["Cux2-Ai96", "Rpb4-Ai96"], default="Cux2-Ai96") #select the functional dataset for tau regressions
    parser.add_argument("--geneLimit", type=int, default=-1) #for testing purposes to load a subset of merfish-imputed data, set to -1 to include all genes
    parser.add_argument("--restrict_merfish_imputed_values", type=bool, default=False) #condition to restrict merfish-imputed dataset to non-imputed genes
    parser.add_argument("--tauPoolSizeArrayFull", default=[4.1]) #[1,2,3,4,5] #in 25um resolution CCF voxels, converted to mm later
    parser.add_argument("--n_splits", type=int, default=5) #number of splits for cross-validations in regressions
    parser.add_argument("--alphaParams", default=[-5, 0, 30]) # [Alpha Lower (10**x), Alpha Upper (10**x), Steps]... alpha values for Lasso regressions
    parser.add_argument("--loadData", type=bool, default=True)
    parser.add_argument("--plotting", type=bool, default=True)
    parser.add_argument("--numPrecision", type=int, default=3)   # Just for display (in plotting and regression text files)
    parser.add_argument("--alphaPrecision", type=int, default=5) # Just for display (in plotting and regression text files)
    parser.add_argument("--verbose", type=bool, default=True)    # For print statements
    parser.add_argument("--predictorOrder", default=[0])           # Select predictors for regressions, and order [0:merfish{-imputed}, 1:pilot]
    parser.add_argument("--regressionsToStart", default=[0, 1])    # Select response variables for regressions, and order [0:tau, 1:CCF]
    parser.add_argument("--max_iter", type=int, default=200) # For layer regressions
    parser.add_argument("--variableManagement", type=bool, default=True) # Removes large variables from memory after use (needs to be expamnded to include more variables)
    parser.add_argument("--plottingConditions", default=[False, True]) # For plotting spatial reconstructions
    parser.add_argument("--arg_parse_test", type=bool, default=False) # For testing the bash argument parser
    parser.add_argument("--job_task_id", type=int, default=0) # For parallel processing
    args = parser.parse_args()

    lineSelection = args.lineSelection
    geneLimit = args.geneLimit
    restrict_merfish_imputed_values = args.restrict_merfish_imputed_values
    tauPoolSizeArrayFull = list(args.tauPoolSizeArrayFull)
    n_splits = args.n_splits
    alphaParams = list(args.alphaParams)
    loadData = args.loadData
    plotting = args.plotting
    numPrecision = args.numPrecision
    alphaPrecision = args.alphaPrecision
    verbose = args.verbose
    predictorOrder = list(args.predictorOrder)
    regressionsToStart = list(args.regressionsToStart)
    max_iter = args.max_iter
    variableManagement = args.variableManagement
    plottingConditions = list(args.plottingConditions)
    arg_parse_test = args.arg_parse_test
    job_task_id = args.job_task_id

    print(f"lineSelection: {lineSelection}")
    print(f"geneLimit: {geneLimit}")
    print(f"restrict_merfish_imputed_values: {restrict_merfish_imputed_values}")
    print(f"tauPoolSizeArrayFull: {tauPoolSizeArrayFull}")
    print(f"n_splits: {n_splits}")
    print(f"alphaParams: {alphaParams}")
    print(f"loadData: {loadData}")
    print(f"plotting: {plotting}")
    print(f"numPrecision: {numPrecision}")
    print(f"alphaPrecision: {alphaPrecision}")
    print(f"verbose: {verbose}")
    print(f"predictorOrder: {predictorOrder}")
    print(f"regressionsToStart: {regressionsToStart}")
    print(f"max_iter: {max_iter}")
    print(f"variableManagement: {variableManagement}")
    print(f"plottingConditions: {plottingConditions}")
    print(f"arg_parse_test: {arg_parse_test}")
    print(f"job_task_id: {job_task_id}")

    if arg_parse_test:
        return

    for restrict_merfish_imputed_values, predictorOrder in zip([True,False],[[0,1],[0]]):

        structList = np.array(['MOp','MOs','VISa','VISp','VISam','VISpm','SS','RSP'])

        areaColors = ['#ff0000','#ff704d',                      #MO, reds
                    '#4dd2ff','#0066ff','#003cb3','#00ffff',    #VIS, blues
                    '#33cc33',                                  #SSp, greens
                    '#a366ff']                                  #RSP, purples


        structNum = structList.shape[0]
        #applyLayerSpecificityFilter = False #ensure that CCM coordinates are contained within a layer specified in layerAppend
        #layerAppend = '2/3'
        #groupSelector = 12  #12 -> IT_7  -> L2/3 IT
                            #4  -> IT_11 -> L4/5 IT
                            #14 -> IT_9  -> L5 IT
                            #11 -> IT_6  -> L6 IT


        #if applyLayerSpecificityFilter:
        #    structList = [x+layerAppend for x in structList]


        lineSelection, my_os, savePath_OSs, download_base = pathSetter(lineSelection) # Line selection is modified if the script is run on Linux

        savePath = savePath_OSs[my_os == 'Windows']

        time_start = datetime.datetime.now()

        standard_scaler = StandardScaler()
        #hotencoder = OneHotEncoder(sparse_output=False)

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
            merfish_CCF_Genes, allMerfishGeneNames = merfishLoader(savePath,download_base,pilotGeneNames,restrict_merfish_imputed_values,geneLimit)
            allTauCCF_Coords, CCF25_bregma, CCF25_lambda = load_tau_CCF(lineSelection, 'IntoTheVoid')

        time_load_data = datetime.datetime.now()
        print(f'Time to load data: {time_load_data - time_start}')

        if plotting:
            # Plot CCF-registered Tau dataset
            fig, ax = plt.subplots(1, 1, figsize=(7,5))
            #hex_0 = '#00ffff'
            #hex_1 = '#ff33cc'
            #value_0 = np.percentile(allTauCCF_Coords[2,:], 0)
            #value_1 = np.percentile(allTauCCF_Coords[2,:], 97.5)
            #all_tau_colors = color_gradient(allTauCCF_Coords[2,:], hex_0, hex_1, 0, 97.5)
            
            img = ax.scatter(allTauCCF_Coords[0,:], allTauCCF_Coords[1,:], s=0.1, c=allTauCCF_Coords[2,:], cmap='cool')
            ax.scatter(CCF25_bregma[0], CCF25_bregma[1], color='blue', label='bregma')
            ax.scatter(CCF25_lambda[0], CCF25_lambda[1], color='purple', label='lambda')
            ax.set_xlabel(r'$L \leftarrow M \rightarrow L$ CCF')
            ax.set_ylabel(r'$A \leftrightarrow P$ CCF')
            ax.set_title(r'25um CCF-Registered $\tau$ Dataset')
            ax.legend()
            cbar = plt.colorbar(img, ax=ax)
            cbar.set_label(r'$\tau$')

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
        merfishLayerNames = ['L2_3 IT_ET','L4_5 IT_ET','L6 IT_ET'] # 'L4_5 IT_ET', 'L5 IT_ET', 'L6 IT_ET'] #['CTX IT, ET']
        #merfish_layers = ['2/3','4/5','6']
        merfish_subLayers = [['2/3'],['4','5'],['6a','6b']]

        pilotLayerNames  =  ['L2_3 IT',   'L4_5 IT',  'L5 IT',    'L6 IT',    'L5 ET']
        layerIDs    =       [12,          4,          14,         11,         17]
        #numLayers = len(layerIDs)


        # ### Testing ###
        # for structIDX,structureOfInterest in enumerate(structList):
        #     if structIDX > -1:
        #         structureOfInterestAppend = structureOfInterest + 'agl6a'
        #         print(tree.get_structures_by_acronym([structureOfInterestAppend]))


        if not os.path.exists(os.path.join(savePath,'Masks')):
            os.makedirs(os.path.join(savePath,'Masks'))

        cell_region = {}
        cell_region['10'] = (np.ones(numMerfishCells)*-1).astype(int)
        cell_region['25'] = (np.ones(fn_CCF.shape[0])*-1).astype(int)
        cell_layer = {}
        cell_layer['10'] = (np.ones(numMerfishCells)*-1).astype(int)
        cell_layer['25'] = np.empty(0)

        if restrict_merfish_imputed_values:
            merfish_datasetName_append = ''
        else:
            merfish_datasetName_append = '-Imputed'

        for resolution,datasetName in zip(['10','25'],['Merfish'+merfish_datasetName_append,'Pilot']):
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
                for layerIDX,layerAppend in enumerate(merfishLayerNames):
                    print('\n')

                    for structIDX,structureOfInterest in enumerate(structList):
                        print(f'Making {layerAppend} {structureOfInterest} CCF mask (resolution={resolution})...')
                        
                        if structureOfInterest == 'RSP':
                            structure_mask = np.zeros((maskDim0,maskDim1,maskDim2))
                            for subLayerAppend in merfish_subLayers[layerIDX]:
                                if subLayerAppend != '4':
                                    for subRSP in ['v','d','agl']:
                                        structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest+subRSP+subLayerAppend])
                                        structureID = structureTree[0]['id']
                                        structure_mask += rsp[f'{resolution}'].make_structure_mask([structureID])
                        
                        if structureOfInterest == 'SS':
                            structure_mask = np.zeros((maskDim0,maskDim1,maskDim2))
                            for subLayerAppend in merfish_subLayers[layerIDX]:
                                for subSS in ['p-n','p-bfd','p-ll','p-m','p-ul','p-tr','p-un','s']:
                                    structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest+subSS+subLayerAppend])
                                    structureID = structureTree[0]['id']
                                    structure_mask += rsp[f'{resolution}'].make_structure_mask([structureID])

                        if (structureOfInterest != 'SS') and (structureOfInterest != 'RSP'):
                            structure_mask = np.zeros((maskDim0,maskDim1,maskDim2))
                            for subLayerAppend in merfish_subLayers[layerIDX]:
                                if not((subLayerAppend == '4') and (structureOfInterest == 'MOp')) and not((subLayerAppend == '4') and (structureOfInterest == 'MOs')):
                                    structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest+subLayerAppend])
                                    structureID = structureTree[0]['id']
                                    structure_mask += rsp[f'{resolution}'].make_structure_mask([structureID])

                        plot_mask(f'{structureOfInterest} {string_sanitizer(layerAppend)}',structure_mask,savePath,resolution)
                        cell_region,cell_layer = cell_region_function(cell_region,cell_layer,resolution,structure_mask,CCFvalues,CCFindexOrder,CCFmultiplier,structIDX,layerIDX)
                    region_count_printer(cell_region,resolution,f'{datasetName} {string_sanitizer(layerAppend)}',structList)
            
            if resolution == '25':
                print(f'\n(no layer masking required, as it is already present in dataset)')
                for structIDX,structureOfInterest in enumerate(structList):
                    print(f'Making {structureOfInterest} CCF mask (resolution={resolution})...')

                    structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest])
                    structureName = structureTree[0]['name']
                    structureID = structureTree[0]['id']
                    structure_mask = rsp[f'{resolution}'].make_structure_mask([structureID])

                    plot_mask(structureOfInterest,structure_mask,savePath,resolution)
                    cell_region,cell_layer = cell_region_function(cell_region,cell_layer,resolution,structure_mask,CCFvalues,CCFindexOrder,CCFmultiplier,structIDX,layerIDX)
                region_count_printer(cell_region,resolution,datasetName,structList)



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

            for layerIDX, (layer, layerName) in enumerate(zip(layerIDs, layerNames)):

                if resolution == '25':
                    layerIDXs = set([i for i, s in enumerate(H2_all) if s == grouping[layer]])
                if resolution == '10':
                    layerIDXs = set(np.where(cell_layer['10'] == layerIDX)[0]) #set(np.where(cell_region['10'] > -1)[0])
                
                if layerIDX == 0:
                    if resolution == '25':
                        cux2columnIDX = np.where(np.array(pilotGeneNames)=='Cux2')[0][0]
                    if resolution == '10':
                        if geneLimit == -1:
                            cux2columnIDX = np.where(np.array(enrichedGeneNames)=='Cux2')[0][0]
                    if geneLimit == -1:
                        cux2IDXs = set(np.where(gene_data[:,cux2columnIDX]>0)[0]) #for layer 2/3 filter out cells not expressing Cux2 (helps to align population to the functional dataset)

                for structIDX,structureOfInterest in enumerate(structList):
                    regionIDXs = set(np.where(cell_region[resolution] == structIDX)[0])
                    
                    H2layerFilter = layerIDXs & regionIDXs
                    if (layerIDX == 0) and (geneLimit == -1):
                        H2layerFilter = H2layerFilter & cux2IDXs
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



        CCF25_ML_Center = CCF25_bregma[0] #227.53027784753124 #this is hard-coded CCF 'true' center (in CCF 25 resolution), this comes from the tform_Tallen2CCF of the mean ML coordinates of bregma & lambda from five Cux mice, this should be replaced with a more robust method!!!
        CCF_ML_Center_mm = CCF25_ML_Center * 0.025
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
        #allTauCCF_Coords = np.load(os.path.join(tauPath,f'{lineSelection}_tauCCF.npy'))
        allTauCCF_Coords[0,:] *= 0.025 #convert 25um resolution functional-registered coordinates to mm
        allTauCCF_Coords[1,:] *= 0.025 #convert "..."

        #meanExpressionThreshArrayFull = [0] #[0.4,0.2,0.1,0]
        #meanH3ThreshArrayFull = [0] #[0.1,0.05,0.025,0]

        if my_os == 'Linux':
            #meanExpressionThreshArray = [meanExpressionThreshArrayFull[int(sys.argv[1])]] #batch job will distribute parameter instances among jobs run in parallel
            #meanH3ThreshArray = [meanH3ThreshArrayFull[int(sys.argv[1])]]                 #same for these regressions, just with a different parameter range
            tauPoolSizeArray = [tauPoolSizeArrayFull[int(job_task_id)]]
        if my_os == 'Windows':
            #meanExpressionThreshArray = meanExpressionThreshArrayFull
            #meanH3ThreshArray = meanH3ThreshArrayFull
            tauPoolSizeArray = tauPoolSizeArrayFull

        meanExpressionThresh,meanH3Thresh = 0,0

        layerNamesList  = [merfishLayerNames,pilotLayerNames]
        numLayersList   = [len(merfishLayerNames),len(pilotLayerNames)]
        resolutionList  = ['10','25']
        datasetNameList = ['Merfish'+merfish_datasetName_append,'Pilot']

        for layerNames,numLayers,resolution,datasetName in zip([layerNamesList[order] for order in predictorOrder],[numLayersList[order] for order in predictorOrder],[resolutionList[order] for order in predictorOrder],[datasetNameList[order] for order in predictorOrder]):

            #for meanExpressionThresh,meanH3Thresh in zip(meanExpressionThreshArray,meanH3ThreshArray):
            
            poolIndex = 0
            for tauPoolSize in tauPoolSizeArray:
                tauPoolSize = np.round(tauPoolSize * 0.025, 4) #convert 25um resolution CCF functional-registered coordinates to mm
                poolIndex += 1
                tauSortedPath = os.path.join(savePath,lineSelection,f'pooling{tauPoolSize}mm')
                tauSortedPath_OSs = [os.path.join(savePath_OSs[0],lineSelection,f'pooling{tauPoolSize}mm'), os.path.join(savePath_OSs[1],lineSelection,f'pooling{tauPoolSize}mm')]
                if not os.path.exists(tauSortedPath):
                    os.makedirs(tauSortedPath)
                
                ### pool tau into a grid for bootstrapping the regression ###
                minML_CCF, maxML_CCF, minAP_CCF, maxAP_CCF = np.min(allTauCCF_Coords[0,:]), np.max(allTauCCF_Coords[0,:]), np.min(allTauCCF_Coords[1,:]), np.max(allTauCCF_Coords[1,:])
                pooledTauCCF_coords = [np.empty((0,4)) for _ in range(numLayers)]
                pooledTauCCF_coords_noGene = [np.empty((0,4)) for _ in range(numLayers)]
                pooledPixelCount_v_CellCount = [np.empty((0,2)) for _ in range(numLayers)]
                resampledTau_aligned = [np.empty((0,1)) for _ in range(numLayers)]
                tau_aligned_forH3 = [np.empty((0,1)) for _ in range(numLayers)]
                pooled_cell_region_geneAligned_H2layerFiltered = [np.empty((0,1)).astype(int) for _ in range(numLayers)]
                pooled_region_label_alignedForTau = [np.empty((0,1)).astype(int) for _ in range(numLayers)]
                total_genes = gene_data_dense_H2layerFiltered[resolution][0].shape[1]
                resampledGenes_aligned = [np.empty((0,total_genes)) for _ in range(numLayers)]
                resampledH3_aligned_H2layerFiltered = [np.empty((0,9)) for _ in range(numLayers)] #[np.empty((1,0)) for _ in range(numLayers)]
                pooledH3_for_spatial = [np.empty((0,9)) for _ in range(numLayers)]
                if resolution == '25':
                    pooled_region_label = [np.empty((0,1)) for _ in range(numLayers)]
                #resampledH3_aligned_H2layerFiltered_OneHot = []
                #H3_per_cell_H2layerFiltered_OneHot = []
                genePoolSaturation = []
                print('\n')
                for layerIDX in range(numLayers):
                    geneProfilePresentCount = 0
                    possiblePoolsCount = 0
                    print(f'Tau-Gene Alignment Pooling ({datasetName}, size {tauPoolSize}mm): {layerNames[layerIDX]}')
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
                            pooledTaus = allTauCCF_Coords[2,current_ML_tau_pooling_IDXs[current_tau_pooling_IDXs]].reshape(-1,1)
                            if pooledTaus.shape[0] > 0:
                                #print(mlCCF_per_cell_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs])
                                possiblePoolsCount += 1
                                if current_cell_pooling_IDXs.shape[0] > 0:
                                    #print(current_tau_ML_pool,current_tau_AP_pool)
                                    geneProfilePresentCount += 1

                                    pooledTauCCF_coords[layerIDX] = np.vstack((pooledTauCCF_coords[layerIDX], np.array((current_tau_ML_pool,current_tau_AP_pool,np.mean(pooledTaus),np.std(pooledTaus))).reshape(1,4))) #switched order
                                    
                                    #######################################################
                                    ### Alignment of functional and expression datasets ###
                                    pool_resample_size = current_cell_pooling_IDXs.shape[0] #<- resample to the size of the expression data (sets expression dataset as the bottleneck)
                                    
                                    tauResamplingIDX = random.choices(np.arange(0,pooledTaus.shape[0]), k=pool_resample_size)
                                    resampledTau_aligned[layerIDX] = np.vstack((resampledTau_aligned[layerIDX],pooledTaus[tauResamplingIDX,:].reshape(-1,1)))
                                    tau_aligned_forH3[layerIDX] = np.vstack((tau_aligned_forH3[layerIDX],pooledTaus.reshape(-1,1))) #old, when expression (below) was resampled to size of functional dataset at pool (k=pooledTaus.shape[0]), this is kept for H3 regressions, which shouldn't be resampled to match th size of the expression dataset
                                    
                                    gene_pool_data = gene_data_dense_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                                    geneResamplingIDX = random.choices(np.arange(0,gene_pool_data.shape[0]), k=pool_resample_size)
                                    resampledGenes_aligned[layerIDX] = np.vstack((resampledGenes_aligned[layerIDX],gene_pool_data[geneResamplingIDX,:].reshape(-1,total_genes)))
                                    ### Alignment handled ###
                                    #########################

                                    if resolution == '25':
                                        H3_pool_data = H3_per_cell_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                                        
                                        data_flattened = H3_pool_data.flatten()
                                        categories = np.arange(1, 10)
                                        counts = np.array([np.sum(data_flattened == category) for category in categories])
                                        normalized_counts = (counts / counts.sum()).reshape(-1,9)

                                        H3ResamplingIDX = random.choices(np.arange(0,1), k=pooledTaus.shape[0])

                                        #resampledH3_aligned_H2layerFiltered[layerIDX] = np.hstack((resampledH3_aligned_H2layerFiltered[layerIDX],H3_pool_data[geneResamplingIDX,:].reshape(1,-1)))
                                        resampledH3_aligned_H2layerFiltered[layerIDX] = np.vstack((resampledH3_aligned_H2layerFiltered[layerIDX],normalized_counts[H3ResamplingIDX,:].reshape(-1,9)))
                                        pooledH3_for_spatial[layerIDX] = np.vstack((pooledH3_for_spatial[layerIDX],normalized_counts.reshape(-1,9)))

                                        pool_region_label = cell_region_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:][0][0].reshape(1,-1)
                                        pooled_region_label[layerIDX] = np.vstack((pooled_region_label[layerIDX],pool_region_label.reshape(-1,1))) ###finish to correctly label brain region for H3 regressions

                                        pooled_region_label_alignedForTau[layerIDX] = np.vstack((pooled_region_label_alignedForTau[layerIDX],pool_region_label[H3ResamplingIDX,:].reshape(-1,1)))

                                    pooledPixelCount_v_CellCount[layerIDX] = np.vstack((pooledPixelCount_v_CellCount[layerIDX],np.array((pooledTaus.shape[0],gene_pool_data.shape[0])).reshape(-1,2)))

                                    cell_region_pool_data = cell_region_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                                    pooled_cell_region_geneAligned_H2layerFiltered[layerIDX] = np.vstack((pooled_cell_region_geneAligned_H2layerFiltered[layerIDX],cell_region_pool_data[geneResamplingIDX,:].reshape(-1,1)))

                                    #gene_pool_ML_CCF = mlCCF_per_cell_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs]].reshape(-1)
                                    #gene_pool_AP_CCF = apCCF_per_cell_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs]].reshape(-1)

                                    #print(f'CCF ML:{current_tau_ML_pool}, CCF AP:{current_tau_AP_pool}, GeneX ML CCFs:{gene_pool_ML_CCF}, GeneX AP CCFs:{gene_pool_AP_CCF}')
                                else:
                                    pooledTauCCF_coords_noGene[layerIDX] = np.vstack((pooledTauCCF_coords_noGene[layerIDX], np.array((current_tau_ML_pool,current_tau_AP_pool,np.mean(pooledTaus),np.std(pooledTaus))).reshape(1,4)))
                    genePoolSaturation.append(geneProfilePresentCount/possiblePoolsCount)
                    # if resolution == '25':
                    #     resampledH3_aligned_H2layerFiltered_OneHot.append(hotencoder.fit_transform(resampledH3_aligned_H2layerFiltered[layerIDX].T))
                    #     H3_per_cell_H2layerFiltered_OneHot.append(hotencoder.fit_transform(H3_per_cell_H2layerFiltered[resolution][layerIDX]))

                for layerIDX in range(numLayers):
                    plt.figure(), plt.title(f'CCF Pooling:{tauPoolSize}mm, Fraction of Tau Pooled Points with at least one Gene Profile:{round(genePoolSaturation[layerIDX],3)}\n{lineSelection}, {layerNames[layerIDX]}')
                    plt.scatter(pooledTauCCF_coords_noGene[layerIDX][:,1],pooledTauCCF_coords_noGene[layerIDX][:,0],color='red',s=0.5)
                    plt.scatter(pooledTauCCF_coords[layerIDX][:,1],pooledTauCCF_coords[layerIDX][:,0],color='green',s=0.5)
                    plt.xlabel(r'A$\leftrightarrow$P (mm)'), plt.ylabel(r'L$\leftrightarrow$M (mm)'), plt.axis('equal')
                    plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{lineSelection}_tauExpressionPooling{tauPoolSize}mm_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                    plt.close()

                    fig, ax = plt.subplots(1,1,figsize=(8,8))
                    plt.suptitle(f'{lineSelection} Tau, CCF Pooling:{tauPoolSize}mm\n{lineSelection}, {layerNames[layerIDX]}')
                    cmap = plt.get_cmap('cool')
                    global_min,global_max = np.log10(1),np.log10(30)
                    norm = matplotlib.colors.Normalize(global_min, global_max)
                    tau_colors = cmap(norm(np.log10(pooledTauCCF_coords[layerIDX][:,2])))
                    ax.scatter(pooledTauCCF_coords[layerIDX][:,1],pooledTauCCF_coords[layerIDX][:,0],color=tau_colors,s=6)
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
                    plt.scatter(pooledPixelCount_v_CellCount[layerIDX][:,0],pooledPixelCount_v_CellCount[layerIDX][:,1],color='black',s=1)
                    #plt.axis('equal')
                    plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{lineSelection}_pooling{tauPoolSize}mm_CellPixelCounts_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                    plt.close()



                standardized_CCF_Tau = [standard_scaler.fit_transform(pooledTauCCF_coords[layerIDX]) for layerIDX in range(numLayers)]
                for layerIDX in range(numLayers):
                    standardized_CCF_Tau[layerIDX][:,1] *= -1 #invert AP CCF for regressions
                linearmodel = LinearRegression()
                for layerIDX in range(numLayers):
                    
                    r_squared_regression = []
                    for dim in range(2):
                        linearmodel.fit(standardized_CCF_Tau[layerIDX][:,dim].reshape(-1,1),standardized_CCF_Tau[layerIDX][:,2].reshape(-1))
                        tau_pred = linearmodel.predict(standardized_CCF_Tau[layerIDX][:,dim].reshape(-1,1))
                        r_squared_regression.append(r2_score(standardized_CCF_Tau[layerIDX][:,2].reshape(-1,1), tau_pred))

                    fig, ax = plt.subplots(1,2,figsize=(8,4))
                    ax[0].scatter(standardized_CCF_Tau[layerIDX][:,0],standardized_CCF_Tau[layerIDX][:,2],color='black',s=1)
                    ax[1].scatter(standardized_CCF_Tau[layerIDX][:,1],standardized_CCF_Tau[layerIDX][:,2],color='black',s=1)
                    ax[0].set_title(f'$R^2$={round(r_squared_regression[0],3)}'), ax[1].set_title(f'$R^2$={round(r_squared_regression[1],3)}')
                    ax[0].set_xlabel('Standardized ML CCF'), ax[1].set_xlabel('Standardized AP CCF')
                    ax[0].set_ylabel('Standardized Tau'), ax[1].set_ylabel('Standardized Tau')
                    plt.suptitle(f'Pooling={tauPoolSize}mm, Standardized {lineSelection} Tau & AP, ML CCF Correlation\n{layerNames[layerIDX]}')
                    plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{lineSelection}_pooling{tauPoolSize}mm_TauCCFcorr_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                    plt.close()



                ######################################################################################
                ### Standard Scaler transform the gene expression and tau data prior to regression ###
                mean_expression_standard = np.zeros_like(mean_expression[resolution])
                # Tau Regressions #
                resampledTau_aligned_standard = [np.zeros_like(resampledTau_aligned[layerIDX]) for layerIDX in range(numLayers)]
                tau_aligned_forH3_standard = [np.zeros_like(tau_aligned_forH3[layerIDX]) for layerIDX in range(numLayers)]
                resampledGenes_aligned_H2layerFiltered_standard = [np.zeros_like(resampledGenes_aligned[layerIDX]) for layerIDX in range(numLayers)]
                # CCF Regressions #
                gene_data_dense_H2layerFiltered_standard = [np.zeros_like(gene_data_dense_H2layerFiltered[resolution][layerIDX]) for layerIDX in range(numLayers)]
                #tau_per_cell_H2layerFiltered_standard = [np.zeros_like(tau_per_cell_H2layerFiltered[layerIDX]) for layerIDX in range(numLayers)]
                mlCCF_per_cell_H2layerFiltered_standard = [np.zeros_like(mlCCF_per_cell_H2layerFiltered[resolution][layerIDX]) for layerIDX in range(numLayers)]
                apCCF_per_cell_H2layerFiltered_standard = [np.zeros_like(apCCF_per_cell_H2layerFiltered[resolution][layerIDX]) for layerIDX in range(numLayers)]
                for layerIDX in range(numLayers):
                    ## Tau ##
                    resampledTau_aligned_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(resampledTau_aligned[layerIDX][:,:]))
                    tau_aligned_forH3_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(tau_aligned_forH3[layerIDX][:,:]))
                    #tau_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(tau_per_cell_H2layerFiltered[layerIDX][:,:]))
                    ## Genes ##
                    mean_expression_standard[layerIDX,:,:] = standard_scaler.fit_transform(mean_expression[resolution][layerIDX,:,:])
                    resampledGenes_aligned_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(resampledGenes_aligned[layerIDX][:,:]))
                    gene_data_dense_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(gene_data_dense_H2layerFiltered[resolution][layerIDX][:,:]))
                    # CCF #
                    mlCCF_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(mlCCF_per_cell_H2layerFiltered[resolution][layerIDX][:,:]))
                    apCCF_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(apCCF_per_cell_H2layerFiltered[resolution][layerIDX][:,:])) * -1 #Standardized AP CCF is inverted for regressions
                        
                #print(np.mean(resampledGenes_aligned_H2layerFiltered_standard[0][:,:],axis=0)) #just to see that the means are zero after standardizing


                """ 
                if poolIndex == 1:
                    regressionsToStart = [0,1]
                    plottingConditions = [False,True] #plot spatial reconstruction?
                else:
                    regressionsToStart = [0,1] #[0] #used to be no need to run spatial regression multiple times across pooling sizes, just when the meanPredictionThresh changes, but this now matters for H3 profiles per pool
                    plottingConditions = [False]#[False] #no need to plot spatial regression plots across "..."
                """

                for namePredictors,predictorTitle,predictorEncodeType,predictorPathSuffix in zip(['Gene Predictors', 'H3 Predictors'],
                                                                                                ['Gene Expression',  'H3 Level'],
                                                                                                ['Standardized',     'Normalized'],
                                                                                                ['GenePredictors',   'H3Predictors']):
                    
                    if (predictorPathSuffix == 'H3Predictors') and ((datasetName == 'Merfish-Imputed') or (datasetName == 'Merfish')):
                        print(f'\nAborting H3 Predictor regressions for Merfish{merfish_datasetName_append} dataset...')
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
                        predictorDataRaw = resampledH3_aligned_H2layerFiltered#[m.T for m in resampledH3_aligned_H2layerFiltered] #H3_per_cell_H2layerFiltered
                        meanPredictionThresh = meanH3Thresh
                        predictorNamesArray = np.arange(1, predictorDataRaw[layerIDX].shape[1]+1, 1)
                        numLayers = len(layerIDs)
                        layerNames = pilotLayerNames
                    
                    # if predictorPathSuffix == 'merfishImputedGenePredictors':
                    #     predictorDataRaw = [raw_merfish_genes]
                    #     meanPredictionThresh = 0.1
                    #     predictorNamesArray = np.array(enrichedGeneNames)
                    #     numLayers = 1
                    #     layerNames = merfishLayerNames

                    if predictorPathSuffix == 'merfishImputedGenePredictors':
                        figWidth = 23
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
                        ax.set_title(f'{layerNames[layerIDX]}, Num Excluded {namePredictors}:{meanPredictorCutoffIDX}, Mean {datasetName} {predictorTitle} Cutoff:{meanPredictionThresh}')
                        if predictorPathSuffix == 'H3Predictors':
                            plt.savefig(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                        else:
                            plt.savefig(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'{datasetName}',f'excluded{datasetName}{predictorPathSuffix}Thresh{meanPredictionThresh}_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                        plt.close()

                    rename = os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}.pdf')
                    #PDFmerger(os.path.join(savePath,'Spatial',f'{predictorPathSuffix}'),f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}_',layerNames,'.pdf',rename)

                    if variableManagement:
                        del predictorDataRaw


                    for regressionType in regressionsToStart:
                        if regressionType == 0: # geneX, H3 -> Tau
                            spatialReconstruction = False
                            tauRegression = True
                            response_dim = 1
                            if predictorPathSuffix == 'GenePredictors':
                                x_data = resampledGenes_aligned_H2layerFiltered_standard
                                y_data = resampledTau_aligned_standard
                                region_label_filtered = pooled_cell_region_geneAligned_H2layerFiltered
                            if predictorPathSuffix == 'H3Predictors':
                                x_data = resampledH3_aligned_H2layerFiltered#[m.T for m in resampledH3_aligned_H2layerFiltered]
                                y_data = tau_aligned_forH3_standard
                                region_label_filtered = pooled_region_label_alignedForTau

                        if regressionType == 1: # geneX, H3 -> CCF
                            spatialReconstruction = True
                            tauRegression = False
                            response_dim = 2
                            if predictorPathSuffix == 'GenePredictors':
                                x_data = gene_data_dense_H2layerFiltered_standard
                                y_data = [np.hstack((apCCF_per_cell_H2layerFiltered_standard[layerIDX],mlCCF_per_cell_H2layerFiltered_standard[layerIDX])) for layerIDX in range(numLayers)]
                                region_label_filtered = cell_region_H2layerFiltered[resolution]
                            if predictorPathSuffix == 'H3Predictors':
                                x_data = pooledH3_for_spatial #[m.T for m in pooledH3_for_spatial] #H3_per_cell_H2layerFiltered
                                y_data = [standardized_CCF_Tau[layerIDX][:,[1,0]] for layerIDX in range(numLayers)]
                                region_label_filtered = [np.array(pooled_region_label[layerIDX]) for layerIDX in range(numLayers)]

                        # if ???:
                        #     y_data = tau_per_cell_H2layerFiltered_standard

                        regressionConditions = [spatialReconstruction,tauRegression,regionalResample] #genePredictors (relic term at index 3)

                        print(f'\nStarting Regressions, Type {regressionType}:')
                        if (regressionType == 0):
                            print(f'{datasetName} {predictorTitle} -> {lineSelection} Tau (CCF Pooling={tauPoolSize}mm, predThresh={meanPredictionThresh}, regionResamp={regressionConditions[2]})')
                            if verbose:
                                predictor_response_info(x_data,y_data)
                            if y_data[0].shape[0] != x_data[0].shape[0]:
                                print(f'Aborting regression, number of predictor and response matrix observations do not agree...')
                                break
                            best_coef_tau,lasso_weight_tau,bestAlpha_tau,alphas_tau,tauPredictions_tau,bestR2_tau,loss_history_test_tau,loss_history_train_tau,dual_gap_history_tau = layerRegressions(response_dim,n_splits,highMeanPredictorIDXs,x_data,y_data,layerNames,regressionConditions,region_label_filtered,alphaParams,max_iter)
                            predictor_condition_numbers_tau = [np.linalg.cond(x) for x in x_data]

                        if (regressionType == 1):
                            print(f'{datasetName} {predictorTitle} -> CCF (predThresh={meanPredictionThresh}, regionResamp={regressionConditions[2]})')
                            if verbose:
                                predictor_response_info(x_data,y_data)
                            if y_data[0].shape[0] != x_data[0].shape[0]:
                                print(f'Aborting regression, number of predictor and response matrix observations do not agree...')
                                break
                            best_coef_spatial,lasso_weight_spatial,bestAlpha_spatial,alphas_spatial,tauPredictions_spatial,bestR2_spatial,loss_history_test_spatial,loss_history_train_spatial,dual_gap_history_spatial = layerRegressions(response_dim,n_splits,highMeanPredictorIDXs,x_data,y_data,layerNames,regressionConditions,region_label_filtered,alphaParams,max_iter)
                            predictor_condition_numbers_spatial = [np.linalg.cond(x) for x in x_data]


                    mean_fold_coef_tau = [np.mean(best_coef_tau[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
                    sd_fold_coef_tau = [np.std(best_coef_tau[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
                    sorted_coef_tau = [np.argsort(mean_fold_coef_tau[layerIDX]) for layerIDX in range(numLayers)]
                    mean_fold_coef_spatial = [np.mean(best_coef_spatial[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
                    sd_fold_coef_spatial = [np.std(best_coef_spatial[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
                    sorted_coef_spatial = [np.argsort(mean_fold_coef_spatial[layerIDX]) for layerIDX in range(numLayers)]


                    params = {}
                    params['n_splits'] = n_splits
                    params['tauPoolSize'] = tauPoolSize
                    params['numLayers'] = numLayers
                    params['meanExpressionThresh'] = meanExpressionThresh
                    params['meanPredictionThresh'] = meanPredictionThresh
                    params['highMeanPredictorIDXs'] = highMeanPredictorIDXs
                    params['numPrecision'] = numPrecision
                    params['alphaPrecision'] = alphaPrecision
                    params['structNum'] = structNum

                    paths = {}
                    paths['savePath'] = savePath_OSs
                    paths['predictorPathSuffix'] = predictorPathSuffix
                    paths['tauSortedPath'] = tauSortedPath_OSs

                    titles = {}
                    titles['predictorTitle'] = predictorTitle
                    titles['datasetName'] = datasetName
                    titles['layerNames'] = layerNames
                    titles['predictorNamesArray'] = predictorNamesArray
                    titles['predictorEncodeType'] = predictorEncodeType

                    plotting_data = {}
                    plotting_data['mean_expression_standard'] = mean_expression_standard
                    plotting_data['tauPredictions_spatial'] = tauPredictions_spatial
                    plotting_data['tauPredictions_tau'] = tauPredictions_tau
                    plotting_data['loss_history_test_spatial'] = loss_history_test_spatial
                    plotting_data['loss_history_test_tau'] = loss_history_test_tau
                    plotting_data['loss_history_train_spatial'] = loss_history_train_spatial
                    plotting_data['loss_history_train_tau'] = loss_history_train_tau
                    plotting_data['dual_gap_history_spatial'] = dual_gap_history_spatial
                    plotting_data['dual_gap_history_tau'] = dual_gap_history_tau
                    plotting_data['predictor_condition_numbers_spatial'] = predictor_condition_numbers_spatial
                    plotting_data['predictor_condition_numbers_tau'] = predictor_condition_numbers_tau

                    model_vals = {}
                    model_vals['sd_fold_coef_tau'] = sd_fold_coef_tau
                    model_vals['sd_fold_coef_spatial'] = sd_fold_coef_spatial
                    model_vals['mean_fold_coef_tau'] = mean_fold_coef_tau
                    model_vals['mean_fold_coef_spatial'] = mean_fold_coef_spatial
                    model_vals['bestR2_spatial'] = bestR2_spatial
                    model_vals['bestR2_tau'] = bestR2_tau
                    model_vals['sorted_coef_spatial'] = sorted_coef_spatial
                    model_vals['sorted_coef_tau'] = sorted_coef_tau
                    model_vals['bestAlpha_spatial'] = bestAlpha_spatial
                    model_vals['bestAlpha_tau'] = bestAlpha_tau
                    model_vals['alphas_spatial'] = alphas_spatial
                    model_vals['alphas_tau'] = alphas_tau
                    model_vals['lasso_weight_spatial'] = lasso_weight_spatial
                    model_vals['lasso_weight_tau'] = lasso_weight_tau

                    meta_dict = {}
                    meta_dict['lineSelection'] = lineSelection
                    meta_dict['structList'] = structList
                    meta_dict['areaColors'] = areaColors
                    meta_dict['plottingConditions'] = plottingConditions
                    meta_dict['params'] = params
                    meta_dict['paths'] = paths
                    meta_dict['titles'] = titles
                    meta_dict['model_vals'] = model_vals
                    meta_dict['plotting_data'] = plotting_data

                    output_dir = os.path.join(tauSortedPath, f'{predictorPathSuffix}', f'{datasetName}')
                    os.makedirs(output_dir, exist_ok=True)
                    with open(os.path.join(output_dir, 'plotting_data.pickle'), 'wb') as handle:
                        pickle.dump(meta_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # temp_path = 'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing\H3\Cux2-Ai96\pooling0.1025mm\GenePredictors\Merfish'
                    # meta_dict = pickle.load(open(os.path.join(temp_path,f'plotting_data.pickle'), 'rb'))

                    # lineSelection = meta_dict['lineSelection']
                    # structList = meta_dict['structList']
                    # areaColors = meta_dict['areaColors']
                    # plottingConditions = meta_dict['plottingConditions']
                    # params = meta_dict['params']
                    # paths = meta_dict['paths']
                    # titles = meta_dict['titles']
                    # model_vals = meta_dict['model_vals']
                    # plotting_data = meta_dict['plotting_data']

                    try:
                        plot_regressions(lineSelection, structList, areaColors, plottingConditions, params, paths, titles, model_vals, plotting_data)
                    except Exception as e:
                        print(f"An error occurred while plotting regressions: {e}")

        ##############################################
        ### Layer-specific expression correlations ###
        try:
            plot_expression_correlations(layerNames, structList, areaColors, mean_expression_standard, savePath)
        except Exception as e:
            print(f"An error occurred while plotting expression correlations: {e}")

        time_end = datetime.datetime.now()
        print(f'Time to run: {time_end - time_load_data}')


if __name__ == '__main__':
    main()
