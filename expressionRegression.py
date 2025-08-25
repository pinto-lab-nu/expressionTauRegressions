import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
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
#from packages.functional_dataset import *
from packages.plotting_util import *
from collections import Counter
import datetime
import re
import argparse
import psutil


#from utils.connect_to_dj import VM
from ccfRegistration.ccf_utils import * #load_tau_CCF, key_CCF, merge_regions, calculate_pooling_grid, passing_census, functional_timescales


test_mode = False # just makes a few things easier when runninng in interactive mode


def string_sanitizer(input_string):
        sanitized_string = re.sub(r'\/','_',input_string)
        return sanitized_string

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', '1'}:
        return True
    elif value.lower() in {'false', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def plot_mask(structureOfInterest,structure_mask,save_path,resolution):
    plt.figure()
    plt.title(f'{structureOfInterest}')
    plt.imshow(np.mean(structure_mask[:,:,:],axis=1)) #restrict to ~40 voxels along axis 1 to get dorsal cortex (for 25 rez), this is just for viz
    plt.savefig(os.path.join(save_path,'Masks',f'{structureOfInterest}_{resolution}.pdf'),dpi=600,bbox_inches='tight')
    plt.close()

def cell_region_function(cell_region, cell_layer, resolution, structure_mask, CCFvalues, CCFindexOrder, CCFmultiplier, structIDX, layerIDX):
    for cell in range(cell_region[resolution].shape[0]):
        currentMask = structure_mask[round(CCFvalues[cell,CCFindexOrder[0]]*CCFmultiplier), round(CCFvalues[cell,CCFindexOrder[1]]*CCFmultiplier), round(CCFvalues[cell,CCFindexOrder[2]]*CCFmultiplier)]
        if currentMask > 0:
            cell_region[resolution][cell] = structIDX
            if resolution == '10':
                cell_layer[resolution][cell] = layerIDX

    return cell_region,cell_layer

def region_count_printer(cell_region,resolution,datasetName,struct_list):
    regionalCounts = Counter(cell_region[resolution])
    print(f'Regional Cell Counts, {datasetName}:')
    for structIDX in range(len(struct_list)):
        print(f'{struct_list[structIDX]}:{regionalCounts[structIDX]}')

def pre_regression_check(verbose, x_data, y_data):
    if verbose:
        predictor_response_info(x_data,y_data)
    if y_data[0].shape[0] != x_data[0].shape[0]:
        print(f'Aborting regression, number of predictor and response matrix observations do not agree...')
        return False
    return True

def get_keys(key_list):
    return [
        {
            "subject_fullname": k[0],
            "session_date": k[1],
            "session_number": k[2],
        }
        for k in key_list
    ]

# def memory_usage():
#     total_size = 0
#     for name, value in globals().items():
#         try:
#             size = sys.getsizeof(value)
#             total_size += size
#             if size > 1024 ** 3: # only print variables larger than ~1GB
#                 print(f'Variable: {name}, Size: {size / (1024 ** 3):.2f} GB')
#         except TypeError:
#             pass
#     print(f'Total memory usage: {total_size / (1024 ** 3):.2f} GB')
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    print(f'Total memory usage: {mem_bytes / (1024 ** 3):.2f} GB')



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--line_selection", choices=["Cux2-Ai96", "Rpb4-Ai96"], default="Cux2-Ai96") #select the functional dataset for tau regressions
    parser.add_argument("--gene_limit", type=int, default=-1) #for testing purposes to load a subset of merfish-imputed data, set to -1 to include all genes
    parser.add_argument("--restrict_merfish_imputed_values", type=str_to_bool, default=False) #condition to restrict merfish-imputed dataset to non-imputed genes
    parser.add_argument("--tau_pool_size_array_full", type=lambda s: [float(item) for item in s.split(',')], default="4.0") #[1,2,3,4,5] #in 25um resolution CCF voxels, converted to mm later
    parser.add_argument("--n_splits", type=int, default=5) #number of splits for cross-validations in regressions
    parser.add_argument("--alpha_params", type=lambda s: [float(item) for item in s.split(',')], default="-5,-2,30") # [Alpha Lower (10**x), Alpha Upper (10**x), Steps]... alpha values for Lasso regressions
    parser.add_argument("--plotting", type=str_to_bool, default=True)
    parser.add_argument("--num_precision", type=int, default=3)   # Just for display (in plotting and regression text files)
    parser.add_argument("--alpha_precision", type=int, default=5) # Just for display (in plotting and regression text files)
    parser.add_argument("--verbose", type=str_to_bool, default=True)    # For print statements
    parser.add_argument("--predictor_order", type=lambda s: [int(item) for item in s.split(',')], default="0")           # Select predictors for regressions, and order [0:merfish{-imputed}, 1:pilot]
    parser.add_argument("--regressions_to_start", type=lambda s: [int(item) for item in s.split(',')], default="0,1,2,3")    # Select predictor/response variables for regressions, and order [0:X->Tau, 1:X->CCF, 2:X,CCF->Tau, 3:X->Tau Res.]
    parser.add_argument("--max_iter", type=int, default=200) # For layer regressions
    parser.add_argument("--variable_management", type=str_to_bool, default=True) # Removes large variables from memory after use (needs to be expanded to include more variables)
    parser.add_argument("--plotting_conditions", type=lambda s: [bool(int(item)) for item in s.split(',')], default="0,1") # For plotting spatial reconstructions
    parser.add_argument("--arg_parse_test", type=str_to_bool, default=False) # For testing the bash argument parser
    parser.add_argument("--job_task_id", type=int, default=0) # For parallel processing
    parser.add_argument("--bootstrapping_scale", type=float, default=1.0) # For CCF coordinate pool bootstrapping (functional-transcriptomic pool registration) scale, default is 1.0
    parser.add_argument("--min_pool_size", type=int, default=3) # Minimum number of pixels and cells in a pool for bootstrapping, default is 3
    parser.add_argument("--preprocessing_only", type=str_to_bool, default=False) # If True, only run preprocessing steps and skip regressions

    if test_mode:
        args = parser.parse_args([])
        gene_limit = 2 # For testing purposes, limit the number of genes to 2
    else:
        args = parser.parse_args()
        gene_limit = args.gene_limit

    line_selection = args.line_selection
    restrict_merfish_imputed_values = args.restrict_merfish_imputed_values
    tau_pool_size_array_full = args.tau_pool_size_array_full
    n_splits = args.n_splits
    alpha_params = args.alpha_params
    plotting = args.plotting
    num_precision = args.num_precision
    alpha_precision = args.alpha_precision
    verbose = args.verbose
    predictor_order = args.predictor_order
    regressions_to_start = args.regressions_to_start
    max_iter = args.max_iter
    variable_management = args.variable_management
    plotting_conditions = args.plotting_conditions
    arg_parse_test = args.arg_parse_test
    job_task_id = args.job_task_id
    bootstrapping_scale = args.bootstrapping_scale
    min_pool_size = args.min_pool_size
    preprocessing_only = args.preprocessing_only

    functional_set = 'renan_set'
    log_tau = False

    # make sure that alpha_params steps is an integer
    alpha_params[2] = int(alpha_params[2])

    print(f"line_selection: {line_selection}")
    print(f"gene_limit: {gene_limit}")
    print(f"restrict_merfish_imputed_values: {restrict_merfish_imputed_values}")
    print(f"tau_pool_size_array_full: {tau_pool_size_array_full}")
    print(f"n_splits: {n_splits}")
    print(f"alpha_params: {alpha_params}")
    print(f"plotting: {plotting}")
    print(f"num_precision: {num_precision}")
    print(f"alpha_precision: {alpha_precision}")
    print(f"verbose: {verbose}")
    print(f"predictor_order: {predictor_order}")
    print(f"regressions_to_start: {regressions_to_start}")
    print(f"max_iter: {max_iter}")
    print(f"variable_management: {variable_management}")
    print(f"plotting_conditions: {plotting_conditions}")
    print(f"arg_parse_test: {arg_parse_test}")
    print(f"job_task_id: {job_task_id}")
    print(f"bootstrapping_scale: {bootstrapping_scale}")
    print(f"min_pool_size: {min_pool_size}")
    print(f"preprocessing_only: {preprocessing_only}")

    if arg_parse_test:
        sys.exit('Test mode: terminating script...')

    
    
    #struct_list = np.array(['MOp','MOs','VISa','VISp','VISam','VISpm','SS','RSP'])
    # area_colors = ['#ff0000','#ff704d',                      #MO, reds
    #             '#4dd2ff','#0066ff','#003cb3','#00ffff',    #VIS, blues
    #             '#33cc33',                                  #SSp, greens
    #             '#a366ff']                                  #RSP, purples
    region_color_dict = {
        'MOp'   : "#1B50FF", #'#ff704d',
        'MOs'   : '#2F4077', #'#ff0000',
        'VISa'  : '#B05F1D', #'#00aeff',
        'VISp'  : '#B08D41', #'#1a31fc',
        'VISam' : "#CBBD4F", #'#21ffff',
        'VISpm' : "#8F680E", #'#2a5cc1',
        'SS'    : '#33cc33',
        'RSP'   : '#a366ff',
    }

    struct_list = np.array(list(region_color_dict.keys()))
    area_colors = list(region_color_dict.values())



    #for restrict_merfish_imputed_values, predictor_order in zip([False], [[0]]): #[False,True],[[0],[0,1]]
    if True: # just a placeholder, should probably just remove the above line and the conditional runs below...

        print(restrict_merfish_imputed_values, predictor_order) # now the default is just merfish-imputed data, no pilot genes

        structNum = len(struct_list)
        #applyLayerSpecificityFilter = False #ensure that CCM coordinates are contained within a layer specified in layerAppend
        #layerAppend = '2/3'
        #groupSelector = 12  #12 -> IT_7  -> L2/3 IT
                            #4  -> IT_11 -> L4/5 IT
                            #14 -> IT_9  -> L5 IT
                            #11 -> IT_6  -> L6 IT


        #if applyLayerSpecificityFilter:
        #    struct_list = [x+layerAppend for x in struct_list]


        my_os, save_path_OSs, download_base = pathSetter()
        save_path = save_path_OSs[my_os == 'Windows']

        time_start = datetime.now()

        standard_scaler = StandardScaler()
        #hotencoder = OneHotEncoder(sparse_output=False)

        ################################
        ### CCF Reference Space Creation
        ### See link for CCF example scripts from the allen: allensdk.readthedocs.io/en/latest/_static/examples/nb/reference_space.html
        tree = {}
        ref_space = {}
        for resolution in [10,25,100]:
            output_dir = os.path.join(save_path,'Data',f'nrrd{resolution}')
            reference_space_key = os.path.join('annotation','ccf_2017')
            ref_space_cache = ReferenceSpaceCache(resolution, 'annotation/ccf_2017', manifest=Path(output_dir) / 'manifest.json') #reference_space_key replaced by 'annotation/ccf_2017'
            # ID 1 is the adult mouse structure graph
            tree[f'{resolution}'] = ref_space_cache.get_structure_tree(structure_graph_id=1)

            annotation, meta = ref_space_cache.get_annotation_volume() #in browser navigate to the .nrrd file and download manually, not working automatically for some reason
            # The file should be moved to the reference space key directory, only needs to be done once
            os.listdir(Path(output_dir) / reference_space_key)
            ref_space[f'{resolution}'] = ref_space_cache.get_reference_space()

        #################
        ### Load data ###
        if functional_set == 'renan_set':
            key_list_intothevoid = [
                ("rsx2237_Cilantro", "2023-05-23", 1),
                ("pss3570_Chip", "2022-09-04", 1),
                ("pss3570_Caesar", "2022-09-04", 1),
                ("pss3570_Champ", "2022-09-04", 1),
                ("rsx2237_Charlie", "2023-05-16", 2),
            ]
        if functional_set == 'extra_set':
            key_list_intothevoid = [
                ("rsx2237_Cilantro", "2023-05-23", 1),
                ("pss3570_Chip", "2022-09-04", 1),
                ("pss3570_Caesar", "2022-09-04", 1),
                ("pss3570_Champ", "2022-09-04", 1),
                ("rsx2237_Charlie", "2023-05-16", 2),
                ("rsx2237_Carmella", "2024-01-12", 2),
                ("rsx2237_Cordelia", "2024-01-12", 2),
                ("rsx2237_Cynthia", "2024-10-03", 2),
            ]
        if functional_set == 'extra_only_set':
            key_list_intothevoid = [
                ("rsx2237_Carmella", "2024-01-12", 2),
                ("rsx2237_Cordelia", "2024-01-12", 2),
                ("rsx2237_Cynthia", "2024-10-03", 2),
            ]

        key_list_intothevoid = get_keys(key_list_intothevoid)

        gene_data_dense, pilot_gene_names, fn_clustid, fn_CCF = pilotLoader(save_path)
        merfish_CCF_Genes, all_merfish_gene_names, gene_categories = merfishLoader(save_path, download_base, pilot_gene_names, restrict_merfish_imputed_values, gene_limit)
        all_tau_CCF_coords, CCF25_bregma, CCF25_lambda = load_tau_CCF(line=line_selection, keys=key_list_intothevoid, task='IntoTheVoid', ts_param_set_id=3, corr_param_set_id=2, tau_cutoff=59.9)
        all_tau_CCF_coords[1,:] *= -1 #invert AP CCF coordinates for regressions

        if log_tau:
            all_tau_CCF_coords[2,:] = np.log(all_tau_CCF_coords[2,:])

        # if verbose:
        #     fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        #     axes = axes.flatten()
        #     for i, current_key in enumerate(key_list_intothevoid):
        #         print(i)
        #         all_tau_CCF_coords, CCF25_bregma, CCF25_lambda = load_tau_CCF(line=line_selection, keys=current_key, task='IntoTheVoid', ts_param_set_id=3, corr_param_set_id=2)
        #         ax = axes[i]
        #         ax.scatter(all_tau_CCF_coords[0, :], all_tau_CCF_coords[1, :], s=1)
        #         ax.set_title(f'{current_key}')
        #         ax.set_ylim([0, 30])

        # dump list of all_merfish_gene_names into a text file for reference
        with open(os.path.join(save_path,f'merfishImputed_allGeneNames.txt'), "w") as file:
            for currentGene in all_merfish_gene_names:
                file.write(currentGene+'\n')

        time_load_data = datetime.now()
        print(f'Time to load data: {time_load_data - time_start}')
        #memory_usage()


        #standardMerfish_CCF_Genes = standard_scaler.fit_transform(merfish_CCF_Genes)
        #standardMerfish_CCF_Genes = pd.DataFrame(standardMerfish_CCF_Genes, columns=merfish_CCF_Genes.columns)

        enriched_gene_names = list(merfish_CCF_Genes.drop(columns=['x_ccf','y_ccf','z_ccf']).columns)
        total_genes = gene_data_dense.shape[1]

        raw_merfish_genes = np.array(merfish_CCF_Genes.drop(columns=['x_ccf','y_ccf','z_ccf']))
        numMerfishCells = merfish_CCF_Genes.shape[0]
        raw_merfish_CCF = np.array(merfish_CCF_Genes.loc[:,['x_ccf','y_ccf','z_ccf']])
        del merfish_CCF_Genes

        print(f'Memory usage of raw_merfish_genes: {round(sys.getsizeof(raw_merfish_genes)/1024/1024,1)} MB')
        print(f'Memory usage of raw_merfish_CCF: {round(sys.getsizeof(raw_merfish_CCF)/1024/1024,1)} MB') #even though there are only three coordinate axes, the precision of these values is higher than the gene expressions (uses up more memory than might be expected)

        # # Plot example gene expression for Fig 1
        # plt.figure(figsize=(15,15))
        # plt.scatter(raw_merfish_CCF[:,0], raw_merfish_CCF[:,2], s=1, c=raw_merfish_genes[:,0], cmap='gray')
        # plt.axis('equal')


        # with open(os.path.join(save_path,f'Chen_merfishImputed_geneOverlap.txt'), "w") as file:
        #     file.write('Gene Overlap (Chen & Merfish-Imputed Datasets):\n\n')

        #     for currentGene in pilot_gene_names:
        #         geneIDX_Text = f'Merfish Imputed IDX of {currentGene}: {np.where(np.array(all_merfish_gene_names)==currentGene)[0]}'
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
        merfish_layer_names = ['L2_3 IT_ET','L4_5 IT_ET','L6 IT_ET'] # 'L4_5 IT_ET', 'L5 IT_ET', 'L6 IT_ET'] #['CTX IT, ET']
        #merfish_layers = ['2/3','4/5','6']
        merfish_subLayers = [['2/3'],['4','5'],['6a','6b']]

        pilotLayerNames  =  ['L2_3 IT',   'L4_5 IT',  'L5 IT',    'L6 IT',    'L5 ET']
        layerIDs    =       [12,          4,          14,         11,         17]
        # H2 names are...   ['IT 7',      'IT 11',    'IT 9',     'IT 6',     'PT']
        #numLayers = len(layerIDs)


        # ### Testing ###
        # for structIDX,structureOfInterest in enumerate(struct_list):
        #     if structIDX > -1:
        #         structureOfInterestAppend = structureOfInterest + 'agl6a'
        #         print(tree.get_structures_by_acronym([structureOfInterestAppend]))


        if not os.path.exists(os.path.join(save_path,'Masks')):
            os.makedirs(os.path.join(save_path,'Masks'))

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

        for resolution, datasetName in zip(['10','25'], ['Merfish'+merfish_datasetName_append,'Pilot']):
            if resolution == '10':
                CCFvalues = raw_merfish_CCF
                CCFmultiplier = 100
                CCFindexOrder = [0,1,2]
            if resolution == '25':
                CCFvalues = fn_CCF
                CCFmultiplier = 1
                CCFindexOrder = [0,1,2]

            maskDim0,maskDim1,maskDim2 = ref_space[f'{resolution}'].make_structure_mask([1]).shape
            
            fig, ax = plt.subplots(1,3,figsize=(15,5))
            for ccfIDX,axes in enumerate(ax):
                axes.hist(CCFvalues[:,ccfIDX], color='black', bins=500)
            plt.suptitle(f'CCF Distributions (resolution={resolution})')
            plt.savefig(os.path.join(save_path,'Masks',f'CCFdistributions_resolution{resolution}.pdf'), dpi=600, bbox_inches='tight')
            plt.close()

            #fig, ax = plt.subplots(1,1,figsize=(8,8))
            if resolution == '10':
                print(f'\n(10 um masks, layer masking required for Merfish dataset)')
                for layerIDX,layerAppend in enumerate(merfish_layer_names):
                    print('\n')

                    for structIDX,structureOfInterest in enumerate(struct_list):
                        print(f'Making {layerAppend} {structureOfInterest} CCF mask (resolution={resolution} um)...')
                        
                        if structureOfInterest == 'RSP':
                            structure_mask = np.zeros((maskDim0,maskDim1,maskDim2))
                            for subLayerAppend in merfish_subLayers[layerIDX]:
                                if subLayerAppend != '4':
                                    for subRSP in ['v','d','agl']:
                                        structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest+subRSP+subLayerAppend])
                                        structureID = structureTree[0]['id']
                                        structure_mask += ref_space[f'{resolution}'].make_structure_mask([structureID])
                        
                        if structureOfInterest == 'SS':
                            structure_mask = np.zeros((maskDim0,maskDim1,maskDim2))
                            for subLayerAppend in merfish_subLayers[layerIDX]:
                                for subSS in ['p-n','p-bfd','p-ll','p-m','p-ul','p-tr','p-un','s']:
                                    structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest+subSS+subLayerAppend])
                                    structureID = structureTree[0]['id']
                                    structure_mask += ref_space[f'{resolution}'].make_structure_mask([structureID])

                        if (structureOfInterest != 'SS') and (structureOfInterest != 'RSP'):
                            structure_mask = np.zeros((maskDim0,maskDim1,maskDim2))
                            for subLayerAppend in merfish_subLayers[layerIDX]:
                                if not((subLayerAppend == '4') and (structureOfInterest == 'MOp')) and not((subLayerAppend == '4') and (structureOfInterest == 'MOs')):
                                    structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest+subLayerAppend])
                                    structureID = structureTree[0]['id']
                                    structure_mask += ref_space[f'{resolution}'].make_structure_mask([structureID])

                        plot_mask(f'{structureOfInterest} {string_sanitizer(layerAppend)}',structure_mask,save_path,resolution)
                        cell_region,cell_layer = cell_region_function(cell_region,cell_layer,resolution,structure_mask,CCFvalues,CCFindexOrder,CCFmultiplier,structIDX,layerIDX)
                    region_count_printer(cell_region,resolution,f'{datasetName} {string_sanitizer(layerAppend)}',struct_list)
            
            if resolution == '25':
                print(f'\n(25 um masks, no layer masking required, as it is already present in dataset)\n')
                for structIDX,structureOfInterest in enumerate(struct_list):
                    print(f'Making {structureOfInterest} CCF mask (resolution={resolution} um)...')

                    structureTree = tree[f'{resolution}'].get_structures_by_acronym([structureOfInterest])
                    structureName = structureTree[0]['name']
                    structureID = structureTree[0]['id']
                    structure_mask = ref_space[f'{resolution}'].make_structure_mask([structureID])

                    plot_mask(structureOfInterest,structure_mask,save_path,resolution)
                    cell_region,cell_layer = cell_region_function(cell_region,cell_layer,resolution,structure_mask,CCFvalues,CCFindexOrder,CCFmultiplier,structIDX,layerIDX)
                region_count_printer(cell_region,resolution,datasetName,struct_list)



        regionalResample = False #resample each cortical region such that it's represented equally, in practice this tends to over-represent smaller regions
        regional_resampling = 3000
        cell_region_H2layerFiltered, gene_data_dense_H2layerFiltered, mlCCF_per_cell_H2layerFiltered, apCCF_per_cell_H2layerFiltered, H3_per_cell_H2layerFiltered, mean_expression = {},{},{},{},{},{}
        for layerNames, numLayers, resolution in zip([pilotLayerNames,merfish_layer_names], [len(pilotLayerNames),len(merfish_layer_names)], ['25','10']):
            if resolution == '10':
                CCFvalues = raw_merfish_CCF
                gene_data = raw_merfish_genes
            if resolution == '25':
                CCFvalues = fn_CCF
                gene_data = gene_data_dense
            
            CCFvalues[:,0] *= -1
            
            #tau_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
            cell_region_H2layerFiltered[resolution] = [np.empty((0,1)).astype(int) for _ in range(numLayers)]
            #tau_SD_per_cell_H2layerFiltered = [np.empty((0,1)) for _ in range(numLayers)]
            gene_data_dense_H2layerFiltered[resolution] = [np.empty((0,gene_data.shape[1])) for _ in range(numLayers)]
            mlCCF_per_cell_H2layerFiltered[resolution] = [np.empty((0,1)) for _ in range(numLayers)]
            apCCF_per_cell_H2layerFiltered[resolution] = [np.empty((0,1)) for _ in range(numLayers)]
            H3_per_cell_H2layerFiltered[resolution] = [np.empty((0,1)) for _ in range(numLayers)]
            mean_expression[resolution] = np.zeros((numLayers,len(struct_list),gene_data.shape[1]))
            #sigma_expression = np.zeros((numLayers,len(struct_list),gene_data_dense.shape[1]))

            for layerIDX, (layer, layerName) in enumerate(zip(layerIDs, layerNames)):

                if resolution == '25':
                    layerIDXs = set([i for i, s in enumerate(H2_all) if s == grouping[layer]])
                if resolution == '10':
                    layerIDXs = set(np.where(cell_layer['10'] == layerIDX)[0]) #set(np.where(cell_region['10'] > -1)[0])
                
                if layerIDX == 0:
                    if resolution == '25':
                        cux2columnIDX = np.where(np.array(pilot_gene_names)=='Cux2')[0][0]
                    if resolution == '10':
                        if gene_limit == -1:
                            cux2columnIDX = np.where(np.array(enriched_gene_names)=='Cux2')[0][0]
                    if gene_limit == -1:
                        cux2IDXs = set(np.where(gene_data[:,cux2columnIDX]>0)[0]) #for layer 2/3 filter out cells not expressing Cux2 (helps to align population to the functional dataset)

                for structIDX,structureOfInterest in enumerate(struct_list):
                    regionIDXs = set(np.where(cell_region[resolution] == structIDX)[0])
                    
                    H2layerFilter = layerIDXs & regionIDXs
                    if (layerIDX == 0) and (gene_limit == -1):
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

        for layerIDX,_ in enumerate(merfish_layer_names):
            mlCCF_per_cell_H2layerFiltered['10'][layerIDX] = -1 * np.abs(mlCCF_per_cell_H2layerFiltered['10'][layerIDX] - CCF_ML_Center_mm) + CCF_ML_Center_mm #must collapse resolution 10um CCF across midline since it is a bilateral coordinate system

        #############################################################################################
        ### visualization of and calculation of high expression genes are combined here, separate ###
        #all_tau_CCF_coords = np.load(os.path.join(tauPath,f'{line_selection}_tauCCF.npy'))
        all_tau_CCF_coords[0,:] *= 0.025 #convert 25um resolution functional-registered coordinates to mm
        all_tau_CCF_coords[1,:] *= 0.025 #convert "..."


        if plotting:
            # Plot CCF-registered Tau dataset
            fig, ax = plt.subplots(1, 1, figsize=(4,3))
            #hex_0 = '#00ffff'
            #hex_1 = '#ff33cc'
            #value_0 = np.percentile(all_tau_CCF_coords[2,:], 0)
            #value_1 = np.percentile(all_tau_CCF_coords[2,:], 97.5)
            #all_tau_colors = color_gradient(all_tau_CCF_coords[2,:], hex_0, hex_1, 0, 97.5)
            
            img = ax.scatter(all_tau_CCF_coords[0,::5], all_tau_CCF_coords[1,::5], s=1, alpha=0.5, c=all_tau_CCF_coords[2,::5], cmap='cool')
            #ax.scatter(CCF25_bregma[0], CCF25_bregma[1], color='blue', label='bregma')
            #ax.scatter(CCF25_lambda[0], CCF25_lambda[1], color='purple', label='lambda')
            ax.set_xlabel(r'CCF ($L \leftarrow M \rightarrow L$)')
            ax.set_ylabel(r'CCF ($P \leftrightarrow A$)')
            ax.set_title(rf'25$\mu$m CCF-Registered {line_selection} $\tau$ Dataset')
            #plt.axvline(x=CCF_ML_Center_mm, color='red', linestyle='--')
            #ax.legend()
            cbar = plt.colorbar(img, ax=ax)
            cbar.set_label(r'$\tau$ (s)')
            #plt.gca().invert_yaxis()
            plt.savefig(os.path.join(save_path, 'fig1_tau.pdf'), bbox_inches='tight')
            plt.close()


            # Plot example gene expression for Fig 1
            plt.figure(figsize=(3,3))
            plt.scatter(raw_merfish_CCF[::30,2], raw_merfish_CCF[::30,0], s=0.35, alpha=0.35, color='black')
            # make y ticks at positionns 2 - 10
            plt.yticks(np.arange(2, 11, 1))
            plt.xticks(np.arange(0, 11, 1))
            # Add dashed line at x=CCF_ML_Center_mm
            plt.axvline(x=CCF_ML_Center_mm, color='red', linestyle='--')
            plt.axis('equal')
            #plt.ylim((3,10))
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(save_path, 'fig1_merfish.pdf'), bbox_inches='tight')
            plt.close()

            # Plot Merfish cell locations in CCF space
            plt.figure(figsize=(1.75,3))
            ml_filter = np.where(mlCCF_per_cell_H2layerFiltered['10'][0].ravel() > 2)[0]
            plt.scatter(mlCCF_per_cell_H2layerFiltered['10'][0][ml_filter][::10,:], apCCF_per_cell_H2layerFiltered['10'][0][ml_filter][::10,:], color='black', s=0.1)
            plt.xticks(np.arange(2, 7, 1))
            plt.axis('equal')
            plt.savefig(os.path.join(save_path, 'fig1_merfish_filtered.pdf'), bbox_inches='tight')
            plt.close()
        
        

        #meanExpressionThreshArrayFull = [0] #[0.4,0.2,0.1,0]
        #meanH3ThreshArrayFull = [0] #[0.1,0.05,0.025,0]

        if my_os == 'Linux':
            #meanExpressionThreshArray = [meanExpressionThreshArrayFull[int(sys.argv[1])]] #batch job will distribute parameter instances among jobs run in parallel
            #meanH3ThreshArray = [meanH3ThreshArrayFull[int(sys.argv[1])]]                 #same for these regressions, just with a different parameter range
            tau_pool_size_array = [tau_pool_size_array_full[int(job_task_id)]]
        if my_os == 'Windows':
            #meanExpressionThreshArray = meanExpressionThreshArrayFull
            #meanH3ThreshArray = meanH3ThreshArrayFull
            tau_pool_size_array = tau_pool_size_array_full

        meanExpressionThresh,meanH3Thresh = 0,0

        layer_names_list  = [merfish_layer_names,pilotLayerNames]
        num_layers_list   = [len(merfish_layer_names),len(pilotLayerNames)]
        resolution_list  = ['10','25']
        dataset_name_list = ['Merfish'+merfish_datasetName_append,'Pilot']

        for layerNames,numLayers,resolution,datasetName in zip([layer_names_list[order] for order in predictor_order],[num_layers_list[order] for order in predictor_order],[resolution_list[order] for order in predictor_order],[dataset_name_list[order] for order in predictor_order]):

            #for meanExpressionThresh,meanH3Thresh in zip(meanExpressionThreshArray,meanH3ThreshArray):
            
            poolIndex = 0
            for tauPoolSize in tau_pool_size_array:
                tauPoolSize = np.round(tauPoolSize * 0.025, 4) #convert 25um resolution CCF functional-registered coordinates to mm
                poolIndex += 1
                tauSortedPath = os.path.join(save_path,line_selection,f'pooling{tauPoolSize}mm')
                tauSortedPath_OSs = [os.path.join(save_path_OSs[0],line_selection,f'pooling{tauPoolSize}mm'), os.path.join(save_path_OSs[1],line_selection,f'pooling{tauPoolSize}mm')]
                if not os.path.exists(tauSortedPath):
                    os.makedirs(tauSortedPath)
                
                ### pool tau into a grid for bootstrapping the regression ###
                minML_CCF, maxML_CCF, minAP_CCF, maxAP_CCF = np.min(all_tau_CCF_coords[0,:]), np.max(all_tau_CCF_coords[0,:]), np.min(all_tau_CCF_coords[1,:]), np.max(all_tau_CCF_coords[1,:])
                pooledTauCCF_coords = [np.empty((0,4)) for _ in range(numLayers)]
                pooledTauCCF_coords_noGene = [np.empty((0,4)) for _ in range(numLayers)]
                pooledPixelCount_v_CellCount = [np.empty((0,2)) for _ in range(numLayers)]
                resampledTau_aligned = [np.empty((0,1)) for _ in range(numLayers)]
                resampledAP_aligned = [np.empty((0,1)) for _ in range(numLayers)]
                resampledML_aligned = [np.empty((0,1)) for _ in range(numLayers)]
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
                    tau_cell_poolsize_mean_ratio = 0
                    print(f'Tau-Gene Alignment Pooling ({datasetName}, size {tauPoolSize}mm): {layerNames[layerIDX]}')
                    for current_tau_ML_pool in np.arange(minML_CCF,CCF_ML_Center_mm,tauPoolSize):
                        current_ML_tau_pooling_IDXs = np.where(np.abs(np.abs(all_tau_CCF_coords[0,:]-CCF_ML_Center_mm)-np.abs(current_tau_ML_pool-CCF_ML_Center_mm))<(tauPoolSize/2))[0] #our pixel space extents bilaterally, but CCF is unilateral, so 'CCF' coordinates from pixel space need to reflected over the ML center axis (CCF_ML_center)
                        
                        # if resolution == '25':
                        #     cellwise_ML_CCF_25 = mlCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1)
                        # if resolution == '10':
                        #     cellwise_ML_CCF_25 = np.abs((mlCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1) * ((10/25)*100)) - CCF_ML_Center)

                        current_ML_cell_pooling_IDXs = np.where(np.abs(mlCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1)-current_tau_ML_pool)<(tauPoolSize/2))[0]
                        for current_tau_AP_pool in np.arange(minAP_CCF,maxAP_CCF,tauPoolSize):
                            current_tau_pooling_IDXs = np.where(np.abs(all_tau_CCF_coords[1,current_ML_tau_pooling_IDXs]-current_tau_AP_pool)<(tauPoolSize/2))

                            # if resolution == '25':
                            #     cellwise_AP_CCF_25 = apCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1)
                            # if resolution == '10':
                            #     cellwise_AP_CCF_25 = apCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1) * ((10/25)*100)

                            current_cell_pooling_IDXs = np.where(np.abs(apCCF_per_cell_H2layerFiltered[resolution][layerIDX].reshape(-1)[current_ML_cell_pooling_IDXs]-current_tau_AP_pool)<(tauPoolSize/2))[0]
                            pooledTaus = all_tau_CCF_coords[2,current_ML_tau_pooling_IDXs[current_tau_pooling_IDXs]].reshape(-1,1)
                            if pooledTaus.shape[0] >= min_pool_size:
                                #print(mlCCF_per_cell_H2layerFiltered[layerIDX][current_ML_cell_pooling_IDXs])
                                possiblePoolsCount += 1
                                if current_cell_pooling_IDXs.shape[0] >= min_pool_size:
                                    #print(current_tau_ML_pool,current_tau_AP_pool)
                                    #sys.exit('testing :)')
                                    tau_cell_poolsize_ratio = pooledTaus.shape[0] / current_cell_pooling_IDXs.shape[0]
                                    tau_cell_poolsize_mean_ratio += tau_cell_poolsize_ratio
                                    
                                    geneProfilePresentCount += 1

                                    pooledTauCCF_coords[layerIDX] = np.vstack((pooledTauCCF_coords[layerIDX], np.array((current_tau_ML_pool,current_tau_AP_pool,np.mean(pooledTaus),np.std(pooledTaus))).reshape(1,4))) #switched order
                                    
                                    #######################################################
                                    ### Alignment of functional and expression datasets ###
                                    pool_resample_size = int(bootstrapping_scale * pooledTaus.shape[0]) #<- resample to the size of the functional data (sets functional dataset as the bottleneck)
                                    #pool_resample_size = int(bootstrapping_scale * current_cell_pooling_IDXs.shape[0]) #<- resample to the size of the expression data (sets expression dataset as the bottleneck)
                                    
                                    tauResamplingIDX = random.choices(np.arange(0,pooledTaus.shape[0]), k=pool_resample_size)
                                    resampledTau_aligned[layerIDX] = np.vstack((resampledTau_aligned[layerIDX],pooledTaus[tauResamplingIDX,:].reshape(-1,1)))
                                    tau_aligned_forH3[layerIDX] = np.vstack((tau_aligned_forH3[layerIDX],pooledTaus.reshape(-1,1))) #old, when expression (below) was resampled to size of functional dataset at pool (k=pooledTaus.shape[0]), this is kept for H3 regressions, which shouldn't be resampled to match th size of the expression dataset
                                    
                                    gene_pool_data = gene_data_dense_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                                    geneResamplingIDX = random.choices(np.arange(0,gene_pool_data.shape[0]), k=pool_resample_size)
                                    resampledGenes_aligned[layerIDX] = np.vstack((resampledGenes_aligned[layerIDX],gene_pool_data[geneResamplingIDX,:].reshape(-1,total_genes)))

                                    pool_apCCF = np.ones((pool_resample_size, 1)) * current_tau_AP_pool
                                    pool_mlCCF = np.ones((pool_resample_size, 1)) * current_tau_ML_pool
                                    resampledAP_aligned[layerIDX] = np.vstack((resampledAP_aligned[layerIDX],pool_apCCF))
                                    resampledML_aligned[layerIDX] = np.vstack((resampledML_aligned[layerIDX],pool_mlCCF))
                                    ### Alignment handled ###
                                    #########################

                                    if resolution == '25':
                                        H3_pool_data = H3_per_cell_H2layerFiltered[resolution][layerIDX][current_ML_cell_pooling_IDXs[current_cell_pooling_IDXs],:]
                                        
                                        data_flattened = H3_pool_data.flatten()
                                        categories = np.arange(1, 10)
                                        counts = np.array([np.sum(data_flattened == category) for category in categories])
                                        normalized_counts = (counts / counts.sum()).reshape(-1,9)

                                        H3ResamplingIDX = random.choices(np.arange(0,1), k=pool_resample_size) #k=pooledTaus.shape[0])

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

                    tau_cell_poolsize_mean_ratio /= geneProfilePresentCount
                    print(f'Tau:Cell Pool Size Mean Ratio = {round(tau_cell_poolsize_mean_ratio,3)}')
                    # if resolution == '25':
                    #     resampledH3_aligned_H2layerFiltered_OneHot.append(hotencoder.fit_transform(resampledH3_aligned_H2layerFiltered[layerIDX].T))
                    #     H3_per_cell_H2layerFiltered_OneHot.append(hotencoder.fit_transform(H3_per_cell_H2layerFiltered[resolution][layerIDX]))
                


                for layerIDX in range(numLayers):
                    plt.figure(), plt.title(f'CCF Pooling:{tauPoolSize}mm, Fraction of Tau Pooled Points with at least one Gene Profile:{round(genePoolSaturation[layerIDX],3)}\n{line_selection}, {layerNames[layerIDX]}')
                    plt.scatter(pooledTauCCF_coords_noGene[layerIDX][:,1],pooledTauCCF_coords_noGene[layerIDX][:,0],color='red',s=0.5)
                    plt.scatter(pooledTauCCF_coords[layerIDX][:,1],pooledTauCCF_coords[layerIDX][:,0],color='green',s=0.5)
                    plt.xlabel(r'A$\leftrightarrow$P (mm)'), plt.ylabel(r'L$\leftrightarrow$M (mm)'), plt.axis('equal')
                    plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{line_selection}_tauExpressionPooling{tauPoolSize}mm_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                    plt.close()

                    fig, ax = plt.subplots(1,1,figsize=(2.5,3))
                    cmap = plt.get_cmap('cool')
                    sc = ax.scatter(pooledTauCCF_coords[layerIDX][:,0], pooledTauCCF_coords[layerIDX][:,1]*-1, c=pooledTauCCF_coords[layerIDX][:,2], s=2, cmap=cmap)
                    ax.set_xlabel(r'L$\leftrightarrow$M (mm)')
                    ax.set_ylabel(r'P$\leftrightarrow$A (mm)')
                    ax.axis('equal')
                    ax.set_xticks(np.arange(2, 6, 1))
                    # Move colorbar further down below the x-axis label
                    #fig.subplots_adjust(bottom=0.22)  # Increase bottom margin to make space
                    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7)
                    cbar.set_label(f'$\\tau$ (s)')
                    plt.savefig(os.path.join(save_path, 'tau_pooling.pdf'), bbox_inches='tight')
                    plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{line_selection}_tauPooling{tauPoolSize}mm_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                    plt.close()
                    # fig, ax = plt.subplots(1,1,figsize=(8,8))
                    # plt.suptitle(f'{line_selection} Tau, CCF Pooling:{tauPoolSize}mm\n{line_selection}, {layerNames[layerIDX]}')
                    # cmap = plt.get_cmap('cool')
                    # global_min,global_max = np.log10(1),np.log10(30)
                    # norm = matplotlib.colors.Normalize(global_min, global_max)
                    # tau_colors = cmap(norm(np.log10(pooledTauCCF_coords[layerIDX][:,2])))
                    # ax.scatter(pooledTauCCF_coords[layerIDX][:,1],pooledTauCCF_coords[layerIDX][:,0],color=tau_colors,s=6)
                    # ax.set_xlabel(r'A$\leftrightarrow$P (mm)'), ax.set_ylabel(r'L$\leftrightarrow$M (mm)'), ax.axis('equal')
                    # mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    # mappable.set_array(tau_colors)
                    # cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=35, pad=0.1, orientation='horizontal')
                    # cbar_ticks = np.arange(global_min, global_max, 1)
                    # cbar.set_ticks(cbar_ticks)
                    # cbar.set_ticklabels(10**(cbar_ticks),fontsize=6,rotation=45)
                    # cbar.set_label('Tau', rotation=0)
                    # plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{line_selection}_tauPooling{tauPoolSize}mm_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                    # plt.close()

                    plt.figure(), plt.xlabel('Pool Pixel Count'), plt.ylabel('Pool Cell Count')
                    plt.title(f'Pool Size:{tauPoolSize}mm, Pixel & Cell Counts by CCF Pool\n{layerNames[layerIDX]}\n{line_selection}, {layerNames[layerIDX]}')
                    plt.scatter(pooledPixelCount_v_CellCount[layerIDX][:,0],pooledPixelCount_v_CellCount[layerIDX][:,1],color='black',s=1)
                    #plt.axis('equal')
                    plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{line_selection}_pooling{tauPoolSize}mm_CellPixelCounts_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                    plt.close()


                # pooledTauCCF_coords[layer] has columns: ML, AP, tau_mean, tau_std
                standardized_CCF_Tau = [standard_scaler.fit_transform(pooledTauCCF_coords[layerIDX]) for layerIDX in range(numLayers)]
                # for layerIDX in range(numLayers):
                #     standardized_CCF_Tau[layerIDX][:,1] *= -1 #invert standardized AP CCF for regressions
                APML_tau_models = []
                r_correlation_ml = []
                p_correlation_ml = []
                r_correlation_ap = []
                p_correlation_ap = []
                for layerIDX in range(numLayers):
                    linearmodel = LinearRegression()
                    
                    r_squared_regression = []
                    for dim in range(2):
                        linearmodel.fit(standardized_CCF_Tau[layerIDX][:,dim].reshape(-1,1),standardized_CCF_Tau[layerIDX][:,2].reshape(-1))
                        tau_pred = linearmodel.predict(standardized_CCF_Tau[layerIDX][:,dim].reshape(-1,1))
                        r_squared_regression.append(r2_score(standardized_CCF_Tau[layerIDX][:,2].reshape(-1,1), tau_pred))
                        # calculate r and p for correlation
                        r, p = scipy.stats.pearsonr(standardized_CCF_Tau[layerIDX][:,dim].reshape(-1), standardized_CCF_Tau[layerIDX][:,2].reshape(-1))
                        if dim == 0: #ML
                            r_correlation_ml.append(r)
                            p_correlation_ml.append(p)
                        if dim == 1: #AP
                            r_correlation_ap.append(r)
                            p_correlation_ap.append(p)

                    fig, ax = plt.subplots(1,2,figsize=(8,4))
                    ax[0].scatter(standardized_CCF_Tau[layerIDX][:,0],standardized_CCF_Tau[layerIDX][:,2],color='black',s=1)
                    ax[1].scatter(standardized_CCF_Tau[layerIDX][:,1],standardized_CCF_Tau[layerIDX][:,2],color='black',s=1)
                    ax[0].set_title(f'$R^2$={round(r_squared_regression[0],3)}'), ax[1].set_title(f'$R^2$={round(r_squared_regression[1],3)}')
                    ax[0].set_xlabel('Standardized ML CCF'), ax[1].set_xlabel('Standardized AP CCF')
                    ax[0].set_ylabel('Standardized Tau'), ax[1].set_ylabel('Standardized Tau')
                    plt.suptitle(f'Pooling={tauPoolSize}mm, Standardized {line_selection} Tau & AP, ML CCF Correlation\n{layerNames[layerIDX]}')
                    plt.savefig(os.path.join(tauSortedPath,f'{datasetName}_{line_selection}_pooling{tauPoolSize}mm_TauCCFcorr_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                    plt.close()

                    ### AP, ML -> Tau regression fit ###
                    linearmodel.fit(standardized_CCF_Tau[layerIDX][:,[1,0]], standardized_CCF_Tau[layerIDX][:,2].reshape(-1)) # [1,0] to switch to AP,ML order
                    tau_pred = linearmodel.predict(standardized_CCF_Tau[layerIDX][:,[1,0]])
                    r_squared_regression_APML_tau = r2_score(standardized_CCF_Tau[layerIDX][:,2].reshape(-1,1), tau_pred)
                    APML_tau_models.append(linearmodel)
                    #print(r_squared_regression_APML_tau)
                    #print(linearmodel.coef_, linearmodel.intercept_)
                    # plt.figure()
                    # plt.scatter(standardized_CCF_Tau[layerIDX][:,1], tau_pred, color='black', s=1)
                    # plt.title(f'R^2: {r_squared_regression_APML_tau}')
                    # plt.show()

                if datasetName == 'Pilot':
                    layerIDX = 0 #just plot for layer 2/3
                    for title_append in ['ylimit','']:
                        # Plot tau along ML and AP axes
                        fig, ax = plt.subplots(1,2,figsize=(5,2))
                        n_points = resampledML_aligned[layerIDX].shape[0]
                        jitter = np.random.normal(0, 0.05, n_points)
                        region_colors = [area_colors[int(idx)] for idx in pooled_region_label_alignedForTau[layerIDX]]
                        # Shuffle indices for random z-order
                        shuffle_idx = np.random.permutation(n_points)
                        print(f'shuffle_idx shape: {shuffle_idx.shape}, region_color array shape: {np.array(region_colors).shape}, resampled ML shape: {resampledML_aligned[layerIDX].shape}')
                        print([resampledML_aligned[idx].shape for idx in range(numLayers)])
                        ml_vals = resampledML_aligned[layerIDX][:,0][shuffle_idx] + jitter[shuffle_idx]
                        ap_vals = resampledAP_aligned[layerIDX][:,0][shuffle_idx] + jitter[shuffle_idx]
                        tau_vals = tau_aligned_forH3[layerIDX][:,0][shuffle_idx]
                        region_colors_shuffled = np.array(region_colors)[shuffle_idx]
                        ax[0].scatter(ml_vals, tau_vals, color=region_colors_shuffled, s=0.4)
                        ax[0].set_title(f'r={round(r_correlation_ml[layerIDX],3)}, p={p_correlation_ml[layerIDX]:.3e}')
                        ax[1].scatter(ap_vals, tau_vals, color=region_colors_shuffled, s=0.4)
                        ax[1].set_title(f'r={round(r_correlation_ap[layerIDX],3)}, p={p_correlation_ap[layerIDX]:.3e}')
                        ax[0].set_xlabel('ML CCF (mm)'), ax[1].set_xlabel('AP CCF (mm)')
                        ax[0].set_ylabel(f'$\\tau$ (s)')
                        if title_append == 'ylimit':
                            ax[0].set_ylim(0, 29)
                            ax[1].set_ylim(0, 29)
                        region_color_patch = [mpatches.Patch(color=area_colors[i], label=struct_list[i]) for i in range(len(struct_list))]
                        ax[1].legend(handles=region_color_patch, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small', borderaxespad=0.)
                        plt.savefig(os.path.join(save_path,f'{line_selection}_Tau{title_append}_by_ML_AP_{layerNames[layerIDX]}.pdf'), bbox_inches='tight')
                        plt.close()

                    param_settings = {'mc_param_set_id': 1,
                    'pixel_param_set_id': 1,
                    'dff_param_set_id': 2,
                    'wf_inclusion_param_set_id': 2,
                    'ts_param_set_id': 2,
                    'corr_param_set_id': 1}
                    plt.hist(all_tau_CCF_coords[2,:], color='black', bins=75)
                    plt.title(f'{line_selection} Tau Distribution Histogram \n {param_settings} \n {key_list_intothevoid}')
                    plt.xlabel(f'$\\tau$ (s)'), plt.ylabel('Counts')
                    plt.savefig(os.path.join(save_path,f'{line_selection}_Tau_Histogram.pdf'), bbox_inches='tight')
                    plt.close()

                ######################################################################################
                ### Standard Scaler transform the gene expression and tau data prior to regression ###
                mean_expression_standard = np.zeros_like(mean_expression[resolution])
                # Tau Regressions #
                resampledTau_aligned_standard = [np.zeros_like(resampledTau_aligned[layerIDX]) for layerIDX in range(numLayers)]
                tau_aligned_forH3_standard = [np.zeros_like(tau_aligned_forH3[layerIDX]) for layerIDX in range(numLayers)]
                resampledGenes_aligned_H2layerFiltered_standard = [np.zeros_like(resampledGenes_aligned[layerIDX]) for layerIDX in range(numLayers)]
                # CCF Regressions #
                resampledAP_aligned_standard = [np.zeros_like(resampledAP_aligned[layerIDX]) for layerIDX in range(numLayers)]
                resampledML_aligned_standard = [np.zeros_like(resampledML_aligned[layerIDX]) for layerIDX in range(numLayers)]
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
                    resampledAP_aligned_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(resampledAP_aligned[layerIDX][:,:]))
                    resampledML_aligned_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(resampledML_aligned[layerIDX][:,:]))
                    mlCCF_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(mlCCF_per_cell_H2layerFiltered[resolution][layerIDX][:,:]))
                    apCCF_per_cell_H2layerFiltered_standard[layerIDX][:,:] = standard_scaler.fit_transform(np.asarray(apCCF_per_cell_H2layerFiltered[resolution][layerIDX][:,:]))
                        
                #print(np.mean(resampledGenes_aligned_H2layerFiltered_standard[0][:,:],axis=0)) #just to see that the means are zero after standardizing


                """ 
                if poolIndex == 1:
                    regressions_to_start = [0,1]
                    plotting_conditions = [False,True] #plot spatial reconstruction?
                else:
                    regressions_to_start = [0,1] #[0] #used to be no need to run spatial regression multiple times across pooling sizes, just when the meanPredictionThresh changes, but this now matters for H3 profiles per pool
                    plotting_conditions = [False]#[False] #no need to plot spatial regression plots across "..."
                """

                for namePredictors,predictorTitle,predictorEncodeType,predictorPathSuffix in zip(['Gene Predictors', 'H3 Predictors'],
                                                                                                ['Gene Expression',  'H3 Level'],
                                                                                                ['Standardized',     'Normalized'],
                                                                                                ['GenePredictors',   'H3Predictors']):
                    
                    if (predictorPathSuffix == 'H3Predictors') and ((datasetName == 'Merfish-Imputed') or (datasetName == 'Merfish')):
                        print(f'\nAborting H3 Predictor regressions for Merfish{merfish_datasetName_append} dataset...')
                        print(f'please load Pilot dataset if your intent is to run H3 Predictor regressions.\n')
                        break
                    
                    if not os.path.exists(os.path.join(save_path,'Spatial',f'{predictorPathSuffix}',f'{datasetName}')):
                        os.makedirs(os.path.join(save_path,'Spatial',f'{predictorPathSuffix}',f'{datasetName}'))

                    if predictorPathSuffix == 'GenePredictors':
                        predictorDataRaw = gene_data_dense_H2layerFiltered[resolution]
                        meanPredictionThresh = meanExpressionThresh
                        if resolution == '25':
                            #predictorDataRaw = gene_data_dense_H2layerFiltered[resolution]
                            #meanPredictionThresh = meanExpressionThresh
                            predictorNamesArray = np.array(pilot_gene_names)
                            #numLayers = len(pilotLayerNames)
                            #layerNames = pilotLayerNames
                        if resolution == '10':
                            #predictorDataRaw = [raw_merfish_genes]
                            #meanPredictionThresh = meanExpressionThresh
                            predictorNamesArray = np.array(enriched_gene_names)
                            #numLayers = len(merfish_layer_names)
                            #layerNames = merfish_layer_names

                    if predictorPathSuffix == 'H3Predictors':
                        predictorDataRaw = resampledH3_aligned_H2layerFiltered#[m.T for m in resampledH3_aligned_H2layerFiltered] #H3_per_cell_H2layerFiltered
                        meanPredictionThresh = meanH3Thresh
                        predictorNamesArray = np.arange(1, predictorDataRaw[layerIDX].shape[1]+1, 1)
                        numLayers = len(layerIDs)
                        layerNames = pilotLayerNames
                    
                    # if predictorPathSuffix == 'merfishImputedGenePredictors':
                    #     predictorDataRaw = [raw_merfish_genes]
                    #     meanPredictionThresh = 0.1
                    #     predictorNamesArray = np.array(enriched_gene_names)
                    #     numLayers = 1
                    #     layerNames = merfish_layer_names

                    if predictorPathSuffix == 'merfishImputedGenePredictors':
                        figWidth = 23
                    else:
                        figWidth = 15

                    highMeanPredictorIDXs = [[] for _ in range(numLayers)]
                    num_predictors = len(predictorNamesArray)
                    volcano_log2_fc = [np.zeros(num_predictors) for _ in range(numLayers)]
                    volcano_neg_log10_p = [np.ones(num_predictors) for _ in range(numLayers)]
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
                            plt.savefig(os.path.join(save_path,'Spatial',f'{predictorPathSuffix}',f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                        else:
                            plt.savefig(os.path.join(save_path,'Spatial',f'{predictorPathSuffix}',f'{datasetName}',f'excluded{datasetName}{predictorPathSuffix}Thresh{meanPredictionThresh}_{layerNames[layerIDX]}.pdf'),dpi=600,bbox_inches='tight')
                        plt.close()

                        ########################################################################
                        ### Script for transcriptomic tau-fold change in expression strength ###
                        if datasetName != 'Pilot':
                            tau_25th, tau_75th = np.percentile(resampledTau_aligned[layerIDX], [25, 75])
                            tau_25th_idxs = np.where(resampledTau_aligned[layerIDX] <= tau_25th)[0]
                            tau_75th_idxs = np.where(resampledTau_aligned[layerIDX] >= tau_75th)[0]

                            raw_p_values = np.ones(num_predictors)

                            for geneIDX in range(num_predictors):
                                gene_expression_tau25th = resampledGenes_aligned[layerIDX][tau_25th_idxs, geneIDX]
                                gene_expression_tau75th = resampledGenes_aligned[layerIDX][tau_75th_idxs, geneIDX]

                                # Calculate log2 fold change (add small value to avoid division by zero)
                                mean_25 = np.mean(gene_expression_tau25th) + 1e-10
                                mean_75 = np.mean(gene_expression_tau75th) + 1e-10
                                volcano_log2_fc[layerIDX][geneIDX] = np.log2(mean_75 / mean_25)

                                # Calculate p-value using t-test
                                p_val = ttest_ind(gene_expression_tau75th, gene_expression_tau25th, equal_var=False)[1] # Index at 1 (p-value) b/c index at 0 is the t-statistic
                                raw_p_values[geneIDX] = p_val
                                #volcano_neg_log10_p[layerIDX][geneIDX] = -np.log10(p_val)

                                if verbose:
                                    if geneIDX < 5: # Show a small number of genes as example of their expression distribution across the tau quartiles
                                        plt.figure(figsize=(8, 6))
                                        plt.hist(gene_expression_tau25th, bins=100, alpha=0.5, color='blue')
                                        plt.hist(gene_expression_tau75th, bins=100, alpha=0.5, color='orange')
                                        plt.axvline(np.mean(gene_expression_tau25th), color='blue', linestyle='dashed', linewidth=1, label=f'Mean Expression, Bottom Quartile $\\tau$')
                                        plt.axvline(np.mean(gene_expression_tau75th), color='orange', linestyle='dashed', linewidth=1, label=f'Mean Expression, Top Quartile $\\tau$')
                                        plt.xlabel('Gene Expression')
                                        plt.ylabel('Frequency')
                                        plt.title(f'{predictorNamesArray[geneIDX]} Expression Distribution Across $\\tau$ Quartiles')
                                        plt.legend()
                                        plt.savefig(os.path.join(tauSortedPath, f'{datasetName}_{line_selection}_geneExpressionDistributionAcrossTauQuartiles_{predictorNamesArray[geneIDX]}_{layerNames[layerIDX]}.pdf'), bbox_inches='tight')
                                        plt.close()
                            
                            # FDR correction of p-values
                            _, fdr_corrected_pvals, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
                            volcano_neg_log10_p[layerIDX] = -np.log10(fdr_corrected_pvals)
                            
                            # plt.figure(figsize=(8, 6))
                            # plt.scatter(volcano_log2_fc[layerIDX], volcano_neg_log10_p[layerIDX], s=3, color='black')
                            # plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='FDR p=0.05')
                            # plt.axvline(x=0, color='black', linestyle='--', label='FC=0')
                            # plt.title(f'Gene Expression Fold Change Across $\\tau$ Top & Bottom Quartiles ({datasetName}, {layerNames[layerIDX]})')
                            # plt.xlabel(f'$Log_2$ Expression Fold Change')
                            # plt.ylabel(f'$-Log_{{10}}$ FDR-corrected p-value')
                            # # Annotate each gene with its name
                            # for i, gene_name in enumerate(predictorNamesArray):
                            #     plt.annotate(gene_name, (volcano_log2_fc[layerIDX][i], volcano_neg_log10_p[layerIDX][i]), fontsize=10, alpha=1.0)
                            # plt.legend()
                            # plt.savefig(os.path.join(tauSortedPath, f'{datasetName}_{line_selection}_volcanoGeneExpressionFoldChangeAcrossTauQuartiles_{layerNames[layerIDX]}.pdf'), bbox_inches='tight')
                            # plt.close()
                        ############################################################################
                        ### End script for transcriptomic tau-fold change in expression strength ###

                    rename = os.path.join(save_path,'Spatial',f'{predictorPathSuffix}',f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}.pdf')
                    #PDFmerger(os.path.join(save_path,'Spatial',f'{predictorPathSuffix}'),f'excluded{predictorPathSuffix}Thresh{meanPredictionThresh}_',layerNames,'.pdf',rename)

                    if variable_management:
                        del predictorDataRaw

                    if not preprocessing_only:
                        model_vals = {}
                        model_vals['sd_fold_coef'] = {}
                        model_vals['mean_fold_coef'] = {}
                        model_vals['bestR2'] = {}
                        model_vals['sorted_coef'] = {}
                        model_vals['bestAlpha'] = {}
                        model_vals['alphas'] = {}
                        model_vals['lasso_weight'] = {}

                        plotting_data = {}
                        if datasetName != 'Pilot':
                            plotting_data['volcano_log2_fc'] = volcano_log2_fc
                            plotting_data['volcano_neg_log10_p'] = volcano_neg_log10_p

                        #plotting_data['distractor_genes'] = gene_categories['metabolic_genes']
                        plotting_data['mean_expression_standard'] = mean_expression_standard

                        plotting_data['tauPredictions'] = {}
                        plotting_data['loss_history_test'] = {}
                        plotting_data['loss_history_train'] = {}
                        plotting_data['dual_gap_history'] = {}
                        plotting_data['predictor_condition_numbers'] = {}

                        for regressionType in regressions_to_start:
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
                                    y_data = [np.hstack((apCCF_per_cell_H2layerFiltered_standard[layerIDX], mlCCF_per_cell_H2layerFiltered_standard[layerIDX])) for layerIDX in range(numLayers)]
                                    region_label_filtered = cell_region_H2layerFiltered[resolution]
                                if predictorPathSuffix == 'H3Predictors':
                                    x_data = pooledH3_for_spatial #[m.T for m in pooledH3_for_spatial] #H3_per_cell_H2layerFiltered
                                    y_data = [standardized_CCF_Tau[layerIDX][:,[1,0]] for layerIDX in range(numLayers)]
                                    region_label_filtered = [np.array(pooled_region_label[layerIDX]) for layerIDX in range(numLayers)]
                            
                            if (regressionType == 2) or (regressionType == 3): # geneX+AP+ML, H3+AP+ML -> Tau (or Tau Residuals)
                                spatialReconstruction = False
                                tauRegression = True
                                response_dim = 1
                                if predictorPathSuffix == 'GenePredictors':
                                    #x_data = resampledGenes_aligned_H2layerFiltered_standard
                                    x_data = [np.hstack((resampledGenes_aligned_H2layerFiltered_standard[layerIDX], resampledAP_aligned_standard[layerIDX], resampledML_aligned_standard[layerIDX])) for layerIDX in range(numLayers)]
                                    y_data = resampledTau_aligned_standard
                                    region_label_filtered = pooled_cell_region_geneAligned_H2layerFiltered
                                if predictorPathSuffix == 'H3Predictors':
                                    #x_data = resampledH3_aligned_H2layerFiltered#[m.T for m in resampledH3_aligned_H2layerFiltered]
                                    x_data = [np.hstack((resampledH3_aligned_H2layerFiltered[layerIDX], resampledAP_aligned_standard[layerIDX], resampledML_aligned_standard[layerIDX])) for layerIDX in range(numLayers)]
                                    y_data = tau_aligned_forH3_standard
                                    region_label_filtered = pooled_region_label_alignedForTau

                            # if ???:
                            #     y_data = tau_per_cell_H2layerFiltered_standard

                            regressionConditions = [spatialReconstruction,tauRegression,regionalResample] #genePredictors (relic term at index 3)

                            print(f'\nStarting Regressions, Type {regressionType}:')
                            print(f'(CCF Pooling={tauPoolSize}mm, predThresh={meanPredictionThresh}, regionResamp={regressionConditions[2]})')
                            if (regressionType == 0):
                                reconstruction_type = 'tau'
                                print(f'{datasetName} {predictorTitle} -> {line_selection} Tau')
                                if not (check_val := pre_regression_check(verbose, x_data, y_data)):
                                    break
                                best_coef, lasso_weight, bestAlpha, alphas, tauPredictions, bestR2, loss_history_test, loss_history_train, dual_gap_history = layerRegressions(response_dim, n_splits, highMeanPredictorIDXs, x_data, y_data, layerNames, regressionConditions, region_label_filtered, alpha_params, max_iter)

                            if (regressionType == 1):
                                reconstruction_type = 'spatial'
                                print(f'{datasetName} {predictorTitle} -> CCF')
                                if not (check_val := pre_regression_check(verbose, x_data, y_data)):
                                    break
                                best_coef, lasso_weight, bestAlpha, alphas, tauPredictions, bestR2, loss_history_test, loss_history_train, dual_gap_history = layerRegressions(response_dim, n_splits, highMeanPredictorIDXs, x_data, y_data, layerNames, regressionConditions, region_label_filtered, alpha_params, max_iter)

                            if (regressionType == 2):
                                reconstruction_type = 'XCCF_tau'
                                print(f'{datasetName} {predictorTitle} + AP + ML -> {line_selection} Tau')
                                if not (check_val := pre_regression_check(verbose, x_data, y_data)):
                                    break
                                highMeanPredictorIDXs_XCCF = [np.concatenate((highMeanPredictorIDXs[layerIDX], np.array([x_data[layerIDX].shape[1]-2, x_data[layerIDX].shape[1]-1]))) for layerIDX in range(numLayers)] #add AP and ML CCF indices to high mean predictors
                                best_coef, lasso_weight, bestAlpha, alphas, tauPredictions, bestR2, loss_history_test, loss_history_train, dual_gap_history = layerRegressions(response_dim, n_splits, highMeanPredictorIDXs_XCCF, x_data, y_data, layerNames, regressionConditions, region_label_filtered, alpha_params, max_iter)
                            
                            if (regressionType == 3):
                                reconstruction_type = 'X_tauRes'
                                print(f'{datasetName} {predictorTitle} -> {line_selection} Tau Residuals (from CCF -> Tau Regression)')
                                ### generate the residuals from CCF -> Tau regression ###
                                #tau_hat = [APML_tau_models[layerIDX].predict(x_data[layerIDX][:,-2:]).reshape(-1,1) for layerIDX in range(numLayers)]
                                #target_tau_residuals = [y_data[layerIDX] - tau_hat[layerIDX] for layerIDX in range(numLayers)]

                                tau_hat = [np.zeros_like(y_data[layerIDX]) for layerIDX in range(numLayers)]
                                tau_residuals = [np.zeros_like(y_data[layerIDX]) for layerIDX in range(numLayers)]
                                for layerIDX in range(numLayers):
                                    linearmodel = LinearRegression()
                                    print(f'x_data type: {type(x_data[layerIDX][:,-2:])}, x_data shape: {x_data[layerIDX][:,-2:].shape}')
                                    print(f'y_data type: {type(y_data[layerIDX].reshape(-1,1))}, y_data shape: {y_data[layerIDX].reshape(-1,1).shape}')
                                    linearmodel.fit(np.asarray(x_data[layerIDX][:,-2:]), np.asarray(y_data[layerIDX].reshape(-1,1))) # no need to cross-validate here, since we are just using the AP and ML CCF predictors to generate tau residuals
                                    tau_hat[layerIDX] = linearmodel.predict(np.asarray(x_data[layerIDX][:,-2:])).reshape(-1,1)
                                    tau_residuals[layerIDX] = y_data[layerIDX] - tau_hat[layerIDX]

                                # for layerIDX in range(numLayers):
                                #     plt.figure()
                                #     plt.hist(tau_residuals[layerIDX], bins=75, color='black');
                                
                                # for layerIDX in range(numLayers):
                                #     r_squared_regression_APML_tau_layer = r2_score(y_data[layerIDX], tau_hat[layerIDX])
                                #     plt.figure()
                                #     plt.scatter(y_data[layerIDX], tau_hat[layerIDX], color='black', s=0.1);
                                #     plt.title(f'R^2: {r_squared_regression_APML_tau_layer}')

                                x_data = [x_data[layerIDX][:,:-2] for layerIDX in range(numLayers)] #remove AP and ML CCF predictors from x_data once we have generated the tau residuals
                                y_data = tau_residuals

                                if not (check_val := pre_regression_check(verbose, x_data, y_data)):
                                    break
                                best_coef, lasso_weight, bestAlpha, alphas, tauPredictions, bestR2, loss_history_test, loss_history_train, dual_gap_history = layerRegressions(response_dim, n_splits, highMeanPredictorIDXs, x_data, y_data, layerNames, regressionConditions, region_label_filtered, alpha_params, max_iter)
                                
                            predictor_condition_numbers = [np.linalg.cond(x) for x in x_data]
                            mean_fold_coef = [np.mean(best_coef[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
                            sd_fold_coef = [np.std(best_coef[layerIDX][:,:,:],axis=0) for layerIDX in range(numLayers)]
                            sorted_coef = [np.argsort(mean_fold_coef[layerIDX]) for layerIDX in range(numLayers)]
                            
                            model_vals['sd_fold_coef'][reconstruction_type] = sd_fold_coef
                            model_vals['mean_fold_coef'][reconstruction_type] = mean_fold_coef
                            model_vals['bestR2'][reconstruction_type] = bestR2
                            model_vals['sorted_coef'][reconstruction_type] = sorted_coef
                            model_vals['bestAlpha'][reconstruction_type] = bestAlpha
                            model_vals['alphas'][reconstruction_type] = alphas
                            model_vals['lasso_weight'][reconstruction_type] = lasso_weight
                            plotting_data['tauPredictions'][reconstruction_type] = tauPredictions
                            plotting_data['loss_history_test'][reconstruction_type] = loss_history_test
                            plotting_data['loss_history_train'][reconstruction_type] = loss_history_train
                            plotting_data['dual_gap_history'][reconstruction_type] = dual_gap_history
                            plotting_data['predictor_condition_numbers'][reconstruction_type] = predictor_condition_numbers


                        params = {}
                        params['n_splits'] = n_splits
                        params['tauPoolSize'] = tauPoolSize
                        params['numLayers'] = numLayers
                        params['meanExpressionThresh'] = meanExpressionThresh
                        params['meanPredictionThresh'] = meanPredictionThresh
                        params['highMeanPredictorIDXs'] = highMeanPredictorIDXs
                        params['num_precision'] = num_precision
                        params['alpha_precision'] = alpha_precision
                        params['structNum'] = structNum
                        params['regressions_to_start'] = regressions_to_start

                        paths = {}
                        paths['save_path'] = save_path_OSs
                        paths['predictorPathSuffix'] = predictorPathSuffix
                        paths['tauSortedPath'] = tauSortedPath_OSs

                        titles = {}
                        titles['predictorTitle'] = predictorTitle
                        titles['datasetName'] = datasetName
                        titles['layerNames'] = layerNames
                        titles['predictorNamesArray'] = predictorNamesArray
                        titles['predictorEncodeType'] = predictorEncodeType

                        meta_dict = {}
                        meta_dict['line_selection'] = line_selection
                        meta_dict['struct_list'] = struct_list
                        meta_dict['area_colors'] = area_colors
                        meta_dict['plotting_conditions'] = plotting_conditions
                        meta_dict['params'] = params
                        meta_dict['paths'] = paths
                        meta_dict['titles'] = titles
                        meta_dict['model_vals'] = model_vals
                        meta_dict['plotting_data'] = plotting_data

                        output_dir = os.path.join(tauSortedPath, f'{predictorPathSuffix}', f'{datasetName}')
                        os.makedirs(output_dir, exist_ok=True)
                        with open(os.path.join(output_dir, 'plotting_data.pickle'), 'wb') as handle:
                            pickle.dump(meta_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        # temp_path = 'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing\H3\Cux2-Ai96\pooling0.1mm\GenePredictors\Merfish-Imputed'
                        # meta_dict = pickle.load(open(os.path.join(temp_path,f'plotting_data.pickle'), 'rb'))

                        # line_selection = meta_dict['line_selection']
                        # struct_list = meta_dict['struct_list']
                        # area_colors = meta_dict['area_colors']
                        # plotting_conditions = meta_dict['plotting_conditions']
                        # params = meta_dict['params']
                        # paths = meta_dict['paths']
                        # titles = meta_dict['titles']
                        # model_vals = meta_dict['model_vals']
                        # plotting_data = meta_dict['plotting_data']

                        try:
                            plot_regressions(True, True, line_selection, struct_list, area_colors, plotting_conditions, params, paths, titles, model_vals, plotting_data, regressions_to_plot=[0,1,2,3], paper_plotting=True, cross_layers=[0,2])
                        except Exception as e:
                            print(f"An error occurred while plotting regressions: {e}")

        ##############################################
        ### Layer-specific expression correlations ###
        try:
            plot_expression_correlations(layerNames, struct_list, area_colors, mean_expression_standard, save_path)
        except Exception as e:
            print('!!!!! ERROR !!!!!')
            print(f"An error occurred while plotting expression correlations: {e}")

        time_end = datetime.now()
        print(f'Time to run: {time_end - time_load_data}')


if __name__ == '__main__':
    main()
