import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse import csc_matrix
import scipy
from sklearn.preprocessing import StandardScaler
import os
import sys
import platform
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
import anndata


def merfishLoader(savePath,download_base,pilotGeneNames,geneLimit=-1):

    print(f'Loading Merfish-Imputed Dataset...')

    ##############################################################################################################
    ### Code modified from: alleninstitute.github.io/abc_atlas_access/notebooks/merfish_imputed_genes_example.html
    #download_base = Path(r'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing\Seq')
    abc_cache = AbcProjectCache.from_s3_cache(download_base)
    print(abc_cache.current_manifest)
    print(abc_cache.cache.manifest_file_names)
    abc_cache.load_manifest('releases/20240831/manifest.json')


    cell = abc_cache.get_metadata_dataframe(directory='MERFISH-C57BL6J-638850', file_name='cell_metadata_with_cluster_annotation', dtype={"cell_label": str,"neurotransmitter": str})
    cell.set_index('cell_label', inplace=True)

    merfishXYZ = cell.loc[:,['x','y','z','class']]

    merfishCCF = abc_cache.get_metadata_dataframe(directory='MERFISH-C57BL6J-638850-CCF',
                                                  file_name='ccf_coordinates',
                                                  dtype={"cell_label": str})
    merfishCCF.rename(columns={'x': 'x_ccf',
                               'y': 'y_ccf',
                               'z': 'z_ccf'},
                               inplace=True)
    merfishCCF.drop(['parcellation_index'], axis=1, inplace=True)
    merfishCCF.set_index('cell_label', inplace=True)
    merfishCCF = merfishCCF.join(merfishXYZ, how='inner').drop(columns=['x','y','z'])


    imputed_h5ad_path = abc_cache.get_data_path('MERFISH-C57BL6J-638850-imputed', 'C57BL6J-638850-imputed/log2')
    adata = anndata.read_h5ad(imputed_h5ad_path, backed='r')


    allMerfishGeneNames = list(adata.var.gene_symbol)
    geneFamilyList = ['Grin', 'Grm', 'Grik', 'Gria', 'Gabr', 'Kcnj', 'Kcna', 'Kcnn', 'Scn', 'Cacn', 'Clca', 'Clcn']

    with open(os.path.join(savePath,f'familyGenes_merfishImputed.txt'), "w") as file:
        file.write('Gene Family Representation (Merfish-Imputed Dataset):\n\n')

        enriched_gene_names = []
        for currentGeneFamily in geneFamilyList:
            fullGeneFamily = []
            for geneIDX,currentGene in enumerate(allMerfishGeneNames):
                if currentGene[:len(currentGeneFamily)] == currentGeneFamily:
                    fullGeneFamily.append(allMerfishGeneNames[geneIDX])
                    enriched_gene_names.append(allMerfishGeneNames[geneIDX])

            geneText = f'Gene Family {currentGeneFamily}: {fullGeneFamily}\n\n'
            #print(geneText)
            file.write(geneText)
        file.write(f'Full Gene List: {enriched_gene_names}')
    

    unrepresentedPilotGenes = []
    with open(os.path.join(savePath,f'pilot_merfishImputed_geneOverlap.txt'), "w") as file:
        file.write('Gene Overlap (Pilot & Merfish-Imputed Datasets):\n\n')

        for currentGene in pilotGeneNames:
            pilot2merfishIDX = np.where(np.array(allMerfishGeneNames)==currentGene)[0]
            geneIDX_Text = f'Merfish Imputed IDX of {currentGene}: {pilot2merfishIDX}'
            print(geneIDX_Text)
            file.write(geneIDX_Text+'\n')
            if (pilot2merfishIDX.shape[0] > 0) and (np.where(np.array(enriched_gene_names)==currentGene)[0].shape[0] < 1):
                unrepresentedPilotGenes.append(currentGene)


    used_genes_list = enriched_gene_names + unrepresentedPilotGenes
    used_genes_list = used_genes_list[:geneLimit]
    pred = [x in used_genes_list for x in adata.var.gene_symbol]
    gene_filtered = adata.var[pred]


    gene_subset = adata[:, gene_filtered.index].to_df()
    adata.file.close()
    del adata

    gene_subset.rename(columns=gene_filtered.to_dict()['gene_symbol'], inplace=True)

    joined = merfishCCF.join(gene_subset, on='cell_label')

    filter_IT_ET = joined['class'][(joined["class"] == '01 IT-ET Glut')] #for isolating dataset to just cortical IT & ET neurons
    filter_IT_ET = filter_IT_ET.rename("unique_name")

    joined_filtered = joined.join(filter_IT_ET, how='inner')
    joined_filtered = joined_filtered.drop(columns=['class','unique_name'])

    return joined_filtered, allMerfishGeneNames


def pilotLoader(savePath):

    print(f'Loading Pilot Dataset...')

    #projectPath = r'c:\Users\lai7370\OneDrive - Northwestern University\PilotData'
    #PilotData = h5py.File(os.path.join(projectPath,'filt_neurons_fixedbent_CCF.mat'))

    PilotData = scipy.io.loadmat(os.path.join(savePath,'Data','filt_neurons_fixedbent_CCF.mat'))

    gene_data = PilotData['filt_neurons']['expmat'][0][0]
    gene_data_dense = csc_matrix.todense(gene_data)
    gene_names = PilotData['filt_neurons']['genes'][0][0]
    geneNames = [item[0] for item in gene_names[:,0]]
    np.save(os.path.join(savePath,'geneNames_Pilot'),np.asarray(geneNames))

    clustid = PilotData['filt_neurons']['clustid'][0][0]
    fn_clustid = [[id[0] for id in cell] for cell in clustid]
    fn_clustid = np.array(fn_clustid).reshape(-1)

    #fn_slice = PilotData['filt_neurons']['slice'][0][0]
    #fn_slice = np.array(fn_slice).reshape(-1)

    #fn_pos = PilotData['filt_neurons']['pos'][0][0]

    fn_CCF = PilotData['filt_neurons']['CCF'][0][0]

    return gene_data_dense, geneNames, fn_clustid, fn_CCF


def pathSetter(lineSelection):
    my_os = platform.system()

    if my_os == 'Linux':
        lineSelection  = ['Rpb4-Ai96','Cux2-Ai96','C57BL6/J','PV-Ai96'][int(sys.argv[2])]

    projectFolder = "lineFilter" + lineSelection

    if my_os == 'Linux':
        tauPath = os.path.join(r'/mnt/fsmresfiles/Tau_Processing/',projectFolder+'/')
        savePath = os.path.join(r'/mnt/fsmresfiles/Tau_Processing/H3/')
        download_base = Path(r'/mnt/fsmresfiles/Tau_Processing/Seq/')
    if my_os == 'Windows':
        tauPath = os.path.join(r'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing',projectFolder)
        savePath = os.path.join(r'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing\H3')
        download_base = Path(r'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing\Seq')
    
    print(download_base)

    return lineSelection, my_os, tauPath, savePath, download_base