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


def merfishLoader(savePath, download_base, pilotGeneNames, restrict_merfish_imputed_values, geneLimit=-1, manifest_version='20240831', ensured_gene_list=['Cux2'], add_extra_genes=True):
    '''
    Load the Merfish-Imputed Dataset and filter genes based on pilot dataset and ensured gene list.
    Args:
        savePath: Path to save the output files
        download_base: Base path for downloading data
        pilotGeneNames: List of gene names from the pilot dataset
        restrict_merfish_imputed_values: Boolean to restrict values to those in the pilot dataset
        geneLimit: Limit on the number of genes to include, -1 for no limit
        manifest_version: Version of the manifest to use for loading data
        ensured_gene_list: make sure to include these genes, regardless of whether they are imputed or not
    '''

    print(f'\nLoading Merfish-Imputed Dataset...')

    ##############################################################################################################
    ### Code modified from: alleninstitute.github.io/abc_atlas_access/notebooks/merfish_imputed_genes_example.html
    #download_base = Path(r'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing\Seq')
    abc_cache = AbcProjectCache.from_s3_cache(download_base)
    print(abc_cache.current_manifest)
    print(abc_cache.cache.manifest_file_names)
    abc_cache.load_manifest(f'releases/{manifest_version}/manifest.json')

    abc_cache.list_directories



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


    imputed_h5ad_path = abc_cache.get_data_path('MERFISH-C57BL6J-638850-imputed', 'C57BL6J-638850-imputed/log2') #will download data if it doesn't already exist, ~2hr for ~50GB
    print(f'imputed_h5ad_path: {imputed_h5ad_path}')
    adata = anndata.read_h5ad(imputed_h5ad_path, backed='r')


    merfishGenes = list(np.array(abc_cache.get_metadata_dataframe(directory='MERFISH-C57BL6J-638850', file_name='gene').set_index('gene_identifier').gene_symbol))
    allMerfishImputedGeneNames = list(np.array(abc_cache.get_metadata_dataframe(directory='MERFISH-C57BL6J-638850-imputed', file_name='gene').set_index('gene_identifier').gene_symbol)) #list(adata.var.gene_symbol)
    merfish_imputed_long_gene_names = list(np.array(abc_cache.get_metadata_dataframe(directory='MERFISH-C57BL6J-638850-imputed', file_name='gene').set_index('gene_identifier').name))

    channelGeneFamilyList = ['Grin', 'Grm', 'Grik', 'Gria', 'Gabr', 'Kcnj', 'Kcna', 'Kcnn', 'Scn', 'Cacn', 'Clca', 'Clcn', 'Cdh']
    plast_conn_genes = ['FOSB', 'EGR1', 'NPAS4', 'MEF2C', 'FOXP1', 'CAMK2A', 'CAMK2B', 'SYN1', 'SYN2', 'HOMER1', 'SHANK1', 'SHANK2', 'NRXN1', 'NRXN2', 'NRXN3', 'NLGN1', 'NLGN3', 'NCAM1', 'CCND1']
    plast_conn_genes = [gene.capitalize() for gene in plast_conn_genes]
    metabolic_genes = ['Mgll', 'Chst2', 'St6galnac5', 'Car4', 'Dpp4', 'Echdc2', 'Man2a1', 'St6gal1', 'Chst11']
    ion_channel_genes, gaba_receptor_genes, glu_receptor_genes = [], [], []

    with open(os.path.join(savePath,f'familyGenes_merfishImputed.txt'), "w") as file:
        file.write('Gene Family Representation (Merfish-Imputed Dataset):\n\n')

        if add_extra_genes:
            enriched_gene_names_merfish_imputed = plast_conn_genes.copy() + metabolic_genes.copy() # Start with plasticity & connectivity genes, add to this list
        else:
            enriched_gene_names_merfish_imputed = []

        for currentGeneFamily in channelGeneFamilyList:
            fullGeneFamily = []
            for geneIDX, currentGene in enumerate(allMerfishImputedGeneNames):
                if currentGene[:len(currentGeneFamily)] == currentGeneFamily:
                    fullGeneFamily.append(currentGene)
                    if currentGene not in enriched_gene_names_merfish_imputed:
                        enriched_gene_names_merfish_imputed.append(currentGene)
                    
                    if currentGeneFamily in ['Grin', 'Grm', 'Grik', 'Gria']:
                        glu_receptor_genes.append(currentGene)
                    elif currentGeneFamily in ['Gabr']:
                        gaba_receptor_genes.append(currentGene)
                    elif currentGeneFamily in ['Kcnj', 'Kcna', 'Kcnn', 'Scn', 'Cacn', 'Clca', 'Clcn']:
                        ion_channel_genes.append(currentGene)
                    elif currentGeneFamily in ['Cdh']:
                        plast_conn_genes.append(currentGene)

            geneText = f'Gene Family {currentGeneFamily}: {fullGeneFamily}\n\n'
            #print(geneText)
            file.write(geneText)
        file.write(f'Full Gene List ({len(enriched_gene_names_merfish_imputed)} total genes): {enriched_gene_names_merfish_imputed}')
    
    gene_categories = {
        'metabolic_genes': metabolic_genes,
        'ion_channel_genes': ion_channel_genes,
        'gaba_receptor_genes': gaba_receptor_genes,
        'glu_receptor_genes': glu_receptor_genes,
        'plast_conn_genes': plast_conn_genes,
    }

    unrepresentedPilotGenes = []
    with open(os.path.join(savePath,f'pilot_merfishImputed_geneOverlap.txt'), "w") as file:
        file.write('Gene Overlap (Pilot & Merfish-Imputed Datasets):\n\n')

        for currentGene in pilotGeneNames:
            pilot2merfishIDX = np.where(np.array(allMerfishImputedGeneNames)==currentGene)[0]
            geneIDX_Text = f'Merfish Imputed IDX of {currentGene}: {pilot2merfishIDX}'
            #print(geneIDX_Text)
            file.write(geneIDX_Text+'\n')
            if (pilot2merfishIDX.shape[0] > 0) and (np.where(np.array(enriched_gene_names_merfish_imputed)==currentGene)[0].shape[0] < 1):
                unrepresentedPilotGenes.append(currentGene)
                
    unique_enriched_genes = set(enriched_gene_names_merfish_imputed) - set(pilotGeneNames)
    with open(os.path.join(savePath, 'unique_enriched_genes.txt'), "w") as file:
        file.write('Genes in enriched_gene_names_merfish_imputed but not in pilotGeneNames:\n\n')
        for gene in unique_enriched_genes:
            file.write(f'{gene}\n')
        file.write(f'\nNumber of unique enriched genes: {len(unique_enriched_genes)}\n')
    print(f'Number of unique enriched genes: {len(unique_enriched_genes)}')


    if restrict_merfish_imputed_values:
        used_genes_list = list(((set(enriched_gene_names_merfish_imputed) | set(pilotGeneNames)) & set(merfishGenes)) | set(ensured_gene_list))
    else:
        used_genes_list = list(((set(enriched_gene_names_merfish_imputed) | set(pilotGeneNames)) & set(allMerfishImputedGeneNames)) | set(ensured_gene_list)) #enriched_gene_names_merfish_imputed + unrepresentedPilotGenes

    if geneLimit == -1:
        used_genes_list = used_genes_list
    else:
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

    joined = joined.join(filter_IT_ET, how='inner') #joined_filtered = joined.join(filter_IT_ET, how='inner')
    joined = joined.drop(columns=['class','unique_name']) #joined_filtered = joined_filtered.drop(columns=['class','unique_name'])

    enriched_gene_names = list(joined.drop(columns=['x_ccf','y_ccf','z_ccf']).columns)
    used_genes_idxs = np.where(np.isin(allMerfishImputedGeneNames, enriched_gene_names))[0]
    used_gene_longnames = np.array(merfish_imputed_long_gene_names)[used_genes_idxs]

    return joined, allMerfishImputedGeneNames, gene_categories, used_gene_longnames


def pilotLoader(save_path, data_path):

    print(f'\nLoading Pilot Dataset...')

    #projectPath = r'c:\Users\lai7370\OneDrive - Northwestern University\PilotData'
    #PilotData = h5py.File(os.path.join(projectPath,'filt_neurons_fixedbent_CCF.mat'))

    PilotData = scipy.io.loadmat(os.path.join(data_path, 'filt_neurons_fixedbent_CCF.mat'))

    gene_data = PilotData['filt_neurons']['expmat'][0][0]
    gene_data_dense = csc_matrix.todense(gene_data)
    gene_names = PilotData['filt_neurons']['genes'][0][0]
    geneNames = [item[0] for item in gene_names[:,0]]
    np.save(os.path.join(save_path,'geneNames_Pilot'), np.asarray(geneNames))

    clustid = PilotData['filt_neurons']['clustid'][0][0]
    fn_clustid = [[id[0] for id in cell] for cell in clustid]
    fn_clustid = np.array(fn_clustid).reshape(-1)

    #fn_slice = PilotData['filt_neurons']['slice'][0][0]
    #fn_slice = np.array(fn_slice).reshape(-1)

    #fn_pos = PilotData['filt_neurons']['pos'][0][0]

    fn_CCF = PilotData['filt_neurons']['CCF'][0][0]

    return gene_data_dense, geneNames, fn_clustid, fn_CCF


def pathSetter(output_to_repo=True):
    my_os = platform.system()

    if not output_to_repo:
        linux_prepend = r'/mnt/fsmresfiles/Tau_Processing/'
        windows_prepend = r'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing'

        savePath_linux = os.path.join(linux_prepend, 'H3/')
        savePath_windows = os.path.join(windows_prepend, 'H3')
        output_path = [savePath_linux, savePath_windows]

        if my_os == 'Linux':
            download_base = Path(os.path.join(linux_prepend, 'Seq/'))
        if my_os == 'Windows':
            download_base = Path(os.path.join(windows_prepend, 'Seq'))
    
    if output_to_repo:
        output_path = os.path.join(os.getcwd(), "Plotting")
        download_base = os.path.join(os.getcwd(), "Data")
    
    print(f'merfish download_base: {download_base}')
    print(f'download_base contents: {os.listdir(download_base)}')

    return output_path, download_base