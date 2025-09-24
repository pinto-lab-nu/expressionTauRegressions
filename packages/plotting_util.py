import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import platform
import matplotlib.patches as mpatches
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from utils.set_rcParams import paper_rcParams
from geneCategories import gene_categories

# Define color constants
BACKGROUND_GENE_COLOR_PDF = (0, 0, 0)
INTEREST_GENE_COLOR_PDF = (1, 0, 1)
BACKGROUND_GENE_COLOR_SVG = (0.6, 0.6, 0.6)
INTEREST_GENE_COLOR_SVG = (0.5, 0, 0.5)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

category_color_dict = {
    "Ion"       : "green",          # Ion channels and ion homeostasis
    "Gaba"      : "purple",         # GABAergic transmission
    "Glut"      : "#4C42FF",      # Glutamatergic transmission
    "Dev"       : "#FF9900",      # Synaptic function, connectivity, and plasticity, neuronal development
    "Others"    : "#000000"       # Others
}

def hex_to_rgb(hex_color):
    """Convert a hex color to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert an RGB tuple to a hex color."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def color_gradient(values, start_hex, end_hex, L_percentile=2.5, U_percentile=97.5):
    """Map values to a color gradient between two hex colors."""
    # Convert hex colors to RGB
    start_rgb = np.array(hex_to_rgb(start_hex))
    end_rgb = np.array(hex_to_rgb(end_hex))

    min_val = np.percentile(values, L_percentile) # For visualization, just focus on the center of the distribution
    max_val = np.percentile(values, U_percentile) # "..."
    
    # Normalize the values to a 0-1 range
    normalized_vals = (values - min_val) / (max_val - min_val)
    normalized_vals = np.clip(normalized_vals, 0, 1)  # Ensure values stay within range
    
    # Interpolate each RGB component
    colors = start_rgb + (end_rgb - start_rgb) * normalized_vals[:, None]
    colors = colors.astype(int)  # Ensure RGB values are integers
    
    # Convert each interpolated RGB color to hex
    hex_colors = [rgb_to_hex(tuple(color)) for color in colors]
    
    return hex_colors

def gene_category_counter(gene_name, category_counts, gene_categories):
    if gene_name in gene_categories['Ion channels and ion homeostasis']:
        category_counts[0] += 1
    elif gene_name in gene_categories['GABAergic transmission']:
        category_counts[1] += 1
    elif gene_name in gene_categories['Glutamatergic transmission']:
        category_counts[2] += 1
    elif gene_name in gene_categories['Synaptic function, connectivity, and plasticity, neuronal development']:
        category_counts[3] += 1
    elif gene_name in gene_categories['Others']:
        category_counts[4] += 1
    
    return category_counts



def plot_regressions(save_figs, generate_summary, meta_dict, used_merfish_genes_longnames, regressions_to_plot='All', paper_plotting=False,
                     cross_layers='All', super_verbose=False):

    paper_rcParams()
    mpl.rcParams['axes.titlesize'] = 8
    mpl.rcParams['axes.labelsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7

    my_os = platform.system()
    num_print_betas = 10 # Number of betas to print in the output file

    lineSelection = meta_dict['line_selection']
    structList = meta_dict['struct_list']
    areaColors = meta_dict['area_colors']
    plottingConditions = meta_dict['plotting_conditions']
    params = meta_dict['params']
    paths = meta_dict['paths']
    titles = meta_dict['titles']
    model_vals = meta_dict['model_vals']
    plotting_data = meta_dict['plotting_data']

    n_splits = params['n_splits']
    tauPoolSize = params['tauPoolSize']
    numLayers = params['numLayers']
    meanExpressionThresh = params['meanExpressionThresh']
    meanPredictionThresh = params['meanPredictionThresh']
    highMeanPredictorIDXs = params['highMeanPredictorIDXs']
    numPrecision = params['num_precision']
    alphaPrecision = params['alpha_precision']
    structNum = params['structNum']
    
    if regressions_to_plot == 'All':
        regressions_to_plot = params['regressions_to_start']
    
    if cross_layers == 'All':
        cross_layers = list(np.arange(numLayers))

    savePath = paths['save_path'][my_os == 'Windows']
    predictorPathSuffix = paths['predictorPathSuffix']
    tauSortedPath = paths['tauSortedPath'][my_os == 'Windows']

    predictorTitle = titles['predictorTitle']
    datasetName = titles['datasetName']
    layerNames = titles['layerNames']
    predictorNamesArray = titles['predictorNamesArray']
    predictorEncodeType = titles['predictorEncodeType']

    #distractor_genes = plotting_data['distractor_genes']
    mean_expression_standard = plotting_data['mean_expression_standard']
    
    tauPredictions_spatial = plotting_data['tauPredictions']['spatial']
    # tauPredictions_tau = plotting_data['tauPredictions_tau']
    # tauPredictions_XCCF_tau = plotting_data['tauPredictions_XCCF_tau']
    # tauPredictions_X_tauRes = plotting_data['tauPredictions_X_tauRes']
    # loss_history_test_spatial = plotting_data['loss_history_test_spatial']
    # loss_history_test_tau = plotting_data['loss_history_test_tau']
    # loss_history_test_XCCF_tau = plotting_data['loss_history_test_XCCF_tau']
    # loss_history_test_X_tauRes = plotting_data['loss_history_test_X_tauRes']
    # loss_history_train_spatial = plotting_data['loss_history_train_spatial']
    # loss_history_train_tau = plotting_data['loss_history_train_tau']
    # loss_history_train_XCCF_tau = plotting_data['loss_history_train_XCCF_tau']
    # loss_history_train_X_tauRes = plotting_data['loss_history_train_X_tauRes']
    # dual_gap_history_spatial = plotting_data['dual_gap_history_spatial']
    # dual_gap_history_tau = plotting_data['dual_gap_history_tau']
    # dual_gap_history_XCCF_tau = plotting_data['dual_gap_history_XCCF_tau']
    # dual_gap_history_X_tauRes = plotting_data['dual_gap_history_X_tauRes']
    # predictor_condition_numbers_spatial = plotting_data['predictor_condition_numbers_spatial']
    # predictor_condition_numbers_tau = plotting_data['predictor_condition_numbers_tau']
    # predictor_condition_numbers_XCCF_tau = plotting_data['predictor_condition_numbers_XCCF_tau']
    # predictor_condition_numbers_X_tauRes = plotting_data['predictor_condition_numbers_X_tauRes']

    sd_fold_coef_tau = model_vals['sd_fold_coef']['tau']
    sd_fold_coef_spatial = model_vals['sd_fold_coef']['spatial']
    # sd_fold_coef_XCCF_tau = model_vals['sd_fold_coef_XCCF_tau']
    # sd_fold_coef_X_tauRes = model_vals['sd_fold_coef_X_tauRes']
    mean_fold_coef_tau = model_vals['mean_fold_coef']['tau']
    mean_fold_coef_spatial = model_vals['mean_fold_coef']['spatial']
    # mean_fold_coef_XCCF_tau = model_vals['mean_fold_coef_XCCF_tau']
    # mean_fold_coef_X_tauRes = model_vals['mean_fold_coef_X_tauRes']

    # bestR2_spatial = model_vals['bestR2_spatial']
    # bestR2_tau = model_vals['bestR2_tau']
    # bestR2_XCCF_tau = model_vals['bestR2_XCCF_tau']
    # bestR2_X_tauRes = model_vals['bestR2_X_tauRes']
    # sorted_coef_spatial = model_vals['sorted_coef_spatial']
    # sorted_coef_tau = model_vals['sorted_coef_tau']
    # sorted_coef_XCCF_tau = model_vals['sorted_coef_XCCF_tau']
    # sorted_coef_X_tauRes = model_vals['sorted_coef_X_tauRes']
    # bestAlpha_spatial = model_vals['bestAlpha_spatial']
    # bestAlpha_tau = model_vals['bestAlpha_tau']
    # bestAlpha_XCCF_tau = model_vals['bestAlpha_XCCF_tau']
    # bestAlpha_X_tauRes = model_vals['bestAlpha_X_tauRes']
    # alphas_spatial = model_vals['alphas_spatial']
    # alphas_tau = model_vals['alphas_tau']
    # alphas_XCCF_tau = model_vals['alphas_XCCF_tau']
    # alphas_X_tauRes = model_vals['alphas_X_tauRes']
    # lasso_weight_spatial = model_vals['lasso_weight_spatial']
    # lasso_weight_tau = model_vals['lasso_weight_tau']
    # lasso_weight_XCCF_tau = model_vals['lasso_weight_XCCF_tau']
    # lasso_weight_X_tauRes = model_vals['lasso_weight_X_tauRes']

    if datasetName != 'Pilot':
        volcano_log2_fc = plotting_data['volcano_log2_fc']
        volcano_neg_log10_p = plotting_data['volcano_neg_log10_p']

    linearmodel = LinearRegression()

    if predictorTitle == 'Gene Expression':
        gene_color_list = [[] for _ in range(numLayers)]
        for layerIDX in range(numLayers):
            other_genes = []
            for gene_idx, gene_text in enumerate(predictorNamesArray[highMeanPredictorIDXs[layerIDX]]):
                if gene_text in gene_categories['Ion channels and ion homeostasis']:
                    gene_color = category_color_dict["Ion"]
                elif gene_text in gene_categories['GABAergic transmission']:
                    gene_color = category_color_dict["Gaba"]
                elif gene_text in gene_categories['Glutamatergic transmission']:
                    gene_color = category_color_dict["Glut"]
                elif gene_text in gene_categories['Synaptic function, connectivity, and plasticity, neuronal development']:
                    gene_color = category_color_dict["Dev"]
                elif gene_text in gene_categories['Others']:
                    gene_color = category_color_dict["Others"]
                    other_genes.append(gene_text)
                else:
                    gene_color = 'red'
                gene_color_list[layerIDX].append(gene_color)

        #gene_categories['other_genes'] = other_genes
        #gene_categories['other_genes'].extend(gene_categories['metabolic_genes'])

        legend_handles = [
            mpatches.Patch(color=category_color_dict["Ion"], label='Ion channels and ion homeostasis'),
            mpatches.Patch(color=category_color_dict["Glut"], label='Glutamatergic transmission'),
            mpatches.Patch(color=category_color_dict["Gaba"], label='GABAergic transmission'),
            mpatches.Patch(color=category_color_dict["Dev"], label='Synaptic function, connectivity, and plasticity, neuronal development'),
            mpatches.Patch(color=category_color_dict["Others"], label='Others')
        ]

        category_colors = list(category_color_dict.values())

    ###################################################################
    ################### Regression Outputs Plotting ###################
    resampTitle = f'predThresh={meanPredictionThresh}'

    # if volcano_outputs:
    #     if datasetName != 'Pilot':
    #         # Create empty dataframe to hold volcano plot data, columns: 'Transcript', 'Transcript_Fullname', 'Category', 'Fold_Change', 'p-value', 'Layer'
    #         volcano_plot_DF = pd.DataFrame(columns=['Transcript', 'Transcript Fullname', 'Category', 'Expression Cross-Tau Fold Change', 'p-value', 'Transcript Layer'])
            
    #         fc_sig_frac_fig_list = []
    #         fc_sig_frac_pos_fig_list = []
    #         volc_fig_list = []

    #         for layerIDX in cross_layers:
    #             volcano_sig_category_fractions = np.zeros(5)
    #             volcano_sig_positiveFC_category_fractions = np.zeros(5)

    #             if paper_plotting:
    #                 volc_fig, ax = plt.subplots(figsize=(2, 2))
    #             else:
    #                 volc_fig, ax = plt.figure(figsize=(8, 6))
    #             for i, (fc, p, gene_color, gene_name, long_gene_name) in enumerate(zip(volcano_log2_fc[layerIDX], volcano_neg_log10_p[layerIDX], gene_color_list[layerIDX], predictorNamesArray[highMeanPredictorIDXs[layerIDX]], used_merfish_genes_longnames[highMeanPredictorIDXs[layerIDX]])):
    #                 #plt.scatter(volcano_log2_fc[layerIDX], volcano_neg_log10_p[layerIDX], s=3, color='black')
    #                 if (np.abs(fc) < 0.2) or (p < -np.log10(0.001)):
    #                     gene_color = 'lightgrey'
    #                 ax.scatter(fc, p, s=3, color=gene_color)

    #                 current_gene_category = 'Uncategorized'
    #                 if gene_name in gene_categories['Ion channels and ion homeostasis']:
    #                     current_gene_category = 'Ion channels and ion homeostasis'
    #                 elif gene_name in gene_categories['GABAergic transmission']:
    #                     current_gene_category = 'GABAergic transmission'
    #                 elif gene_name in gene_categories['Glutamatergic transmission']:
    #                     current_gene_category = 'Glutamatergic transmission'
    #                 elif gene_name in gene_categories['Synaptic function, connectivity, and plasticity, neuronal development']:
    #                     current_gene_category = 'Synaptic function, connectivity, and plasticity, neuronal development'
    #                 elif gene_name in gene_categories['Others']:
    #                     current_gene_category = 'Others'

    #                 if (np.abs(fc) >= 0.2) and (p >= -np.log10(0.001)):
    #                     ax.annotate(gene_name, (fc, p), fontsize=8, alpha=1.0, color=gene_color)
    #                     volcano_sig_category_fractions = gene_category_counter(gene_name, volcano_sig_category_fractions, gene_categories)
    #                 if (fc >= 0.2) and (p >= -np.log10(0.001)):
    #                     volcano_sig_positiveFC_category_fractions = gene_category_counter(gene_name, volcano_sig_positiveFC_category_fractions, gene_categories)
                    
    #                 volcano_plot_DF = pd.concat([volcano_plot_DF, pd.DataFrame({'Transcript': [gene_name], 'Transcript Fullname': [long_gene_name], 'Category': [current_gene_category], 'Expression Cross-Tau Fold Change': [fc], 'p-value': [10**(-p)], 'Transcript Layer': [layerNames[layerIDX]]})], ignore_index=True)

    #             #plt.axhline(y=-np.log10(0.001), color='red', linestyle='--', label='FDR p=0.001')
    #             #plt.axvline(x=0, color='black', linestyle='--', label='FC=0')
    #             if not paper_plotting:
    #                 volc_fig.suptitle(f'Gene Expression Fold Change Across $\\tau$ Top & Bottom Quartiles ({datasetName}, {layerNames[layerIDX]})')
    #             ax.set_xlabel(f'$Log_2$ Expression Fold Change')
    #             ax.set_ylabel(f'$-Log_{{10}}$ FDR-corrected p-value')
    #             # Annotate each gene with its name
    #             # for i, gene_name in enumerate(predictorNamesArray):
    #             #     plt.annotate(gene_name, (volcano_log2_fc[layerIDX][i], volcano_neg_log10_p[layerIDX][i]), fontsize=10, alpha=1.0)
    #             plt.legend(
    #                 handles=legend_handles,
    #                 handleheight=0.4,
    #                 handlelength=0.7,
    #                 fontsize=5,
    #             )
    #             if save_figs:
    #                 plt.savefig(os.path.join(tauSortedPath, f'{predictorPathSuffix}', f'{datasetName}', f'{datasetName}_Volcano_{layerNames[layerIDX]}.pdf'), bbox_inches='tight')
    #                 plt.close()
    #             else:
    #                 volc_fig_list.append(volc_fig)

    #             volcano_sig_category_fractions /= np.sum(volcano_sig_category_fractions)
    #             volcano_sig_positiveFC_category_fractions /= np.sum(volcano_sig_positiveFC_category_fractions)
    #             # plot pie chart of volcano sig FC category fractions
    #             fc_sig_frac_fig, ax = plt.subplots(1,1,figsize=(2,2))
    #             ax.pie(
    #                 volcano_sig_category_fractions,
    #                 colors=category_colors,
    #                 autopct='%1.1f%%',
    #                 startangle=90,
    #                 textprops={'fontsize': 10, 'color': '#333333'}
    #             )
    #             if save_figs:
    #                 plt.savefig(os.path.join(tauSortedPath, f'{predictorPathSuffix}',f'{datasetName}', f'{datasetName}_VolcanoSigCategoryFractions_{layerNames[layerIDX]}.pdf'), bbox_inches='tight')
    #                 plt.close()
    #             else:
    #                 fc_sig_frac_fig_list.append(fc_sig_frac_fig)

    #             # plot pie chart of volcano sig positive FC category fractions
    #             fc_sig_frac_pos_fig, ax = plt.subplots(1,1,figsize=(2,2))
    #             ax.pie(
    #                 volcano_sig_positiveFC_category_fractions,
    #                 colors=category_colors,
    #                 autopct='%1.1f%%',
    #                 startangle=90,
    #                 textprops={'fontsize': 10, 'color': '#333333'}
    #             )
    #             if save_figs:
    #                 plt.savefig(os.path.join(tauSortedPath, f'{predictorPathSuffix}',f'{datasetName}', f'{datasetName}_VolcanoSigPositiveFCCategoryFractions_{layerNames[layerIDX]}.pdf'), bbox_inches='tight')
    #                 plt.close()
    #             else:
    #                 fc_sig_frac_pos_fig_list.append(fc_sig_frac_pos_fig)
            
    #         # Dump volcano_plot_DF into a csv file
    #         volcano_plot_DF.to_csv(os.path.join(tauSortedPath, f'{predictorPathSuffix}', f'{datasetName}', f'{datasetName}_Expression_Fold_Change.csv'), index=False)
        
    #     return volc_fig_list, fc_sig_frac_fig_list, fc_sig_frac_pos_fig_list, volcano_plot_DF
    

    # Create empty dataframe to hold regression data, columns: 'Predictor', 'Predictor Fullname', 'Category', 'Regression Type', 'Response Variable', 'Beta', 'Beta SD', 'Beta p-value'
    regression_info_DF = pd.DataFrame(columns=['Predictor', 'Predictor Fullname', 'Category', 'Regression Type', 'Response Variable', 'Beta', 'Beta SD', 'Beta p-value'])

    #for spatialReconstruction in plottingConditions: #[True,False]
    for reconstruction_type in np.asarray(['spatial', 'tau', 'X_tauRes', 'XCCF_tau'])[regressions_to_plot]:
        if reconstruction_type == 'spatial':
            print('Spatial Response Plotting...')
            reconstruction_longname = 'Expression -> CFF'
            spatialReconstruction = True
            recon_type = 'Spatial'
            plottingTitles = ["A-P CCF","M-L CCF"]
            response_dim = 2
            used_predictors_longnames = used_merfish_genes_longnames
            if predictorPathSuffix == 'H3Predictors':
                save_name = f'{datasetName}H3_to_Spatial'
                titleAppend = f'Spatial Reconstruction from {datasetName} {predictorTitle} (pooling={tauPoolSize}mm, {resampTitle})'
                plottingDir = os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'pooling={tauPoolSize}mm')
            else:
                save_name = f'{datasetName}X_to_Spatial'
                titleAppend = f'Spatial Reconstruction from {datasetName} {predictorTitle} ({resampTitle})'
                plottingDir = os.path.join(savePath,'Spatial',f'{predictorPathSuffix}',f'{datasetName}')

        if reconstruction_type == 'tau':
            print('Tau Response Plotting...')
            reconstruction_longname = f'Expression -> {lineSelection} Tau'
            spatialReconstruction = False
            recon_type = '$\\tau$'
            #save_name = 'X_tau'
            plottingTitles = ["Tau"]
            titleAppend = f'{lineSelection} Tau Reconstruction from {datasetName} {predictorTitle} (pooling={tauPoolSize}mm, {resampTitle})'
            response_dim = 1
            plottingDir = os.path.join(tauSortedPath,f'{predictorPathSuffix}',f'{datasetName}')
            used_predictors_longnames = used_merfish_genes_longnames
            if predictorPathSuffix == 'H3Predictors':
                save_name = f'{datasetName}H3_to_Tau'
            else:
                save_name = f'{datasetName}X_to_Tau'

        if reconstruction_type == 'X_tauRes':
            print('Tau Residuals Response Plotting...')
            reconstruction_longname = f'Expression -> {lineSelection} Tau Residuals'
            spatialReconstruction = False
            recon_type = '$\\tau$ Residuals'
            #save_name = 'X_tauRes'
            plottingTitles = ["Tau Residuals"]
            titleAppend = f'{lineSelection} Tau Residuals Reconstruction from {datasetName} {predictorTitle} (pooling={tauPoolSize}mm, {resampTitle})'
            response_dim = 1
            plottingDir = os.path.join(tauSortedPath,f'{predictorPathSuffix}',f'{datasetName}')
            used_predictors_longnames = used_merfish_genes_longnames
            if predictorPathSuffix == 'H3Predictors':
                save_name = f'{datasetName}H3_to_TauRes'
            else:
                save_name = f'{datasetName}X_to_TauRes'

        if reconstruction_type == 'XCCF_tau':
            print('Expression, CCF Predictors -> Tau Response Plotting...')
            reconstruction_longname = f'Expression, CCF -> {lineSelection} Tau'
            spatialReconstruction = False
            recon_type = '$\\beta$[Expression,CCF]=$\\tau$'
            #save_name = 'XCCF_tau'
            plottingTitles = ["Tau"]
            titleAppend = f'X,CCF to {lineSelection} Tau Reconstruction from {datasetName} {predictorTitle} (pooling={tauPoolSize}mm, {resampTitle})'
            response_dim = 1
            plottingDir = os.path.join(tauSortedPath,f'{predictorPathSuffix}',f'{datasetName}')
            predictorNamesArray = np.hstack((predictorNamesArray,['AP','ML']))
            used_predictors_longnames = np.hstack((used_merfish_genes_longnames, ['Anterior-Posterior CCF', 'Medial-Lateral CCF']))
            sorted_coef = model_vals['sorted_coef'][reconstruction_type]
            highMeanPredictorIDXs = [np.concatenate((highMeanPredictorIDXs[layerIDX], np.array([sorted_coef[layerIDX].shape[1]-2, sorted_coef[layerIDX].shape[1]-1]))) for layerIDX in range(numLayers)]
            if predictorPathSuffix == 'H3Predictors':
                save_name = f'{datasetName}H3CCF_to_Tau'
            else:
                save_name = f'{datasetName}XCCF_to_Tau'

        tauPredictions = plotting_data['tauPredictions'][reconstruction_type]
        loss_history_test = plotting_data['loss_history_test'][reconstruction_type]
        loss_history_train = plotting_data['loss_history_train'][reconstruction_type]
        dual_gap_history = plotting_data['dual_gap_history'][reconstruction_type]
        predictor_condition_numbers = plotting_data['predictor_condition_numbers'][reconstruction_type]
        bestR2 = model_vals['bestR2'][reconstruction_type]
        mean_fold_coef = model_vals['mean_fold_coef'][reconstruction_type]
        sorted_coef = model_vals['sorted_coef'][reconstruction_type]
        bestAlpha = model_vals['bestAlpha'][reconstruction_type]
        alphas = model_vals['alphas'][reconstruction_type]
        lasso_weight = model_vals['lasso_weight'][reconstruction_type]
        sd_fold_coef = model_vals['sd_fold_coef'][reconstruction_type]

        if not os.path.exists(plottingDir):
            os.makedirs(plottingDir)

        if generate_summary:
            with open(os.path.join(plottingDir,f'{save_name}_Summary.txt'), "w") as file:
                file.write(f'{titleAppend}\n\n')

            for layerIDX,layerName in enumerate(layerNames):
                bestR2_mean, bestR2_SD = round(np.mean(bestR2[layerIDX,:]),numPrecision), round(np.std(bestR2[layerIDX,:]),numPrecision)
                bestAlpha_mean, bestAlpha_SD = np.mean(bestAlpha[layerIDX,:]), np.std(bestAlpha[layerIDX,:])
                
                with open(os.path.join(plottingDir,f'{save_name}_Summary.txt'), "a") as file:
                    file.write(f'{layerName}, Best R2+-SD:{bestR2_mean}+-{bestR2_SD} (at alpha+-SD={round(bestAlpha_mean,alphaPrecision)}+-{round(bestAlpha_SD,alphaPrecision)})\n')
                    for dim in range(response_dim):
                        if spatialReconstruction:
                            file.write(f'####### {plottingTitles[dim]} #######\n')
                        file.write(f'Highest + Predictors:{predictorNamesArray[highMeanPredictorIDXs[layerIDX]][sorted_coef[layerIDX][dim,:]][-num_print_betas:]}\n')
                        file.write(f'Predictor Weights:{np.round(mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]][-num_print_betas:],3)}\n')
                        file.write(f'SD of Weights:{np.round(sd_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]][-num_print_betas:],3)}\n')
                        file.write(f'Lowest - Predictors:{predictorNamesArray[highMeanPredictorIDXs[layerIDX]][sorted_coef[layerIDX][dim,:]][:num_print_betas]}\n')
                        file.write(f'Predictor Weights:{np.round(mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]][:num_print_betas],3)}\n')
                        file.write(f'SD of Weights:{np.round(sd_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]][:num_print_betas],3)}\n')
                        # if spatialReconstruction and (predictorPathSuffix == 'GenePredictors') and (meanExpressionThresh == 0) and (dim == 0):
                        #     file.write(f'Central A-P betas on margins of Tau beta distribution: {significantTau_centeredSpatial_betas_list[layerIDX]}\n')
                    file.write(f'\n')
        

        #print(layerNames)
        #print(np.array(layerNames)[cross_layers])
        # R^2 by layer for model type
        layer_R2_mean, layer_R2_SD = np.mean(bestR2, axis=1), np.std(bestR2, axis=1)
        r2_layer_modeltype, ax = plt.subplots(figsize=(5,5))
        ax.bar(np.arange(len(cross_layers)), layer_R2_mean[cross_layers], yerr=layer_R2_SD[cross_layers], capsize=5, color='gray', edgecolor='black')
        ax.set_xticks(np.arange(len(cross_layers)), np.array(layerNames)[cross_layers])
        ax.set_ylabel(f'{recon_type} Reconstruction $R^2$ ($\pm$ {n_splits}-fold SD)')
        if not paper_plotting:
            ax.set_title(f'{recon_type} Reconstruction $R^2$ by Layer ({predictorPathSuffix}, {datasetName})')
        if save_figs:
            r2_layer_modeltype.savefig(os.path.join(plottingDir, f'{save_name}_R2_by_layer.pdf'), dpi=600, bbox_inches='tight')
        plt.close()


        beta_dict = {}
        beta_dict['mean_fold_coef'] = mean_fold_coef
        beta_dict['sd_fold_coef'] = sd_fold_coef
        beta_dict['sorted_coef'] = sorted_coef
        beta_dict['layerNames'] = layerNames
        beta_dict['regressionTitle'] = titleAppend
        beta_dict['spatialReconstruction'] = spatialReconstruction
        with open(os.path.join(plottingDir,f'betaDictionary.txt'), 'wb+') as f:
            pickle.dump(beta_dict, f)
        
        if spatialReconstruction and (predictorPathSuffix == 'GenePredictors') and (meanExpressionThresh == 0): #(datasetName == 'Pilot') #removed Pilot qualifier
            for mean_fold_coef_plot, sd_fold_coef_plot, typeTitle in zip([mean_fold_coef_spatial,mean_fold_coef_tau],[sd_fold_coef_spatial,sd_fold_coef_tau],['Spatial','Tau']):
                if typeTitle == 'Spatial':
                    dimArray = [0,1]
                    dimTitleArray = ['A-P','M-L']
                if typeTitle == 'Tau':
                    dimArray = [0]
                    dimTitleArray = ['']
                for dim,dimTitle in zip(dimArray,dimTitleArray):
                    fig, axes = plt.subplots(len(cross_layers), len(cross_layers), figsize=(15,15))
                    axes = np.atleast_2d(axes)
                    if not paper_plotting:
                        plt.suptitle(f'Cross-Layer {typeTitle} {dimTitle} $\\beta$ Correlations, {datasetName} Genes')
                    for fig_idx_0, layerIDX0 in enumerate(cross_layers):
                        for fig_idx_1, layerIDX1 in enumerate(cross_layers):

                            linearmodel.fit(mean_fold_coef_plot[layerIDX1][dim].reshape(-1,1),mean_fold_coef_plot[layerIDX0][dim].reshape(-1,1))
                            beta_pred = linearmodel.predict(mean_fold_coef_plot[layerIDX1][dim].reshape(-1,1))
                            L2L_r2 = r2_score(mean_fold_coef_plot[layerIDX0][dim].reshape(-1,1), beta_pred)
                            
                            axes[fig_idx_0, fig_idx_1].set_title(f'$R^2$={round(L2L_r2,3)}')
                        
                            for i, predictorText in enumerate(predictorNamesArray):
                                axes[fig_idx_0, fig_idx_1].scatter(mean_fold_coef_plot[layerIDX1][dim][i], mean_fold_coef_plot[layerIDX0][dim][i], color=gene_color_list[0][i],s=0.15)
                                axes[fig_idx_0, fig_idx_1].errorbar(mean_fold_coef_plot[layerIDX1][dim][i], mean_fold_coef_plot[layerIDX0][dim][i], xerr=sd_fold_coef_plot[layerIDX1][dim][i], yerr=sd_fold_coef_plot[layerIDX0][dim][i], fmt="o", color=gene_color_list[layerIDX][i], markersize=0.15, elinewidth=0.15)
                                # text_color = 'red' if predictorText in distractor_genes else 'black'
                                axes[fig_idx_0, fig_idx_1].annotate(predictorText, (mean_fold_coef_plot[layerIDX1][dim][i], mean_fold_coef_plot[layerIDX0][dim][i]), fontsize=3, color=gene_color_list[0][i])
                            if layerIDX1 == 0:
                                axes[fig_idx_0, fig_idx_1].set_ylabel(f"{layerNames[layerIDX0]} {dimTitle}$\\beta$")
                            if layerIDX0 == numLayers-1:
                                axes[fig_idx_0, fig_idx_1].set_xlabel(f"{layerNames[layerIDX1]} {dimTitle}$\\beta$")
                    axes[fig_idx_0, fig_idx_1].legend(handles=legend_handles, loc='upper left', fontsize='small', title='Gene Categories')
                    if save_figs:
                        plt.savefig(os.path.join(savePath,'Spatial',f'crossLayer_{datasetName}_{typeTitle}{dimTitle}_Beta_Correlations.pdf'),dpi=600,bbox_inches='tight')
                    plt.close()
        
        AP_v_tau_beta_fig = None
        AP_v_tau_beta_pie_list = []
        if spatialReconstruction and (predictorPathSuffix == 'GenePredictors') and (meanExpressionThresh == 0):
            tau_L_cutoff = 10 #Lower percentile cuttoff for Tau coefficients (used for marginal gene selection)
            spatial_L_cutoff = 10 #Lower percentile cuttoff for Spatial coefficients (used for central gene selection)
            plotting_percentile_offset = 5 #Percentile offset for selection criteria (just for visualization)
            
            tau_U_cutoff = 100 - tau_L_cutoff
            spatial_U_cutoff = 100 - spatial_L_cutoff
            typeTitle = 'A-P vs Tau'
            dim = 0
            tau_L_cutoff_plot = [tau_L_cutoff] #[tau_L_cutoff-plotting_percentile_offset, tau_L_cutoff, tau_L_cutoff+plotting_percentile_offset]
            spatial_L_cutoff_plot = [spatial_L_cutoff] #[spatial_L_cutoff-plotting_percentile_offset, spatial_L_cutoff, spatial_L_cutoff+plotting_percentile_offset]
            tau_U_cutoff_plot = 100 - np.array(tau_L_cutoff_plot)
            spatial_U_cutoff_plot = 100 - np.array(spatial_L_cutoff_plot)

            # tau_percentile_colors = color_gradient(tau_L_cutoff_plot, '#9999ff', '#0000cc', 0, 100) #Blues (light to dark)
            # spatial_percentile_colors = color_gradient(spatial_L_cutoff_plot, '#ff9999', '#cc0000', 0, 100) #Reds (light to dark)
            tau_percentile_colors = ['grey']
            spatial_percentile_colors = ['grey']

            paper_font = 'Arial'
            for fileType in ['.pdf']: #,'.svg']:
                significantTau_centeredSpatial_betas_list = []
                # if paper_plotting:
                #     cross_layers = [0] # layer 2/3 only for paper
                file_append = 'Layer2_3' if paper_plotting else ''
                AP_v_tau_beta_fig, axes = plt.subplots(len(cross_layers), len(cross_layers))
                axes = np.atleast_2d(axes)
                if not paper_plotting:
                    plt.suptitle(f'Cross-Layer {typeTitle} $\\beta$ Correlations, {datasetName} Genes\n(purple labels: marginal {tau_L_cutoff*2}% of Tau $\\beta$s AND central {spatial_U_cutoff-spatial_L_cutoff}% of Spatial $\\beta$s)', fontname=paper_font)
                for fig_idx_0, layerIDX0 in enumerate(cross_layers):
                    for fig_idx_1, layerIDX1 in enumerate(cross_layers):

                        linearmodel.fit(mean_fold_coef_tau[layerIDX1][dim].reshape(-1,1), mean_fold_coef_spatial[layerIDX0][dim].reshape(-1,1))
                        beta_pred = linearmodel.predict(mean_fold_coef_tau[layerIDX1][dim].reshape(-1,1))
                        L2L_r2 = r2_score(mean_fold_coef_spatial[layerIDX0][dim].reshape(-1,1), beta_pred)

                        tau_beta_L = np.percentile(mean_fold_coef_tau[layerIDX1][dim], tau_L_cutoff)
                        tau_beta_U = np.percentile(mean_fold_coef_tau[layerIDX1][dim], tau_U_cutoff)
                        extreme_tau_betas = np.where((mean_fold_coef_tau[layerIDX1][dim] <= tau_beta_L) | (mean_fold_coef_tau[layerIDX1][dim] >= tau_beta_U))
                        extreme_tau_genes_set = set(predictorNamesArray[extreme_tau_betas[0]])

                        spatial_beta_L = np.percentile(mean_fold_coef_spatial[layerIDX0][dim], spatial_L_cutoff)
                        spatial_beta_U = np.percentile(mean_fold_coef_spatial[layerIDX0][dim], spatial_U_cutoff)
                        center_spatial_betas = np.where((mean_fold_coef_spatial[layerIDX0][dim] >= spatial_beta_L) & (mean_fold_coef_spatial[layerIDX0][dim] <= spatial_beta_U))
                        center_spatial_genes_set = set(predictorNamesArray[center_spatial_betas[0]])

                        significant_tau_centered_spatial_predictors = list(extreme_tau_genes_set & center_spatial_genes_set)
                        significantGenesIDXs = [np.where(predictorNamesArray == sigGene)[0][0] for sigGene in significant_tau_centered_spatial_predictors]

                        significant_tau_centered_spatial_predictors_category_fractions = np.zeros(5)
                        for sig_tau_nonsig_ccf_gene in significant_tau_centered_spatial_predictors:
                            significant_tau_centered_spatial_predictors_category_fractions = gene_category_counter(sig_tau_nonsig_ccf_gene, significant_tau_centered_spatial_predictors_category_fractions, gene_categories)
                        significant_tau_centered_spatial_predictors_category_fractions /= np.sum(significant_tau_centered_spatial_predictors_category_fractions)

                        if layerIDX0 == layerIDX1:
                            significantTau_centeredSpatial_betas_list.append(significant_tau_centered_spatial_predictors)

                        if fileType == '.pdf':
                            backgroundGeneColor = BACKGROUND_GENE_COLOR_PDF
                            interestGeneColor = INTEREST_GENE_COLOR_PDF
                            linewidth = 1.0
                        elif fileType == '.svg':
                            backgroundGeneColor = BACKGROUND_GENE_COLOR_SVG
                            interestGeneColor = INTEREST_GENE_COLOR_SVG
                            linewidth = 0.3
                        
                        colorArray = np.array([backgroundGeneColor for _ in predictorNamesArray])
                        colorArray[significantGenesIDXs] = interestGeneColor

                        axes[fig_idx_0, fig_idx_1].set_title(f'$R^2$={round(L2L_r2,3)}', fontname=paper_font)
                        # axes[fig_idx_0, fig_idx_1].scatter(mean_fold_coef_tau[layerIDX1][dim], mean_fold_coef_spatial[layerIDX0][dim],
                        #                                     color=colorArray, edgecolors=colorArray, s=0.15)
                        
                        tau_beta_L_percentiles = np.percentile(mean_fold_coef_tau[layerIDX1][dim], tau_L_cutoff_plot)
                        tau_beta_U_percentiles = np.percentile(mean_fold_coef_tau[layerIDX1][dim], tau_U_cutoff_plot)
                        spatial_beta_L_percentiles = np.percentile(mean_fold_coef_spatial[layerIDX0][dim], spatial_L_cutoff_plot)
                        spatial_beta_U_percentiles = np.percentile(mean_fold_coef_spatial[layerIDX0][dim], spatial_U_cutoff_plot)
                        tau_coef_min = np.min(np.array(mean_fold_coef_tau))
                        tau_coef_max = np.max(np.array(mean_fold_coef_tau))
                        spatial_coef_min = np.min(np.array(mean_fold_coef_spatial))
                        spatial_coef_max = np.max(np.array(mean_fold_coef_spatial))
                        for cutoff_IDX in range(len(tau_L_cutoff_plot)):
                            axes[fig_idx_0, fig_idx_1].vlines(tau_beta_L_percentiles[cutoff_IDX], spatial_coef_min, spatial_coef_max, colors=tau_percentile_colors[cutoff_IDX], linewidth=linewidth, linestyle='dashed', label=f'{tau_L_cutoff_plot[cutoff_IDX]}-{tau_U_cutoff_plot[cutoff_IDX]}th ' + r'$\tau$ $\beta$ percentiles')
                            axes[fig_idx_0, fig_idx_1].vlines(tau_beta_U_percentiles[cutoff_IDX], spatial_coef_min, spatial_coef_max, colors=tau_percentile_colors[cutoff_IDX], linewidth=linewidth, linestyle='dashed')
                            axes[fig_idx_0, fig_idx_1].hlines(spatial_beta_L_percentiles[cutoff_IDX], tau_coef_min, tau_coef_max, colors=spatial_percentile_colors[cutoff_IDX], linewidth=linewidth, linestyle='dashed', label=f'{spatial_L_cutoff_plot[cutoff_IDX]}-{spatial_U_cutoff_plot[cutoff_IDX]}th ' + r'spatial $\beta$ percentiles')
                            axes[fig_idx_0, fig_idx_1].hlines(spatial_beta_U_percentiles[cutoff_IDX], tau_coef_min, tau_coef_max, colors=spatial_percentile_colors[cutoff_IDX], linewidth=linewidth, linestyle='dashed')

                        if layerIDX0 == 0 and layerIDX1 == len(layerNames)-1:
                            axes[fig_idx_0, fig_idx_1].legend(loc='upper right', prop={'family': paper_font, 'size': 5})

                        # # Calculate the point density over a grid
                        # x_vals = mean_fold_coef_tau[layerIDX1][dim]
                        # y_vals = mean_fold_coef_spatial[layerIDX0][dim]

                        # # Define grid
                        # X, Y = np.mgrid[x_vals.min():x_vals.max():100j, y_vals.min():y_vals.max():100j]
                        # positions = np.vstack([X.ravel(), Y.ravel()])
                        # values = np.vstack([x_vals, y_vals])
                        # kernel = gaussian_kde(values)
                        # Z = np.reshape(kernel(positions).T, X.shape)

                        # # Plot the contours
                        # levels = np.linspace(0.5, 0.9, 5)
                        # contour_levels = np.percentile(Z, levels * 100)
                        # level_color = color_gradient(levels, '#809fff', '#001a66', 0, 100)
                        # contour_labels = [f'{int(level * 100)}th density percentile' for level in levels]

                        # # Create proxy artists for the legend (only once)
                        # if layerIDX0 == 0 and layerIDX1 == 0:
                        #     proxy_artists = [plt.Line2D([0], [0], color=level_color[i], linewidth=linewidth) for i in range(len(levels))]

                        # axes[layerIDX0, layerIDX1].contour(
                        #     X, Y, Z, levels=contour_levels, colors=level_color, alpha=0.9, linewidths=linewidth
                        # )

                        # # Add the legend to the first subplot
                        # if layerIDX0 == 0 and layerIDX1 == 0:
                        #     axes[layerIDX0, layerIDX1].legend(proxy_artists, contour_labels, loc='upper right', prop={'family': paper_font})
                            
                        for i, predictorText in enumerate(predictorNamesArray):
                            #gene_color = 'red' if predictorText in distractor_genes else colorArray[i]

                            if paper_plotting:
                                if np.where(predictorNamesArray[significantGenesIDXs] == predictorText)[0].shape[0] > 0:
                                    gene_color = gene_color_list[0][i]
                                    axes[fig_idx_0, fig_idx_1].annotate(predictorText,
                                                                        (mean_fold_coef_tau[layerIDX1][dim][i], mean_fold_coef_spatial[layerIDX0][dim][i]),
                                                                        color=gene_color) #, fontsize=3, fontname=paper_font)
                                else:
                                    gene_color = 'lightgrey'
                            else:
                                if ((fileType == '.eps') and (np.where(predictorNamesArray[significantGenesIDXs] == predictorText)[0].shape[0] > 0)) or (fileType == '.pdf'):
                                    gene_color = gene_color_list[0][i]
                                    axes[fig_idx_0, fig_idx_1].annotate(predictorText,
                                                                        (mean_fold_coef_tau[layerIDX1][dim][i], mean_fold_coef_spatial[layerIDX0][dim][i]),
                                                                        color=gene_color) #, fontsize=3, fontname=paper_font)
                                else:
                                    gene_color = 'lightgrey'
                            
                            axes[fig_idx_0, fig_idx_1].errorbar(mean_fold_coef_tau[layerIDX1][dim][i], mean_fold_coef_spatial[layerIDX0][dim][i],
                                                                xerr=sd_fold_coef_tau[layerIDX1][dim][i], yerr=sd_fold_coef_spatial[layerIDX0][dim][i],
                                                                fmt="o", color=gene_color, markersize=3) #, elinewidth=0.15)
                            
                        if (layerIDX1 == 0) or (cross_layers == [0]):
                            axes[fig_idx_0, fig_idx_1].set_ylabel(f"{layerNames[layerIDX0]} A-P $\\beta$", fontname=paper_font)
                        if (layerIDX0 == numLayers-1) or (cross_layers == [0]):
                            axes[fig_idx_0, fig_idx_1].set_xlabel(f"{layerNames[layerIDX1]} $\\tau$ $\\beta$", fontname=paper_font)
                        
                        AP_v_tau_beta_pie, ax_pie = plt.subplots(1,1,figsize=(2,2))
                        ax_pie.pie(
                            significant_tau_centered_spatial_predictors_category_fractions,
                            colors=category_colors,
                            autopct='%1.1f%%',
                            startangle=90,
                            textprops={'fontsize': 10, 'color': '#333333'},
                        )
                        AP_v_tau_beta_pie_list.append(AP_v_tau_beta_pie)
                        if save_figs:
                            AP_v_tau_beta_pie.savefig(os.path.join(savePath, f'Marginal_Tau_Central_AP_Category_Fractions.pdf'), bbox_inches='tight')
                        plt.close()

                if save_figs:
                    AP_v_tau_beta_fig.savefig(os.path.join(savePath,f'crossLayer_{datasetName}_{typeTitle}_Beta_Correlations_{file_append}{fileType}'), format=fileType[1:], dpi=600, bbox_inches='tight')
                plt.close()

        
        # with open(os.path.join(plottingDir,f'{save_name}_Summary.txt'), "w") as file:
        #     file.write(f'{titleAppend}\n\n')

        for layerIDX, layerName in zip(cross_layers, np.array(layerNames)[cross_layers]):

            model_beta_fig_list = []
            pie_sig_beta_category_fig_list = []
            pie_sig_pos_beta_category_fig_list = []

            if super_verbose:
                ### Loss landscape ###
                fig, axes = plt.subplots(3,n_splits,figsize=(17,18))
                for foldIDX in range(n_splits):
                    for hist_IDX,(history_type,history_title) in enumerate(zip([loss_history_train,loss_history_test,dual_gap_history],['Lasso Training Loss','Lasso Testing Loss','Duality Gap'])):
                        axes[hist_IDX,foldIDX].plot(history_type[layerIDX][foldIDX])
                        axes[hist_IDX,foldIDX].set_xlabel('Iteration')
                        axes[hist_IDX,foldIDX].set_ylabel(f'{history_title}')
                        axes[hist_IDX,foldIDX].set_yscale('log')
                        axes[hist_IDX,foldIDX].set_title(f'fold:{foldIDX}')
                plt.suptitle(f'{titleAppend}, {layerName}, Lasso Loss and Duality Gap by Iteration,\nPredictor Condition Number (of complete matrix): {round(predictor_condition_numbers[layerIDX],numPrecision)}')
                if save_figs:
                    plt.savefig(os.path.join(plottingDir,f'{save_name}_LossConvergence_{layerName}.pdf'),dpi=600,bbox_inches='tight')
                plt.close()
            
            # bestR2_mean, bestR2_SD = round(np.mean(bestR2[layerIDX,:]),numPrecision), round(np.std(bestR2[layerIDX,:]),numPrecision)
            # bestAlpha_mean, bestAlpha_SD = np.mean(bestAlpha[layerIDX,:]), np.std(bestAlpha[layerIDX,:])
            
            # with open(os.path.join(plottingDir,f'{save_name}_Summary.txt'), "a") as file:
            #     file.write(f'{layerName}, Best R2+-SD:{bestR2_mean}+-{bestR2_SD} (at alpha+-SD={round(bestAlpha_mean,alphaPrecision)}+-{round(bestAlpha_SD,alphaPrecision)})\n')
            #     for dim in range(response_dim):
            #         if spatialReconstruction:
            #             file.write(f'####### {plottingTitles[dim]} #######\n')
            #         file.write(f'Highest + Predictors:{predictorNamesArray[highMeanPredictorIDXs[layerIDX]][sorted_coef[layerIDX][dim,:]][-num_print_betas:]}\n')
            #         file.write(f'Predictor Weights:{np.round(mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]][-num_print_betas:],3)}\n')
            #         file.write(f'SD of Weights:{np.round(sd_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]][-num_print_betas:],3)}\n')
            #         file.write(f'Lowest - Predictors:{predictorNamesArray[highMeanPredictorIDXs[layerIDX]][sorted_coef[layerIDX][dim,:]][:num_print_betas]}\n')
            #         file.write(f'Predictor Weights:{np.round(mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]][:num_print_betas],3)}\n')
            #         file.write(f'SD of Weights:{np.round(sd_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]][:num_print_betas],3)}\n')
            #         # if spatialReconstruction and (predictorPathSuffix == 'GenePredictors') and (meanExpressionThresh == 0) and (dim == 0):
            #         #     file.write(f'Central A-P betas on margins of Tau beta distribution: {significantTau_centeredSpatial_betas_list[layerIDX]}\n')
            #     file.write(f'\n')
            
            if reconstruction_type == 'XCCF_tau':
                gene_color_list[layerIDX].extend(['black', 'black'])
            gene_color_list_layer_sorted = list(np.array(gene_color_list[layerIDX])[[sorted_coef[layerIDX][response_dim-1,:]]].flatten()) ### TODO: fix response_dim, was just dim

            if super_verbose:
                fig, axes = plt.subplots(response_dim,1,figsize=(10,10))
                plt.suptitle(f'{titleAppend}, {layerName}, best $R^2\pm$SD: {bestR2_mean}$\pm${bestR2_SD} ($\\alpha\pm$SD: {round(bestAlpha_mean,alphaPrecision)}$\pm${round(bestAlpha_SD,alphaPrecision)})')
                for dim,ax in enumerate(np.atleast_1d(axes)):
                    ax.set_title(f'{plottingTitles[dim]}')
                    for idx, color in enumerate(gene_color_list[layerIDX]):
                        ax.plot(alphas, lasso_weight[layerIDX][:,dim,idx], color=color)
                    ax.vlines(x=bestAlpha_mean, ymin=-10**5, ymax=10**5, color='black')
                    ax.fill_betweenx(y=np.array([-10**5,10**5]), x1=bestAlpha_mean-bestAlpha_SD, x2=bestAlpha_mean+bestAlpha_SD, color='gray', alpha=0.3)
                    ax.set_xscale('log')
                    ax.set_ylim(np.min(lasso_weight[layerIDX][:,:]), np.max(lasso_weight[layerIDX][:,:]))
                    ax.set_xlabel(f'$\\alpha$')
                    ax.set_ylabel(f'$\\beta$')
                if save_figs:
                    plt.savefig(os.path.join(plottingDir,f'{save_name}_LassoWeightsAll_{layerName}.pdf'),dpi=600,bbox_inches='tight')
                plt.close()
            
            if spatialReconstruction:
                fig, ax = plt.subplots(1,1,figsize=(10,10))
                ax.set_title(f'{titleAppend}\nA-P vs M-L $\\beta$ Values\n{layerName}, $\\alpha\pm$SD={round(bestAlpha_mean,alphaPrecision)}$\pm${round(bestAlpha_SD,alphaPrecision)}, $R^2\pm$SD={bestR2_mean}$\pm${bestR2_SD}, error:5-fold SD')
                ax.scatter(mean_fold_coef[layerIDX][0,:],mean_fold_coef[layerIDX][1,:],color='black',s=0.5)
                plt.errorbar(mean_fold_coef[layerIDX][0,:], mean_fold_coef[layerIDX][1,:], xerr=sd_fold_coef[layerIDX][0,:], yerr=sd_fold_coef[layerIDX][1,:], fmt="o", color='black')
                for i, predictorText in enumerate(predictorNamesArray[highMeanPredictorIDXs[layerIDX]]):
                    ax.annotate(predictorText, (mean_fold_coef[layerIDX][0,i], mean_fold_coef[layerIDX][1,i]), color=gene_color_list_layer_sorted[i])
                ax.set_xlabel(f'A-P $\\beta$')
                ax.set_ylabel(f'M-L $\\beta$')
                if save_figs:
                    plt.savefig(os.path.join(plottingDir,f'{save_name}_APvsML_BetaWeights_{layerName}.pdf'),dpi=600,bbox_inches='tight')
                plt.close()

            
            for dim in range(response_dim):
                upper_coef = mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]] + sd_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]]
                lower_coef = mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]] - sd_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]]
                # find the indicies of the predictors whose error bars cross y=0
                zero_crossing_idx = np.where((upper_coef >= 0) & (lower_coef <= 0))[0]

                beta_means = mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]]
                beta_stds = sd_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]]
                df = n_splits-1
                # t-statistics (testing H0: beta = 0)
                t_stats = beta_means / (beta_stds / np.sqrt(n_splits))
                # Two-sided p-values using t-distribution
                pvals = 2 * stats.t.sf(np.abs(t_stats), df=df)
                # fill in nan p-values with 1
                pvals = np.nan_to_num(pvals, nan=1.0)
                # Optionally apply FDR (Benjamini-Hochberg)
                reject, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
                reject_pvals_idxs = np.where(pvals > 0.05)[0]

                inverse_coef_sorted_idx = np.argsort(sorted_coef[layerIDX][dim, :])
                pvals_unsorted = pvals[inverse_coef_sorted_idx]

                regression_sig_category_fractions = np.zeros(5)
                regression_sig_posBeta_category_fractions = np.zeros(5)

                for idx, current_gene in enumerate(predictorNamesArray[highMeanPredictorIDXs[layerIDX][sorted_coef[layerIDX][dim,:]]]):
                    if idx not in reject_pvals_idxs:
                        if current_gene in gene_categories['Ion channels and ion homeostasis']:
                            regression_sig_category_fractions[0] += 1
                        elif current_gene in gene_categories['GABAergic transmission']:
                            regression_sig_category_fractions[1] += 1
                        elif current_gene in gene_categories['Glutamatergic transmission']:
                            regression_sig_category_fractions[2] += 1
                        elif current_gene in gene_categories['Synaptic function, connectivity, and plasticity, neuronal development']:
                            regression_sig_category_fractions[3] += 1
                        elif current_gene in gene_categories['Others']:
                            regression_sig_category_fractions[4] += 1
                    if idx not in reject_pvals_idxs and beta_means[idx] > 0:
                        if current_gene in gene_categories['Ion channels and ion homeostasis']:
                            regression_sig_posBeta_category_fractions[0] += 1
                        elif current_gene in gene_categories['GABAergic transmission']:
                            regression_sig_posBeta_category_fractions[1] += 1
                        elif current_gene in gene_categories['Glutamatergic transmission']:
                            regression_sig_posBeta_category_fractions[2] += 1
                        elif current_gene in gene_categories['Synaptic function, connectivity, and plasticity, neuronal development']:
                            regression_sig_posBeta_category_fractions[3] += 1
                        elif current_gene in gene_categories['Others']:
                            regression_sig_posBeta_category_fractions[4] += 1
                
                regression_sig_category_fractions /= np.sum(regression_sig_category_fractions)
                regression_sig_posBeta_category_fractions /= np.sum(regression_sig_posBeta_category_fractions)

                # plot pie chart of regression, significant model betas in the different category fractions
                pie_sig_beta_category_fig, ax = plt.subplots(1,1,figsize=(2,2))
                ax.pie(
                    regression_sig_category_fractions,
                    colors=category_colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 10, 'color': '#333333'},
                )
                pie_sig_beta_category_fig_list.append(pie_sig_beta_category_fig)
                if save_figs:
                    plt.savefig(os.path.join(plottingDir,f'{save_name}_RegressionSigCategoryFractions_{layerName}.pdf'), bbox_inches='tight')
                plt.close()

                # plot pie chart of regression, significant *positive* model betas by category
                pie_sig_pos_beta_category_fig, ax = plt.subplots(1,1,figsize=(2,2))
                ax.pie(
                    regression_sig_posBeta_category_fractions,
                    colors=category_colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 10, 'color': '#333333'},
                )
                pie_sig_pos_beta_category_fig_list.append(pie_sig_pos_beta_category_fig)
                if save_figs:
                    plt.savefig(os.path.join(plottingDir,f'{save_name}_RegressionSigPosBetaCategoryFractions_{layerName}.pdf'), bbox_inches='tight')
                plt.close()

                for i, (current_predictor, current_fullname, current_beta, current_beta_sd, current_pval) in enumerate(zip(predictorNamesArray[highMeanPredictorIDXs[layerIDX]], used_predictors_longnames[highMeanPredictorIDXs[layerIDX]], mean_fold_coef[layerIDX][dim,:], sd_fold_coef[layerIDX][dim,:], pvals_unsorted)):
                    
                    current_category = 'Uncategorized'
                    if current_predictor in gene_categories['Ion channels and ion homeostasis']:
                        current_category = 'Ion channels and ion homeostasis'
                    elif current_predictor in gene_categories['GABAergic transmission']:
                        current_category = 'GABAergic transmission'
                    elif current_predictor in gene_categories['Glutamatergic transmission']:
                        current_category = 'Glutamatergic transmission'
                    elif current_predictor in gene_categories['Synaptic function, connectivity, and plasticity, neuronal development']:
                        current_category = 'Synaptic function, connectivity, and plasticity, neuronal development'
                    elif current_predictor in gene_categories['Others']:
                        current_category = 'Others'
                    elif current_predictor in ['ML', 'AP']:
                        current_category = 'Spatial Coordinate'

                    regression_info_DF = pd.concat([regression_info_DF, pd.DataFrame({'Predictor': [current_predictor], 'Predictor Fullname': [current_fullname], 'Category': [current_category], 'Regression Type': [f'{layerNames[layerIDX]} {reconstruction_longname}'], 'Response Variable': [plottingTitles[dim]], 'Beta': [current_beta], 'Beta SD': [current_beta_sd], 'Beta p-value': [current_pval]})], ignore_index=True)


            model_beta_fig, axes = plt.subplots(response_dim,1,figsize=(5,3))
            if not paper_plotting:
                plt.suptitle(f'{titleAppend}\n{layerName}, $\\alpha\pm$SD={round(bestAlpha_mean,alphaPrecision)}$\pm${round(bestAlpha_SD,alphaPrecision)}, $R^2\pm$SD={bestR2_mean}$\pm${bestR2_SD}, error:5-fold SD')
            for dim,ax in enumerate(np.atleast_1d(axes)):
                if not paper_plotting:
                    ax.set_title(f'{plottingTitles[dim]}')
                for i, predictorText in enumerate(predictorNamesArray[highMeanPredictorIDXs[layerIDX][sorted_coef[layerIDX][dim,:]]]):
                    #ax.scatter(np.arange(0,gene_data_dense.shape[1],1),mean_fold_coef[sorted_coef],color='black')
                    #ax.errorbar(np.arange(0,highMeanPredictorIDXs[layerIDX].shape[0],1), mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]], yerr=sd_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,:]], fmt="o", color=gene_color)
                    if i in reject_pvals_idxs:
                        gene_color = 'lightgrey'
                    else:
                        gene_color = gene_color_list_layer_sorted[i]
                    ax.errorbar(i, mean_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,i]], yerr=sd_fold_coef[layerIDX][dim,sorted_coef[layerIDX][dim,i]], fmt="o", color=gene_color, markersize=0, elinewidth=0.8)
                ax.hlines(y=0, xmin=0, xmax=highMeanPredictorIDXs[layerIDX].shape[0],color='black',alpha=0.5,linestyles='dashed')
                ax.set_xticks(np.arange(0, highMeanPredictorIDXs[layerIDX].shape[0], 1))
                ax.set_xticklabels(predictorNamesArray[highMeanPredictorIDXs[layerIDX]][sorted_coef[layerIDX][dim,:]], rotation=90)
                ax.set_ylabel(f'$\\beta$')
                # for tick_idx, (tick_label, color) in enumerate(zip(ax.get_xticklabels(), gene_color_list_layer_sorted)):
                #     if tick_idx in zero_crossing_idx:
                #         tick_label.set_text('')
                #     tick_label.set_color(color)
                # ax.figure.canvas.draw_idle()  # Ensure the changes are rendered
                labels = [label.get_text() for label in ax.get_xticklabels()]
                for idx in reject_pvals_idxs:
                    labels[idx] = ''
                ax.set_xticklabels(labels)
                for tick_label, color in zip(ax.get_xticklabels(), gene_color_list_layer_sorted):
                    tick_label.set_color(color)
                ax.figure.canvas.draw_idle() # Ensure the changes are rendered
                plt.legend(
                    handles=legend_handles,
                    handleheight=0.6,
                    handlelength=1.0,
                    fontsize=7,
                )
            model_beta_fig_list.append(model_beta_fig)
            if save_figs:
                plt.savefig(os.path.join(plottingDir,f'{save_name}_LassoWeights_{layerName}.pdf'), bbox_inches='tight')
            plt.close()
            
            if reconstruction_type != 'XCCF_tau':
                fig1, axes = plt.subplots(response_dim,1,figsize=(12,12))
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
                if save_figs:
                    plt.savefig(os.path.join(plottingDir,f'{save_name}_RegionLevels_{layerName}.pdf'),dpi=600,bbox_inches='tight')
                plt.close()

            for dim in range(response_dim):
                currentPlottingTitle = plottingTitles[dim]
                fig, axes = plt.subplots(3,n_splits,figsize=(20,18))
                plt.suptitle(f'{titleAppend}, {layerName}')
                for foldIDX in range(n_splits):
                    #cell_region_IDX = (cell_region_H2layerFiltered[layerIDX][test_index,:]).astype(int).reshape(-1)
                    test_y = tauPredictions[layerIDX][foldIDX][:,dim]
                    pred_y = tauPredictions[layerIDX][foldIDX][:,dim+response_dim]
                    cell_region_IDX = tauPredictions[layerIDX][foldIDX][:,-1].astype(int)
                    cell_region_colors = np.asarray(areaColors)[cell_region_IDX]

                    for regionIDX,region in enumerate(structList):
                        regionR2IDXs = np.where(cell_region_IDX == regionIDX)
                        label = f'{region}' #R2: {round(r2_score(test_y[regionR2IDXs],pred_y[regionR2IDXs]),3)}'
                        axes[0,foldIDX].scatter(test_y[regionR2IDXs],pred_y[regionR2IDXs],color=areaColors[regionIDX],label=label,s=1)

                    #ax.scatter(test_y,pred_y,color=cell_region_colors,s=1)
                    #ax.axis('equal')
                    #ax.set_xlim(4,10)
                    #ax.set_ylim(4,10)
                    axes[0,foldIDX].set_title(f'Fold: {foldIDX}')
                    axes[0,foldIDX].set_xlabel(f'True {currentPlottingTitle} (standardized)')
                    if foldIDX==0:
                        axes[0,foldIDX].legend()
                    
                    foldTrueMeans, foldTrueSEM, foldPredMeans, foldPredSEM = [],[],[],[]
                    for regionIDX in range(structNum):
                        predictionRegionIDXs = np.where(tauPredictions[layerIDX][foldIDX][:,-1] == regionIDX)

                        foldTrueMeans.append(np.mean(tauPredictions[layerIDX][foldIDX][predictionRegionIDXs,dim]))
                        foldTrueSEM.append(np.std(tauPredictions[layerIDX][foldIDX][predictionRegionIDXs,dim]) / np.sqrt(predictionRegionIDXs[0].shape[0]))
                        foldPredMeans.append(np.mean(tauPredictions[layerIDX][foldIDX][predictionRegionIDXs,dim+response_dim]))
                        foldPredSEM.append(np.std(tauPredictions[layerIDX][foldIDX][predictionRegionIDXs,dim+response_dim]) / np.sqrt(predictionRegionIDXs[0].shape[0]))

                    axes[1,foldIDX].bar(structList,foldTrueMeans,color=areaColors)
                    axes[1,foldIDX].errorbar(structList,foldTrueMeans,foldTrueSEM,fmt='o',markersize=0.15,color='black')
                    axes[2,foldIDX].bar(structList,foldPredMeans,color=areaColors)
                    axes[2,foldIDX].errorbar(structList,foldPredMeans,foldPredSEM,fmt='o',markersize=0.15,color='black')

                axes[0,0].set_ylabel(f'Predicted {currentPlottingTitle} (standardized)')
                axes[1,0].set_ylabel(f'True {currentPlottingTitle} by Region\n(standardized mean +- SEM)')
                axes[2,0].set_ylabel(f'Predicted {currentPlottingTitle} by Region\n(standardized mean +- SEM)')

                if save_figs:
                    plt.savefig(os.path.join(plottingDir,f'{save_name}_Predicted{currentPlottingTitle}_{layerName}.pdf'),dpi=600,bbox_inches='tight')
                plt.close()

            #if poolIndex == 1:
            #for layerIDX,layerName in enumerate(layerNames):
            apR2,mlR2 = [],[]
            for foldIDX in range(n_splits):
                apR2.append(r2_score(tauPredictions_spatial[layerIDX][foldIDX][:,0],tauPredictions_spatial[layerIDX][foldIDX][:,2]))
                mlR2.append(r2_score(tauPredictions_spatial[layerIDX][foldIDX][:,1],tauPredictions_spatial[layerIDX][foldIDX][:,3]))
            fig, axes = plt.subplots(1,2,figsize=(20,10))
            plt.suptitle(f'{layerName} Cross-Fold Spatial Reconstructions from {predictorEncodeType} {datasetName} {predictorTitle}\n{titleAppend}')
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
            if predictorPathSuffix == 'H3Predictors':
                H3SpatialReconstructionPool = f'_pooling{tauPoolSize}mm'
            else:
                H3SpatialReconstructionPool = ''
            if save_figs:
                plt.savefig(os.path.join(plottingDir,f'{save_name}_SpatialReconstruction_{layerName}.pdf'),dpi=600,bbox_inches='tight')
            plt.close()
            
    # Dump regression_info_DF into a CSV
    regression_info_DF.to_csv(os.path.join(plottingDir, f'{datasetName}_Regression_Betas.csv'), index=False)

    return r2_layer_modeltype, model_beta_fig_list, pie_sig_beta_category_fig_list, pie_sig_pos_beta_category_fig_list, regression_info_DF, AP_v_tau_beta_fig, AP_v_tau_beta_pie_list

    # highMeanPredictorIDXs = params['highMeanPredictorIDXs']
    # predictorNamesArray = titles['predictorNamesArray']
    # for layerIDX,layerName in enumerate(layerNames):
    #     fig, ax = plt.subplots(1,2,figsize=(15,7))
    #     plt.suptitle(f'{resampTitle} Spatial and {lineSelection} Tau Reconstruction from {datasetName} {predictorTitle}\n{lineSelection} Tau vs A-P, M-L $\\beta$ Values\n{layerName}, error:5-fold SD')
    #     ax[0].scatter(mean_fold_coef_tau[layerIDX].reshape(-1), mean_fold_coef_spatial[layerIDX][0,:].reshape(-1),color='black',s=0.5)
    #     ax[0].errorbar(mean_fold_coef_tau[layerIDX].reshape(-1), mean_fold_coef_spatial[layerIDX][0,:].reshape(-1), xerr=sd_fold_coef_tau[layerIDX][0,:], yerr=sd_fold_coef_spatial[layerIDX][0,:], fmt="o", color='black')
    #     ax[1].scatter(mean_fold_coef_tau[layerIDX].reshape(-1), mean_fold_coef_spatial[layerIDX][1,:].reshape(-1),color='black',s=0.5)
    #     ax[1].errorbar(mean_fold_coef_tau[layerIDX].reshape(-1), mean_fold_coef_spatial[layerIDX][1,:].reshape(-1), xerr=sd_fold_coef_tau[layerIDX][0,:], yerr=sd_fold_coef_spatial[layerIDX][1,:], fmt="o", color='black')
    #     for i, predictorText in enumerate(predictorNamesArray[highMeanPredictorIDXs[layerIDX]]):
    #         ax[0].annotate(predictorText, (mean_fold_coef_tau[layerIDX][0,i], mean_fold_coef_spatial[layerIDX][0,i]), color=gene_color_list[layerIDX][i])
    #         ax[1].annotate(predictorText, (mean_fold_coef_tau[layerIDX][0,i], mean_fold_coef_spatial[layerIDX][1,i]), color=gene_color_list[layerIDX][i])
    #     ax[0].set_xlabel(f'Tau $\\beta$')
    #     ax[0].set_ylabel(f'A-P $\\beta$')
    #     ax[1].set_xlabel(f'Tau $\\beta$')
    #     ax[1].set_ylabel(f'M-L $\\beta$')
    #     plt.legend(
    #         handles=legend_handles,
    #         handleheight=0.8,
    #         handlelength=1.2
    #     )
    #     #plt.savefig(os.path.join(plottingDir,f'{datasetName}_{resampTitle}_{lineSelection}Tau_vs_AP&ML_Betas_{layerName}.pdf'),dpi=600,bbox_inches='tight')
    #     plt.savefig(os.path.join(tauSortedPath,f'{predictorPathSuffix}',f'{datasetName}',f'{datasetName}_{resampTitle}_{lineSelection}Tau_vs_AP&ML_Betas_{layerName}.pdf'),dpi=600,bbox_inches='tight')
    #     plt.close()
                    



        # sorted_coef = np.argsort(best_coef[layerIDX,foldIDX,:])
        # fig, ax = plt.subplots(1,1,figsize=(15,8))
        # ax.plot(np.arange(0,gene_data_dense.shape[1],1),best_coef[layerIDX,foldIDX,sorted_coef])
        # ax.set_xticks(np.arange(0, gene_data_dense.shape[1], 1))
        # ax.set_xticklabels(np.array(geneNames)[sorted_coef], rotation=90)

def plot_expression_correlations(layerNames, structList, areaColors, mean_expression_standard, savePath):
    linearmodel = LinearRegression()
    numLayers = len(layerNames)
    structNum = len(structList)
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
                axes[layerIDX0,layerIDX1].scatter(mean_expression_standard[layerIDX0,currentRegion,:],mean_expression_standard[layerIDX1,currentRegion,:],label=structList[currentRegion],color=areaColors[currentRegion],s=0.25)
                if layerIDX1 == 0:
                    axes[layerIDX0,layerIDX1].set_ylabel(f"{layerNames[layerIDX0]}\nGene Expressions")
                if layerIDX0 == numLayers-1:
                    axes[layerIDX0,layerIDX1].set_xlabel(f"{layerNames[layerIDX1]}\nGene Expressions")
                if (layerIDX0 == numLayers-1) and (layerIDX1 == numLayers-1):
                    axes[layerIDX0,layerIDX1].legend()
    plt.savefig(os.path.join(savePath,'Spatial',f'crossLayerGeneExpressionCorrelations.pdf'),dpi=600,bbox_inches='tight')
    plt.close()








def plot_gene_enrichment(save_figs, meta_dict, used_merfish_genes_longnames, paper_plotting=False, cross_layers='All'):

    paper_rcParams()
    mpl.rcParams['axes.titlesize'] = 8
    mpl.rcParams['axes.labelsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7

    my_os = platform.system()
    params = meta_dict['params']
    paths = meta_dict['paths']
    titles = meta_dict['titles']
    plotting_data = meta_dict['plotting_data']
    numLayers = params['numLayers']
    highMeanPredictorIDXs = params['highMeanPredictorIDXs']
    
    if cross_layers == 'All':
        cross_layers = list(np.arange(numLayers))

    savePath = paths['save_path'][my_os == 'Windows']
    predictorPathSuffix = paths['predictorPathSuffix']
    tauSortedPath = paths['tauSortedPath'][my_os == 'Windows']
    predictorTitle = titles['predictorTitle']
    datasetName = titles['datasetName']
    layerNames = titles['layerNames']
    predictorNamesArray = titles['predictorNamesArray']

    if datasetName != 'Pilot':
        volcano_log2_fc = plotting_data['volcano_log2_fc']
        volcano_neg_log10_p = plotting_data['volcano_neg_log10_p']

    if predictorTitle == 'Gene Expression':
        gene_color_list = [[] for _ in range(numLayers)]
        for layerIDX in range(numLayers):
            other_genes = []
            for gene_idx, gene_text in enumerate(predictorNamesArray[highMeanPredictorIDXs[layerIDX]]):
                if gene_text in gene_categories['Ion channels and ion homeostasis']:
                    gene_color = category_color_dict["Ion"]
                elif gene_text in gene_categories['GABAergic transmission']:
                    gene_color = category_color_dict["Gaba"]
                elif gene_text in gene_categories['Glutamatergic transmission']:
                    gene_color = category_color_dict["Glut"]
                elif gene_text in gene_categories['Synaptic function, connectivity, and plasticity, neuronal development']:
                    gene_color = category_color_dict["Dev"]
                elif gene_text in gene_categories['Others']:
                    gene_color = category_color_dict["Others"]
                    other_genes.append(gene_text)
                else:
                    gene_color = 'red'
                gene_color_list[layerIDX].append(gene_color)

        legend_handles = [
            mpatches.Patch(color=category_color_dict["Ion"], label='Ion channels and ion homeostasis'),
            mpatches.Patch(color=category_color_dict["Glut"], label='Glutamatergic transmission'),
            mpatches.Patch(color=category_color_dict["Gaba"], label='GABAergic transmission'),
            mpatches.Patch(color=category_color_dict["Dev"], label='Synaptic function, connectivity, and plasticity, neuronal development'),
            mpatches.Patch(color=category_color_dict["Others"], label='Others')
        ]

        category_colors = list(category_color_dict.values())

    ###############################################################
    ################### Gene Enrichment Outputs ###################

    if datasetName != 'Pilot':
        # Create empty dataframe to hold volcano plot data, columns: 'Transcript', 'Transcript_Fullname', 'Category', 'Fold_Change', 'p-value', 'Layer'
        volcano_plot_DF = pd.DataFrame(columns=['Transcript', 'Transcript Fullname', 'Category', 'Expression Cross-Tau Fold Change', 'p-value', 'Transcript Layer'])
        
        fc_sig_frac_fig_list = []
        fc_sig_frac_pos_fig_list = []
        volc_fig_list = []

        for layerIDX in cross_layers:
            volcano_sig_category_fractions = np.zeros(5)
            volcano_sig_positiveFC_category_fractions = np.zeros(5)

            if paper_plotting:
                volc_fig, ax = plt.subplots(figsize=(2, 2))
            else:
                volc_fig, ax = plt.figure(figsize=(8, 6))
            for i, (fc, p, gene_color, gene_name, long_gene_name) in enumerate(zip(volcano_log2_fc[layerIDX], volcano_neg_log10_p[layerIDX], gene_color_list[layerIDX], predictorNamesArray[highMeanPredictorIDXs[layerIDX]], used_merfish_genes_longnames[highMeanPredictorIDXs[layerIDX]])):
                #plt.scatter(volcano_log2_fc[layerIDX], volcano_neg_log10_p[layerIDX], s=3, color='black')
                if (np.abs(fc) < 0.2) or (p < -np.log10(0.001)):
                    gene_color = 'lightgrey'
                ax.scatter(fc, p, s=3, color=gene_color)

                current_gene_category = 'Uncategorized'
                if gene_name in gene_categories['Ion channels and ion homeostasis']:
                    current_gene_category = 'Ion channels and ion homeostasis'
                elif gene_name in gene_categories['GABAergic transmission']:
                    current_gene_category = 'GABAergic transmission'
                elif gene_name in gene_categories['Glutamatergic transmission']:
                    current_gene_category = 'Glutamatergic transmission'
                elif gene_name in gene_categories['Synaptic function, connectivity, and plasticity, neuronal development']:
                    current_gene_category = 'Synaptic function, connectivity, and plasticity, neuronal development'
                elif gene_name in gene_categories['Others']:
                    current_gene_category = 'Others'

                if (np.abs(fc) >= 0.2) and (p >= -np.log10(0.001)):
                    ax.annotate(gene_name, (fc, p), fontsize=8, alpha=1.0, color=gene_color)
                    volcano_sig_category_fractions = gene_category_counter(gene_name, volcano_sig_category_fractions, gene_categories)
                if (fc >= 0.2) and (p >= -np.log10(0.001)):
                    volcano_sig_positiveFC_category_fractions = gene_category_counter(gene_name, volcano_sig_positiveFC_category_fractions, gene_categories)
                
                volcano_plot_DF = pd.concat([volcano_plot_DF, pd.DataFrame({'Transcript': [gene_name], 'Transcript Fullname': [long_gene_name], 'Category': [current_gene_category], 'Expression Cross-Tau Fold Change': [fc], 'p-value': [10**(-p)], 'Transcript Layer': [layerNames[layerIDX]]})], ignore_index=True)

            #plt.axhline(y=-np.log10(0.001), color='red', linestyle='--', label='FDR p=0.001')
            #plt.axvline(x=0, color='black', linestyle='--', label='FC=0')
            if not paper_plotting:
                volc_fig.suptitle(f'Gene Expression Fold Change Across $\\tau$ Top & Bottom Quartiles ({datasetName}, {layerNames[layerIDX]})')
            ax.set_xlabel(f'$Log_2$ Expression Fold Change')
            ax.set_ylabel(f'$-Log_{{10}}$ FDR-corrected p-value')
            # Annotate each gene with its name
            # for i, gene_name in enumerate(predictorNamesArray):
            #     plt.annotate(gene_name, (volcano_log2_fc[layerIDX][i], volcano_neg_log10_p[layerIDX][i]), fontsize=10, alpha=1.0)
            plt.legend(
                handles=legend_handles,
                handleheight=0.4,
                handlelength=0.7,
                fontsize=5,
            )
            volc_fig_list.append(volc_fig)
            if save_figs:
                plt.savefig(os.path.join(tauSortedPath, f'{predictorPathSuffix}', f'{datasetName}', f'{datasetName}_Volcano_{layerNames[layerIDX]}.pdf'), bbox_inches='tight')
            plt.close()

            volcano_sig_category_fractions /= np.sum(volcano_sig_category_fractions)
            volcano_sig_positiveFC_category_fractions /= np.sum(volcano_sig_positiveFC_category_fractions)
            # plot pie chart of volcano sig FC category fractions
            fc_sig_frac_fig, ax = plt.subplots(1,1,figsize=(2,2))
            ax.pie(
                volcano_sig_category_fractions,
                colors=category_colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 10, 'color': '#333333'}
            )
            fc_sig_frac_fig_list.append(fc_sig_frac_fig)
            if save_figs:
                plt.savefig(os.path.join(tauSortedPath, f'{predictorPathSuffix}',f'{datasetName}', f'{datasetName}_VolcanoSigCategoryFractions_{layerNames[layerIDX]}.pdf'), bbox_inches='tight')
            plt.close()

            # plot pie chart of volcano sig positive FC category fractions
            fc_sig_frac_pos_fig, ax = plt.subplots(1,1,figsize=(2,2))
            ax.pie(
                volcano_sig_positiveFC_category_fractions,
                colors=category_colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 10, 'color': '#333333'}
            )
            fc_sig_frac_pos_fig_list.append(fc_sig_frac_pos_fig)
            if save_figs:
                plt.savefig(os.path.join(tauSortedPath, f'{predictorPathSuffix}',f'{datasetName}', f'{datasetName}_VolcanoSigPositiveFCCategoryFractions_{layerNames[layerIDX]}.pdf'), bbox_inches='tight')
            plt.close()
        
        # Dump volcano_plot_DF into a csv file
        volcano_plot_DF.to_csv(os.path.join(tauSortedPath, f'{predictorPathSuffix}', f'{datasetName}', f'{datasetName}_Expression_Fold_Change.csv'), index=False)
    
    return volc_fig_list, fc_sig_frac_fig_list, fc_sig_frac_pos_fig_list, volcano_plot_DF