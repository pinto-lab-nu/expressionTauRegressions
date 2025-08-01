# import pandas as pd
# import numpy as np
# import packages.connect_to_dj as connect_to_dj
# from packages.midprocessing_utils import *
# import itertools

# # Connects to database and creates virtual modules
# VM = connect_to_dj.get_virtual_modules()

# from ccfRegistration.ccf_utils import key_CCF, merge_regions, calculate_pooling_grid, passing_census, functional_timescales





# ##############
# ### Params ###
# binned_pixels = 128 #binned image size, just get this from a table
# raw_image_size = 1024 #OG image size, again, get this from a table -> reduce user-defined params ***more generally do this for all such relevant params!!***
# binning_factor = raw_image_size // binned_pixels
# num_pixels = binned_pixels**2

# resolution = 25 #in um, CCF voxel resolution
# bregma_lambda_distance_avg = 4.1 #in mm, approximation from github.com/petersaj/neuropixels_trajectory_explorer
# bregma_lambda_dist_CCF = bregma_lambda_distance_avg * 1000 / resolution
# CCF25_bregma = [228, 216] #25um bregma in CCF: (216, 18, 228) -> (AP, DV, ML)
# CCF25_lambda = [228, int(CCF25_bregma[1] + bregma_lambda_dist_CCF)] #based on above, CCF lambda: (380, 18, 228)



#### first attempt...
# def load_tau_CCF(line_selection, task):

#     print(f'\nLoading Functional Dataset...')

#     task = 'IntoTheVoid'

#     line_filter = pd.DataFrame((VM['subject'].Subject * VM['session'].Session * VM['behavior'].BehavioralSession & 'experiment_type="widefield"' & 'task="'+task+'"' & 'line="'+line_selection+'"').fetch())

#     passing_sessions = passing_census(0, line_filter.shape[0], line_filter, VM)
### end first attempt...



# def load_tau_CCF(line_selection, task):

#     print(f'\nLoading Functional Dataset...')

#     line_filter = pd.DataFrame((VM['subject'].Subject * VM['session'].Session * VM['behavior'].BehavioralSession & 'experiment_type="widefield"' & 'task="'+task+'"' & 'line="'+line_selection+'"').fetch())
#     passing_sessions = passing_census(0, line_filter.shape[0], line_filter, VM)

#     MOUSE_bregma_list = []
#     Tallen_bregma_list = []
#     MOUSE_lambda_list = []
#     Tallen_lambda_list = []
#     list_of_seen_subjects = []
#     list_of_good_sessions = []
#     for currentPassingSession in passing_sessions:
#         currentSubject,currentDate,currentSession = currentPassingSession[0],currentPassingSession[1],currentPassingSession[2]

#         key = {'subject_fullname': str(currentSubject)
#             ,'session_date': str(currentDate)
#             ,'session_number': currentSession
#         }

#         void_session_PixelAreaLabel = pd.DataFrame((VM['behavior'].BehavioralSession * VM['widefield'].PixelAreaLabel & key).fetch())
#         areaLabels = void_session_PixelAreaLabel['area_label']
#         areaLabelsSet = list(set(areaLabels))
#         vasc_mask = pd.DataFrame((VM['widefield'].VascMask & key).fetch('mask_binned'))
#         voidSync = (VM['widefield'].BehavSync & key)
#         voidSync_VelXY = pd.DataFrame(voidSync.fetch('velocity_by_im_frame'))

#         metaFilterCondition = (voidSync_VelXY.shape[0]>0) and (vasc_mask.shape[0]>0) and (len(areaLabelsSet)>0)

#         if metaFilterCondition:
#             list_of_good_sessions.append(key)
#             if currentSubject not in list_of_seen_subjects:
#                 list_of_seen_subjects.append(currentSubject)
                
#                 q = (VM['widefield'].ReferenceIm & key)
#                 tform_allen2mouse = q.fetch('tform_allen2mouse')[0]
#                 tform_mouse2Tallen = np.linalg.inv(tform_allen2mouse.T)
#                 MOUSEbregma = q.fetch('bregma')[0][0]
#                 MOUSElambda = q.fetch('lambda')[0][0]
#                 Tallen_bregma = (tform_mouse2Tallen @ np.hstack((MOUSEbregma,np.array(1))).reshape(-1,1))[0:-1]
#                 Tallen_lambda = (tform_mouse2Tallen @ np.hstack((MOUSElambda,np.array(1))).reshape(-1,1))[0:-1]

#                 MOUSE_bregma_list.append(MOUSEbregma)
#                 Tallen_bregma_list.append(Tallen_bregma)
#                 MOUSE_lambda_list.append(MOUSElambda)
#                 Tallen_lambda_list.append(Tallen_lambda)

#     Tallen_bregma_array = np.stack(Tallen_bregma_list, axis=0).reshape(-1,2)
#     Tallen_lambda_array = np.stack(Tallen_lambda_list, axis=0).reshape(-1,2)

#     bregma_ML_mean = np.mean(Tallen_bregma_array[:,0])
#     bregma_ML_sd = np.std(Tallen_bregma_array[:,0])
#     bregma_AP_mean = np.mean(Tallen_bregma_array[:,1])
#     bregma_AP_sd = np.std(Tallen_bregma_array[:,1])

#     lambda_ML_mean = np.mean(Tallen_lambda_array[:,0])
#     lambda_ML_sd = np.std(Tallen_lambda_array[:,0])
#     lambda_AP_mean = np.mean(Tallen_lambda_array[:,1])
#     lambda_AP_sd = np.std(Tallen_lambda_array[:,1])

#     num_passing_subjects = Tallen_bregma_array.shape[0]
#     Tallen_bregma_within_dist = []
#     Tallen_lambda_within_dist = []
#     list_of_passing_subjects = []
#     for subject_idx, subject_tmp in enumerate(list_of_seen_subjects):
#         coord_within_dist = np.zeros(8)
        
#         coord_within_dist[0] = Tallen_bregma_array[subject_idx,0] > bregma_ML_mean - bregma_ML_sd
#         coord_within_dist[1] = Tallen_bregma_array[subject_idx,0] < bregma_ML_mean + bregma_ML_sd

#         coord_within_dist[2] = Tallen_bregma_array[subject_idx,1] > bregma_AP_mean - bregma_AP_sd
#         coord_within_dist[3] = Tallen_bregma_array[subject_idx,1] < bregma_AP_mean + bregma_AP_sd

#         coord_within_dist[4] = Tallen_lambda_array[subject_idx,0] > lambda_ML_mean - lambda_ML_sd
#         coord_within_dist[5] = Tallen_lambda_array[subject_idx,0] < lambda_ML_mean + lambda_ML_sd

#         coord_within_dist[6] = Tallen_lambda_array[subject_idx,1] > lambda_AP_mean - lambda_AP_sd
#         coord_within_dist[7] = Tallen_lambda_array[subject_idx,1] < lambda_AP_mean + lambda_AP_sd

#         #print(coord_within_dist)

#         if np.sum(coord_within_dist) == 8:
#             list_of_passing_subjects.append(subject_tmp)

#             Tallen_bregma_within_dist.append(Tallen_bregma_array[subject_idx,:])
#             Tallen_lambda_within_dist.append(Tallen_lambda_array[subject_idx,:])

#     Tallen_bregma_within_dist = np.stack(Tallen_bregma_within_dist)
#     Tallen_lambda_within_dist = np.stack(Tallen_lambda_within_dist)


#     ### define a post-binned coordinate system for the widefield image in the original image space ###
#     binnedCoord_1D = np.arange(binning_factor//2,binned_pixels*binning_factor,binning_factor)
#     grid = np.asarray(list(itertools.product(binnedCoord_1D, binnedCoord_1D)))
#     grid = np.hstack((grid,np.ones((num_pixels,1)))) #adds support for homogeneity in coordinates for 2D affine transform (homogenous 3rd dim)


#     allTauCCF_Coords = np.empty((3,0))
#     for key in list_of_good_sessions:

#         session_idx = np.where(np.array(list_of_passing_subjects) == key['subject_fullname'])[0]
#         if session_idx.shape[0] > 0:

#             session_idx = session_idx[0]

#             tau_array    = (VM['spont_timescales'].WfTau & key).fetch('tau')
#             valid_tau_fit = (VM['spont_timescales'].WfTauInclusion & key).fetch('is_good_tau_pixel')

#             valid_tau_fit_idx = np.where(valid_tau_fit == 1)[0]

#             Tallen_to_CCF_scale = (CCF25_bregma[1] - CCF25_lambda[1]) / (Tallen_bregma_within_dist[session_idx,1] - Tallen_lambda_within_dist[session_idx,1])
#             Tallen_to_CCF_AP_offset = CCF25_bregma[1] - (Tallen_to_CCF_scale * Tallen_bregma_within_dist[session_idx,1])
#             Tallen_to_CCF_ML_offset = CCF25_bregma[0] - (Tallen_to_CCF_scale * Tallen_bregma_within_dist[session_idx,0])
#             tform_Tallen2CCF = np.asarray([[Tallen_to_CCF_scale,0,Tallen_to_CCF_ML_offset],
#                                         [0,Tallen_to_CCF_scale,Tallen_to_CCF_AP_offset],
#                                         [0,0,1]])

#             q = (VM['widefield'].ReferenceIm & key)
#             tform_allen2mouse = q.fetch('tform_allen2mouse')[0]
#             tform_mouse2Tallen = np.linalg.inv(tform_allen2mouse.T)

#             Tallen_Grid = tform_mouse2Tallen @ grid[valid_tau_fit_idx].T
#             CCF_Grid = tform_Tallen2CCF @ Tallen_Grid

#             allTauCCF_Coords = np.hstack((np.vstack((CCF_Grid[0:2,:], tau_array[valid_tau_fit_idx])), allTauCCF_Coords))

#     return allTauCCF_Coords, CCF25_bregma, CCF25_lambda



