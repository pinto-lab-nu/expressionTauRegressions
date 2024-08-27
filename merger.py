import platform
import os
import sys
import re
from packages.regressionUtils import *


lineSelection = 'Cux2-Ai96'
layerNames  = ['L2_3 IT',   'L4_5 IT',  'L5 IT',    'L6 IT',    'L5 ET']



my_os = platform.system()
if my_os == 'Linux':
    lineSelection  = ['Rpb4-Ai96','Cux2-Ai96','C57BL6/J','PV-Ai96'][int(sys.argv[1])]

projectFolder = "lineFilter" + lineSelection

if my_os == 'Linux':
    savePath = os.path.join(r'/mnt/fsmresfiles/Tau_Processing/H3/')
if my_os == 'Windows':
    savePath = os.path.join(r'R:\Basic_Sciences\Phys\PintoLab\Tau_Processing\H3')




for prePath in ['Spatial',lineSelection]:

    for poolIDX,tauPoolSize in enumerate([1,2,4,8,16]):

        for subPath in ['H3Predictors','genePredictors','']:
            if prePath == lineSelection:
                currentPath = os.path.join(savePath,prePath,f'pooling{tauPoolSize}',subPath)
            if (prePath == 'Spatial') and (poolIDX == 0):
                currentPath = os.path.join(savePath,prePath,subPath)
            
            # Regex to capture portions before and after the variable part
            pattern = re.compile(r'^(.*?)(_L\d+.* [EI]T)(.*)$')

            unique_file_names = set()

            for file in os.listdir(currentPath):
                if (file[-4:] == '.pdf'):
                    match = pattern.match(file)
                    if match:
                        combined = match.group(1) + '*****' + match.group(3)
                        unique_file_names.add(combined)

            unique_file_names = sorted(unique_file_names)

            for rename in unique_file_names:
                split0,split1 = rename.split('*****')
                if os.path.exists(os.path.join(currentPath,split0+'_'+layerNames[0]+split1)):
                    PDFmerger(currentPath,split0+'_',layerNames,split1,split0+split1)
            





# for tauPoolSize in [1,2,4,8,16]:
#     tauSortedPath = os.path.join(savePath,lineSelection,f'pooling{tauPoolSize}')

#     for subPath in ['H3Predictors','genePredictors','']:

#         currentPath = os.path.join(tauSortedPath,subPath)

#         # Regex to capture portions before and after the variable part
#         #pattern = re.compile(r'^(.*?)(_L\d+.* [EI]T)?(\.pdf)$')
#         pattern = re.compile(r'^(.*?)(_L\d+.* [EI]T)(.*)$')

#         unique_file_names = set()

#         for file in os.listdir(currentPath):
#             if (file[-4:] == '.pdf'):
#                 match = pattern.match(file)
#                 if match:
#                     combined = match.group(1) + '*****' + match.group(3)
#                     unique_file_names.add(combined)

#         unique_file_names = sorted(unique_file_names)

#         for rename in unique_file_names:
#             split0,split1 = rename.split('*****')
#             if os.path.exists(os.path.join(currentPath,split0+'_'+layerNames[0]+split1)):
#                 PDFmerger(currentPath,split0+'_',layerNames,split1,split0+split1)
