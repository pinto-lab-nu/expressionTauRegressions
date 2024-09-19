import platform
import os
import sys
import re
from packages.regressionUtils import *
from packages.dataloading import *


lineSelection = 'Cux2-Ai96'
pilotLayerNames   = ['L2_3 IT',   'L4_5 IT',   'L5 IT',   'L6 IT',  'L5 ET']
merfishLayerNames = ['L2_3 IT_ET','L4_5 IT_ET','L6 IT_ET']
_, _, savePath = pathSetter(lineSelection)



for prePath in ['Spatial',lineSelection]:

    for poolIDX,tauPoolSize in enumerate([2]): #[2,4]
        tauPoolSize *= 0.025

        for predictionPath in ['H3Predictors','GenePredictors']: #include " '' " in array here, or below?

            for datasetPath,layerNames in zip(['Pilot','Merfish','Merfish-Imputed'],[pilotLayerNames,merfishLayerNames,merfishLayerNames]):

                if prePath == lineSelection:
                    currentPath = os.path.join(savePath,prePath,f'pooling{tauPoolSize}mm',predictionPath,datasetPath)
                if (prePath == 'Spatial') and (poolIDX == 0):
                    currentPath = os.path.join(savePath,prePath,predictionPath,datasetPath)
                
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
                
