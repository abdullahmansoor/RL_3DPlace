from pathlib import Path
import sys
import os
import argparse
import operator
import glob
import copy
import datetime
import random
import time
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import logging
import h2o
import time
import psutil


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s -  [%(pathname)s:%(lineno)d] %(message)s'
)

# Add PDLibs to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "PDLibs"))
#from PLSummarizer import SummarizePlace
import runConfigs.PLconfig_grid as PLconfig_grid
from designgines.PLimportUcla import importUcla
from dataModel.PLConstData import ConfigData
from dataModel.PLLayoutData import LayoutData
from design_engines.PLHierLayoutController import RunHier3DConversion
from design_engines.PLPAHierParametricFolding import PAHierParametricFolding
from graph_library.graph_models_helpers import InitializeEncoderModel, GetEmbeddings

class DHARLflow(object):
    def __init__(self, ag, layoutData , designName, inputDir):
        self.ag = ag
        self.layoutData = layoutData
        self.designName = designName
        self.inputDir = inputDir

        self.layout = None
        self.twl2d = None
        self.features = None
        self.gcnFeaturesColumns = None
        self.columns = None

    @staticmethod
    def GetGCNEmbeddings(layout):
        input_dim = layout.netlist.numNodes
        hidden_dim = input_dim*3
        n_layers = 1
        embedding_dim = 30

        data = layout.netlist.get_torch_data()

        gin_model_path = PLconfig_grid.gin_model_path

        encoder = InitializeEncoderModel("GINEncoder", input_dim, hidden_dim, n_layers, embedding_dim)

        gin_embeddings = GetEmbeddings(encoder, gin_model_path, data)
        gin_embeddings = gin_embeddings.tolist()[0]
        gin_embeddings_columns = [f"gcn{x}_y" for x in range(len(gin_embeddings))]
        #print(f"gin_embeddings={gin_embeddings}")
        gsage_model_path = PLconfig_grid.gsage_model_path

        encoder = InitializeEncoderModel("GraphSAGE", input_dim, hidden_dim, n_layers, embedding_dim)

        gsage_embeddings = GetEmbeddings(encoder, gsage_model_path, data)
        gsage_embeddings = gsage_embeddings.tolist()[0]
        #print(f"gsage_embeddings={gsage_embeddings}")
        gsage_embeddings_columns = [f"gcn{x}_x" for x in range(len(gsage_embeddings))]

        embeddings = gin_embeddings + gsage_embeddings
        columns = gin_embeddings_columns + gsage_embeddings_columns
        return embeddings, columns

    def GetCellCode(self):
        if self.designName == "picorv32a":
            return 0
        elif self.designName == "muxshifter8":
            return 0
        elif self.designName == "muxshifter16":
            return 1
        elif self.designName == "muxshifter16b":
            return 2
        elif self.designName == "muxshifter32":
            return 0
        elif self.designName == "muxshifter64":
            return 1
        elif self.designName == "muxshifter128":
            return 2
        elif self.designName == "muxshifter4":
            return 0
        elif self.designName == "rlcase1":
            return 1
    
    def GetLayoutFeatures(self, inputPlacementFile):
        self.layout = importUcla(
            name=self.layoutData.designName,
            path=self.layoutData.inputDir,
            inputPlacementFile=inputPlacementFile
        )
        #self.layout.netlist.change_location_type()
        self.twl2d = self.layout.netlist.twl_pahc_2d(self.layout.avg_sites_per_row, 1)[0]
        embeddings, embedding_columns = DHARLflow.GetGCNEmbeddings(self.layout)
        cellCode = self.GetCellCode()

        #define dataset columns
        self.columns = embedding_columns  + [ 'twl2d' , 'cellCode']
        data = embeddings + [self.twl2d, cellCode]
        df = pd.DataFrame([data], columns=self.columns)
        #fn = "embedding_dump.csv"
        #df.to_csv(fn)
        #fp = os.path.abspath(fn)
        #print(fp)
        return df
        
    def GetRLAction(self, sInputPlacementFile):
        df_features = self.GetLayoutFeatures(sInputPlacementFile)
        #print("df_features=",df_features)

        # Load saved model
        model_path = f"{PLconfig_grid.rl_model_path}"
        model = h2o.load_model(model_path)
        logger.info("Model loaded")

        # Load input data for prediction
        input_data = h2o.H2OFrame(df_features)
        #print("input_data=",input_data)

        # Make predictions
        predictions = model.predict(input_data)
        predictions = predictions.as_data_frame()
        predictions = int(predictions['predict'][0])
        # Display predictions
        logger.info(predictions)
        return predictions

    def RunSingle(self, sInputPlacementFile):
        start_time = time.time()  # Start time tracking

        layoutData = self.layoutData
        ag = self.ag
        inputDir = os.path.abspath(PLconfig_grid.inputDir)

        try:
            action = self.GetRLAction(sInputPlacementFile)
            logger.info(f"action Code = {action}")
        except Exception as e:
            #using experimental values since some large model files can't be uploaded on github
            if self.designName == "picorv32a": action = 141
            #elif self.designName == "muxshifter128": action = 209
            else: action=209
            logger.info(f"Faced Exception! action Code = {action}")
        logger.info(f"action Code = {action}")
        end_time = time.time()  # End time tracking
        elapsed_time = end_time - start_time

        logger.info(f"RL Policy Execution Time: {elapsed_time:.2f} seconds")

        numberOfCuts, direction, pattern, windowSizeCode = ag.DecodeAction(action)
        logger.info(f"#########\n"
                    f"numberOfCuts={numberOfCuts}\n"
                    f"direction={direction}\n"
                    f"pattern={pattern}\n"
                    f"windowSizeCode={windowSizeCode} \n\n\n\n"
                    )
        foldingParams = PAHierParametricFolding(
            numberOfCuts,
            direction,
            pattern,
            windowSizeCode
        )

        try:
            nl1, summarizer, dm = RunHier3DConversion(
                layoutData,
                foldingParams,
                sInputPlacementFile
            )

            nl1.netlist.change_location_type()
            cf = nl1.netlist.calculate_cost_function(nl1.avg_sites_per_row, foldingParams.bin_size_x, 2)
            self.layout.netlist.change_location_type()
            last_cf = self.layout.netlist.calculate_cost_function(
                self.layout.avg_sites_per_row, 1, 1
            )
            logger.info(f"Grid Coordinates new cf={cf}, last_cf={last_cf}, delta cf = {cf-last_cf}")
            
        except Exception as e:
            logger.info(f"PA HC failed due to exception {e}")
            twl = 99999999


        
def Run(args):
    ag = PLconfig_grid.ag
    if args.sDesignName:
        designName = args.sDesignName
    else:
        designName = PLconfig_grid.designName
    inputDir = PLconfig_grid.inputDir
    confData = ConfigData()
    layoutData = LayoutData(confData, designName=designName)

    runObj = DHARLflow(ag, layoutData , designName, inputDir)
    if not args.sInputPlacement:
        sInputPlacementFile = os.path.join(layoutData.inputDir, f"{layoutData.designName}.pl")
    else:
        sInputPlacementFile = os.path.abspath(args.sInputPlacement)

    runObj.RunSingle(sInputPlacementFile)
        
def main():
    parser = argparse.ArgumentParser()    
    

    parser.add_argument("-inputPlacement", action="store", 
                        dest="sInputPlacement",
                        help="Placement file for evaluation",
                        required=False, type=str)
    parser.add_argument("-designName", action="store", 
                        dest="sDesignName",
                        help="Name of design for DHCARL flow",
                        required=False, type=str)

    args = parser.parse_args()

    # Initialize H2O
    h2o.init()

    process = psutil.Process(os.getpid())

    # Measure initial memory usage
    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    start_time = time.time()  # Start time tracking
        
    Run(args)
    
    # Measure final memory usage
    mem_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    end_time = time.time()  # End time tracking

    # Compute results
    elapsed_time = end_time - start_time
    mem_used = mem_after - mem_before

    print(f"Execution Time: {elapsed_time:.2f} seconds")
    print(f"Memory Used: {mem_used:.2f} MB")
    
if __name__ == "__main__":
    main()
