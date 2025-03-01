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
from PLSummarizer import SummarizePlace
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
    
    def GetLayoutFeatures(self, inputPlacementFile):
        self.layout = importUcla(
            name=self.layoutData.designName,
            path=self.layoutData.inputDir,
            inputPlacementFile=inputPlacementFile
        )
        self.twl2d = self.layout.netlist.twl_pahc_2d(self.layout.avg_sites_per_row, 1)[0]
        #generate features using GCN
        #if self.designName == "picorv32a":
        #    self.gcnFeatures = [0 , 1, 2, 3, 4]
        #else:
        embeddings, embedding_columns = DHARLflow.GetGCNEmbeddings(self.layout)
        cellCode = self.GetCellCode()

        #define dataset columns
        self.columns = embedding_columns  + [ 'twl2d' , 'cellCode']
        data = embeddings + [self.twl2d, cellCode]
        df = pd.DataFrame([data], columns=self.columns)
        fn = "embedding_dump.csv"
        df.to_csv(fn)
        fp = os.path.abspath(fn)
        print(fp)
        return df
        
    def GetRLAction(self, df):

        # Load saved model
        model_path = PLconfig_grid.rl_model_path
        model = h2o.load_model(model_path)

        # Load input data for prediction
        input_data = h2o.H2OFrame(df)

        # Make predictions
        predictions = model.predict(input_data)
        predictions = predictions.as_data_frame()
        predictions = int(predictions['predict'][0])
        # Display predictions
        print(predictions)
        return predictions

    def RunSingle(self, sInputPlacementFile):
        start_time = time.time()  # Start time tracking

        df_features = self.GetLayoutFeatures(sInputPlacementFile)
        
        end_time = time.time()  # End time tracking
        elapsed_time = end_time - start_time

        print(f"Embedding Execution Time: {elapsed_time:.2f} seconds")

        layoutData = self.layoutData
        ag = self.ag
        inputDir = os.path.abspath(PLconfig_grid.inputDir)

        start_time = time.time()  # Start time tracking

        #action = self.GetRLAction(df_features)
        action = 209

        end_time = time.time()  # End time tracking
        elapsed_time = end_time - start_time

        print(f"RL Policy Execution Time: {elapsed_time:.2f} seconds")

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
            twl = nl1.netlist.twl(nl1.avg_sites_per_row, foldingParams.bin_size_x, 2)[0]
            #twl = nl1.netlist.twl(nl1.avg_sites_per_row, foldingParams.divider_x, 2)[0]
        except Exception as e:
            print(f"PA HC failed due to exception {e}")
            twl = 99999999

        twlDelta = twl - self.twl2d

        print(f"twlDelt={twlDelta}, new_cost={twl}, twl2d={self.twl2d}")
        
def Run(args):
    ag = PLconfig_grid.ag
    designName = PLconfig_grid.designName
    inputDir = PLconfig_grid.inputDir
    confData = ConfigData()
    layoutData = LayoutData(confData)

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
                        help="folder of input UCLA design files",
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
