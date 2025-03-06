import datetime
import os
import random
import sys
from pathlib import Path

from designgines.PLGridSpec import Grid
from designgines.PLActionsGen import ParametricActionsGen
import runConfigs.PLconfig_grid as PLconfig_grid

# Get the project root directory dynamically 
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent 
BENCHMARKS_PATH = ROOT_DIR / "data" / "benchmarks"
GRAPH_MODELS_PATH = ROOT_DIR / "data" / "pagn_models"
RLAGENT_MODELS_PATH = ROOT_DIR / "data" / "rlagent_models"


class LayoutData(object):
    def __init__(self,
        constData,
        #designName = 'superblue1'
        #designName = '3node',
        #designName='rlcase1',
        #designName='muxshifter2'
        #designName='muxshifter3'
        designName = "muxshifter4",
        #designName='muxshifter8',
        #designName='muxshifter16',
        #designName='muxshifter16b',
        #designName='muxshifter32',
        #designName='muxshifter64',
        #designName='muxshifter128',
        #designName = 'picorv32a',

        #Folding actions
        run_path =  "/scratch/mansoor4/runsdir/",

        log_name = 'run.log',
        summary_db = 'PLMVDLAMPlace.db',
    ):

        self.constData = constData
        self.designName = designName
        self.number_of_nodes = None
        self.netlist_mode = None
        self.dataset_file = None
        self.modelPath = None
        self.inputDir =  BENCHMARKS_PATH / designName

        if self.designName == 'rlcase1':
            self.number_of_nodes = 18
            self.netlist_mode = 1
            self.gin_model_path = GRAPH_MODELS_PATH / "RLCASE1GFEB15_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "RLCASE1GFEB15_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "StackedEnsemble_BestOfFamily_4_AutoML_1_20250305_32129"

        elif self.designName == 'muxshifter4':
            self.number_of_nodes = 8
            self.netlist_mode = 1
            self.gin_model_path = GRAPH_MODELS_PATH / "MUX4GDEC3_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUX4GDEC3_GraphSAGE_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "StackedEnsemble_BestOfFamily_4_AutoML_1_20250305_32129"
        elif designName == 'muxshifter8':
            self.number_of_nodes = 24
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "MUX8GDEC2_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUX8GDEC2_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250306_02444_model_2"
        elif designName == 'muxshifter16':
            self.number_of_nodes = 64
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "MUX16GDEC3_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUX16GDEC3_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250306_02444_model_2"
        elif self.designName == 'muxshifter16b':
            self.number_of_nodes = 64
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "MUX16BGDEC3_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUX16BGDEC3_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250306_02444_model_2"
        elif self.designName == 'muxshifter32':
            self.number_of_nodes = 160
            self.netlist_mode = 0
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250218_192931_model_8"
            self.gin_model_path = GRAPH_MODELS_PATH / "MUX32GDEC3_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUX32GDEC3_GraphSAGE_ed30_encoder_model.pth"
        elif self.designName == 'muxshifter64':
            self.number_of_nodes = 384
            self.netlist_mode = 0
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250218_192931_model_8"
            self.gin_model_path = GRAPH_MODELS_PATH / "MUX64GDEC3_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUX64GDEC3_GraphSAGE_ed30_encoder_model.pth"
        elif designName == 'muxshifter128':
            self.number_of_nodes = 896
            self.netlist_mode = 0
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250218_192931_model_8"
            self.gin_model_path = GRAPH_MODELS_PATH / "MUX128GDEC2_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUX128GDEC2_GraphSAGE_ed30_encoder_model.pth"
        elif designName == 'picorv32a':
            self.number_of_nodes = 28967
            self.netlist_mode = 0
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_2025006_172801_model_7"
            self.gin_model_path = GRAPH_MODELS_PATH / "PICORV32AGJAN1_4_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "PICORV32AGJAN1_4_GraphSAGEEncoder_ed30_encoder_model.pth"
        self.grid_definition = Grid()
        self.scl_file_path = self.inputDir / f"{self.designName}.scl"
        self.grid_definition.readSclFile(self.scl_file_path)
        self.single_cell_height = self.grid_definition.rows[0].height

        placement_type = constData.input_placement_type
        sequence_type = constData.sequence_type
        state_method = constData.state_method


        self.ag = ParametricActionsGen()
        self.ag.GenerateActions()


        start = datetime.datetime.now().replace(microsecond=0)
        randomizer=random.randrange(10000)
        run_dir_name = "{}_{}_{}_df{}_tm{}_{}_{}".format(constData.algorithm, designName, constData.integration_mode, constData.bin_size_x, constData.test_mode, start.strftime("%Y%m%d_%H%M%S"),randomizer)


        self.update_for_backward_compatability()

    def update_for_backward_compatability(self):
        PLconfig_grid.designName = self.designName
        PLconfig_grid.number_of_nodes = self.number_of_nodes
        PLconfig_grid.netlist_mode = self.netlist_mode
        PLconfig_grid.inputDir = self.inputDir
        PLconfig_grid.gin_model_path = self.gin_model_path
        PLconfig_grid.gsage_model_path = self.gsage_model_path
        PLconfig_grid.rl_model_path = self.rl_model_path
        PLconfig_grid.grid_definition = self.grid_definition
        PLconfig_grid.single_cell_height = self.single_cell_height
        PLconfig_grid.ag = self.ag
