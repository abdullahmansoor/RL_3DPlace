import datetime
import os
import random
import sys
from pathlib import Path
import numpy as np

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
        #designName = "muxshifter4",
        designName='muxshifter8',
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
        
        self.design_selector = DesignSelector(designName)
        self.category = self.design_selector.get_category()
        self.scheme = "scheme4"
        
        if designName == 'muxshifter8':
            self.number_of_nodes = 24
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER8JUN12_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER8JUN12_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250726_40416_model_2"
            #"XGBoost_grid_1_AutoML_1_20250708_30730_model_2"
        elif designName == 'muxshifter16':
            self.number_of_nodes = 64
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER16JUN12_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER16JUN12_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250726_40416_model_2"
        elif self.designName == 'muxshifter16b':
            self.number_of_nodes = 64
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER16BJUN12_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER16BJUN12_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250726_40416_model_2"
        elif self.designName == 'muxshifter32':
            self.number_of_nodes = 160
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER32JUN12_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER32JUN12_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250726_35811_model_8"
            #"XGBoost_grid_1_AutoML_1_20250708_33407_model_8"
        elif self.designName == 'muxshifter64':
            self.number_of_nodes = 384
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER64JUN12_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER64JUN12_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250726_35811_model_8"
        elif designName == 'muxshifter128':
            self.number_of_nodes = 896
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER128JUN12_GINEncoder_ed30_encoder_model.pth"
            #"MUX128GDEC2_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "MUXSHIFTER128JUN12_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250726_35811_model_8"
        elif designName == 'picorv32a':
            self.number_of_nodes = 28967
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "picorv32aJUN12_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "picorv32aJUN12_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_lr_search_selection_AutoML_1_20250725_225850_select_grid_model_1"
            #"XGBoost_lr_search_selection_AutoML_1_20250708_34700_select_grid_model_1"

        elif designName == 'jpeg':
            self.number_of_nodes = 524673
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "JPEGJUN12_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "JPEGJUN12_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_lr_search_selection_AutoML_1_20250725_225850_select_grid_model_1"

        elif designName == 'tate':
            self.number_of_nodes = 236673
            self.netlist_mode = 0
            self.gin_model_path = GRAPH_MODELS_PATH / "TATEJUN12_GINEncoder_ed30_encoder_model.pth"
            self.gsage_model_path = GRAPH_MODELS_PATH / "TATEJUN12_GraphSAGE_ed30_encoder_model.pth"
            self.rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_lr_search_selection_AutoML_1_20250725_225850_select_grid_model_1"

        self.grid_definition = Grid()
        self.scl_file_path = self.inputDir / f"{self.designName}.scl"
        self.grid_definition.readSclFile(self.scl_file_path)
        self.single_cell_height = self.grid_definition.rows[0].height

        placement_type = constData.input_placement_type
        sequence_type = constData.sequence_type
        state_method = constData.state_method


        if self.scheme == "scheme5":
            self.ag = ParametricActionsGen(
                #cutValues=np.array([525, 360, 121, 60, 30, 20, 4, 5, 6, 7]),
                cutValues=np.array([241, 120, 60, 30, 15, 7, 1]),
                directions=np.array([0,1]),
                patterns=np.array([0, 1, 2, 3, 4]),
                windowSizeCodes=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]),
                mode="paHC"
            )
        else:
            self.ag = ParametricActionsGen(mode="paHC")
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

from typing import Dict, List

class DesignSelector:
    DESIGN_MAP: Dict[str, List[str]] = {
        'small': ['muxshifter8', 'muxshifter16', 'muxshifter16b'],
        'medium': ['muxshifter32', 'muxshifter64', 'muxshifter128'],
        'large': ['picorv32a', 'tate', 'jpeg']
    }

    def __init__(self, design_name: str):
        self.design_name = design_name.lower()
        self.category = self._resolve_category()

    def _resolve_category(self) -> str:
        for category, design_list in self.DESIGN_MAP.items():
            if self.design_name in design_list:
                return category
        return 'unknown'

    def get_category(self) -> str:
        return self.category

    @classmethod
    def get_designs_for_category(cls, category: str) -> List[str]:
        category = category.lower()
        if category not in cls.DESIGN_MAP:
            raise ValueError(f"Invalid category: {category}. Choose from {list(cls.DESIGN_MAP.keys())}")
        return cls.DESIGN_MAP[category]
