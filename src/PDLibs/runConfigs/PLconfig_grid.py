import datetime
import os
import random
from pathlib import Path

# Get the project root directory dynamically 
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent 
BENCHMARKS_PATH = ROOT_DIR / "data" / "benchmarks"
GRAPH_MODELS_PATH = ROOT_DIR / "data" / "pagn_models"
RLAGENT_MODELS_PATH = ROOT_DIR / "data" / "rlagent_models"
print(ROOT_DIR)


#designName='rlcase1'
designName='muxshifter4'
#designName='muxshifter8'
#designName='muxshifter16'
#designName='muxshifter16b'
#designName='muxshifter32'
#designName='muxshifter64'
#designName='muxshifter128'
#designName='picorv32a'

integration_mode = '2d'
placement_type = '2d'
layer_values = [ 0, 1 ]
number_of_nodes = None
netlist_mode = None
inputDir = BENCHMARKS_PATH / designName
if designName == 'rlcase1':
    number_of_nodes = 18
    netlist_mode = 1
    gin_model_path = GRAPH_MODELS_PATH / "RLCASE1GFEB15_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = GRAPH_MODELS_PATH / "RLCASE1GFEB15_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = RLAGENT_MODELS_PATH / "StackedEnsemble_BestOfFamily_4_AutoML_1_20250305_32129"
elif designName == 'muxshifter4':
    number_of_nodes = 8
    netlist_mode = 1
    gin_model_path = GRAPH_MODELS_PATH / "MUX4GDEC3_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = GRAPH_MODELS_PATH / "MUX4GDEC3_GraphSAGE_encoder_model.pth"
    rl_model_path = RLAGENT_MODELS_PATH / "StackedEnsemble_BestOfFamily_4_AutoML_1_20250305_32129"
elif designName == 'muxshifter8':
    number_of_nodes = 24
    netlist_mode = 0
    gin_model_path = GRAPH_MODELS_PATH / "MUX8GDEC2_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = GRAPH_MODELS_PATH / "MUX8GDEC2_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250203_111743_model_8"
elif designName == 'muxshifter16':
    number_of_nodes = 64
    netlist_mode = 0
    gin_model_path = GRAPH_MODELS_PATH / "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX16GDEC3_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = GRAPH_MODELS_PATH / "MUX16GDEC3_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250203_111743_model_8"
elif designName == 'muxshifter16b':
    number_of_nodes = 64
    netlist_mode = 0
    gin_model_path = GRAPH_MODELS_PATH / "MUX16BGDEC3_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = GRAPH_MODELS_PATH / "MUX16BGDEC3_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250203_111743_model_8"
elif designName == 'muxshifter32':
    number_of_nodes = 160
    netlist_mode = 0
    gin_model_path = GRAPH_MODELS_PATH / "MUX32GDEC3_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = GRAPH_MODELS_PATH / "MUX32GDEC3_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250218_192931_model_8"
elif designName == 'muxshifter64':
    number_of_nodes = 384
    netlist_mode = 0
    gin_model_path = GRAPH_MODELS_PATH / "MUX64GDEC3_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = GRAPH_MODELS_PATH / "MUX64GDEC3_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250218_192931_model_8"

elif designName == 'muxshifter128':
    number_of_nodes = 896
    netlist_mode = 0
    gin_model_path = GRAPH_MODELS_PATH / "MUX128GDEC2_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = GRAPH_MODELS_PATH / "gcnModels/MUX128GDEC2_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_20250218_192931_model_8"

elif designName == 'picorv32a':
    number_of_nodes = 28967
    netlist_mode = 0
    rl_model_path = RLAGENT_MODELS_PATH / "XGBoost_grid_1_AutoML_1_2025006_172801_model_7"
    gin_model_path = GRAPH_MODELS_PATH / "PICORV32AGJAN1_4_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = GRAPH_MODELS_PATH / "PICORV32AGJAN1_4_GraphSAGEEncoder_ed30_encoder_model.pth"

from designgines.PLGridSpec import Grid
grid_definition = Grid()
scl_file_path = inputDir /  f"{designName}.scl"
grid_definition.readSclFile(scl_file_path)
single_cell_height = grid_definition.rows[0].height

from designgines.PLActionsGen import ParametricActionsGen
ag = ParametricActionsGen(mode="paHC")
ag.GenerateActions()

#actions_map.map()
import_num_sites = 'x'
import_num_rows = 'x'

