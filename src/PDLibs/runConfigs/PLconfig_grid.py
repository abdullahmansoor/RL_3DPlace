import datetime
import os
import random

#designName = 'superblue1'
designName='RLcase1'
#designName='muxshifter2'
#designName='muxshifter3'
#designName='muxshifter4'
#designName='muxshifter8'
#designName='muxshifter16'
#designName='muxshifter16b'
#designName='muxshifter32'
#designName='muxshifter64'
#designName='muxshifter128'
#designName = '3node'
#designName='picorv32a'

integration_mode = '2d'
placement_type = '2d'
integration_mode = '2d'
number_of_nodes = None
netlist_mode = None
if designName == 'RLcase1':
    number_of_nodes = 18
    netlist_mode = 1
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_RLcase1_bin_scheme1_df3_ct3000.csv"
    modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/RLcase1_binFolding/df_dataset_RLcase1_bin_scheme1_df3_ct3000_20230930_233959_7311/my_saved_model"
    modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/RLcase1_binFolding/df_dataset_RLcase1_bin_scheme1_df3_ct4984_20231001_074757_5607/my_saved_model"
    modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/RLcase1_binFolding/df_dataset_RLcase1_bin_scheme6_df3_ct3000_20231001_074813_4902/my_saved_model"
    modelPath = ""
    gin_model_path = "/scratch/mansoor4/RLAgentDS/gModel/RLCASE1GFEB15_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = "/scratch/mansoor4/RLAgentDS/gModel/RLCASE1GFEB15_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = ""
    if integration_mode == '3d':
        inputDir = '/home/mansoor4/test-cases/RLcase13d/'
        #inputDir = "/scratch/mansoor4/test-cases_archive/RLcase13d_archive/archive2"
    else:
        inputDir = '/home/mansoor4/test-cases/RLcase1/'
elif designName == '3node':
    number_of_nodes = 3
    netlist_mode = 1
    inputDir = '/home/mansoor4/test-cases/3node/'
elif designName == 'muxshifter2':
    number_of_nodes = 2
    netlist_mode = 1
    inputDir = '/home/mansoor4/test-cases/muxShifter2_UCLA/'
elif designName == 'muxshifter3':
    number_of_nodes = 3
    netlist_mode = 1
    inputDir = '/home/mansoor4/test-cases/muxShifter3_UCLA/'
elif designName == 'muxshifter4':
    number_of_nodes = 8
    netlist_mode = 1
    modelPath = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX4GDEC3_encoder_model.pth"
    gin_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX4GDEC3_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX4GDEC3_GraphSAGE_encoder_model.pth"
    if integration_mode == '3d':
        inputDir = '/home/mansoor4/test-cases/muxShifter4_UCLA3d/'
    else:
        inputDir = '/home/mansoor4/test-cases/muxShifter4_UCLA/'
elif designName == 'muxshifter8':
    number_of_nodes = 24
    netlist_mode = 0
    dataset_file = "/home/mansoor4/docs/cds_m8_mcc_hvz_bF_comV2.csv"
    modelPath = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX8GDEC2_encoder_model.pth"
    gin_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX8GDEC2_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX8GDEC2_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = "/scratch/mansoor4/RLAgentDS/h2o3/gin_sag_combined_mux8_16_16b_3722/XGBoost_grid_1_AutoML_1_20250203_111743_model_8"
    if integration_mode == '3d':
        inputDir = '/home/mansoor4/test-cases/muxShifter8_UCLA3d/'
    else:
        inputDir = '/home/mansoor4/test-cases/muxShifter8_UCLA/'
elif designName == 'muxshifter16':
    number_of_nodes = 64
    netlist_mode = 0
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter16_scheme1.csv"
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter16_bin_scheme6_df2_ct516.csv"
    #dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter16_bin_scheme6_df3_ct516.csv"
    modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter16_paramFolding/df_dataset_muxshifter16_scheme1_20230828_225556_7644/my_saved_model"
    modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter16_binFolding/df_dataset_muxshifter16_bin_scheme6_df2_ct516_20231001_074735_2817/my_saved_model"
    #modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter16_binFolding/df_dataset_muxshifter16_bin_scheme6_df3_ct516_20231001_074708_7431/my_saved_model"
    modelPath = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX16GDEC3_encoder_model.pth"
    gin_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX16GDEC3_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX16GDEC3_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = "/scratch/mansoor4/RLAgentDS/h2o3/gin_sag_combined_mux8_16_16b_3722/XGBoost_grid_1_AutoML_1_20250203_111743_model_8"
    if integration_mode == '3d':
        inputDir = '/home/mansoor4/test-cases/muxShifter16_UCLA3d/'
        inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter16_UCLA3d_archive/scheme1"
    else:
        inputDir = '/home/mansoor4/test-cases/muxShifter16_UCLA/'
elif designName == 'muxshifter16b':
    number_of_nodes = 64
    netlist_mode = 0
    modelPath = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX16BGDEC3_encoder_model.pth"
    gin_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX16BGDEC3_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX16BGDEC3_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = "/scratch/mansoor4/RLAgentDS/h2o3/gin_sag_combined_mux8_16_16b_3722/XGBoost_grid_1_AutoML_1_20250203_111743_model_8"
    if integration_mode == '3d':
        inputDir = '/home/mansoor4/test-cases/muxShifter16b_UCLA3d/'
        inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter16b_UCLA3d_archive/scheme1"
    else:
        inputDir = '/home/mansoor4/test-cases/muxShifter16b_UCLA/'
elif designName == 'muxshifter32':
    number_of_nodes = 160
    netlist_mode = 0
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme3_df4.csv"
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme1.csv"
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme1_df4.csv"
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme1_df4_ct3000.csv"
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme1_df4_ct13285.csv"
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme6_df4_ct3000.csv"
    modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter32_binFolding/df_dataset_muxshifter32_bin_scheme1_df4_ct3000_20230914_072920_5687/my_saved_model"
    modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter32_binFolding/df_dataset_muxshifter32_bin_scheme1_df4_ct13285_20230914_231120_8857/my_saved_model"
    modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter32_binFolding/df_dataset_muxshifter32_bin_scheme6_df4_ct3000_20231008_220555_9232/my_saved_model"
    modelPath = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX32GDEC3_encoder_model.pth"
    gin_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX32GDEC3_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX32GDEC3_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = "/scratch/mansoor4/RLAgentDS/h2o3/gin_sag_combined_mux32_64_128_8959/XGBoost_grid_1_AutoML_1_20250218_192931_model_8"

    if integration_mode == '3d':
        inputDir = '/home/mansoor4/test-cases/muxShifter32_UCLA3d/'
        inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter32_UCLA3d_archive/archive1"
    else:
        inputDir = '/home/mansoor4/test-cases/muxShifter32_UCLA/'
elif designName == 'muxshifter64':
    number_of_nodes = 384
    netlist_mode = 0
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter64_bin_scheme6_df4_ct3000.csv"
    modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter64_binFolding/df_dataset_muxshifter64_bin_scheme6_df4_ct3000_20231008_220253_3174/my_saved_model"
    modelPath = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX64GDEC3_encoder_model.pth"
    gin_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX64GDEC3_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX64GDEC3_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = "/scratch/mansoor4/RLAgentDS/h2o3/gin_sag_combined_mux32_64_128_8959/XGBoost_grid_1_AutoML_1_20250218_192931_model_8"
    if integration_mode == '3d':
        #inputDir = '/home/mansoor4/test-cases/muxShifter64_UCLA3d/'
        inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter64_UCLA3d_archive/archive1/"
    else:
        inputDir = '/home/mansoor4/test-cases/muxShifter64_UCLA/'
        #inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter64_UCLA_archive/archive1/"


elif designName == 'muxshifter128':
    number_of_nodes = 896
    netlist_mode = 0
    dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter128_bin_scheme1_df4_ct1779.csv"
    modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter128_binFolding/df_dataset_muxshifter128_bin_scheme1_df4_ct1779_20231008_215738_83/my_saved_model"
    modelPath = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX128GDEC2_encoder_model.pth"
    gin_model_path = "/scratch/mansoor4/RLAgentDS/gModel/MUX128GDEC2_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = "/home/mansoor4/CircuitAttributePrediction/supervisedLearning/gcnModels/MUX128GDEC2_GraphSAGE_ed30_encoder_model.pth"
    rl_model_path = "/scratch/mansoor4/RLAgentDS/h2o3/gin_sag_combined_mux32_64_128_8959/XGBoost_grid_1_AutoML_1_20250218_192931_model_8"
    if integration_mode == '3d':
        #inputDir = '/home/mansoor4/test-cases/muxShifter128_UCLA3d/'
        inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter128_UCLA3d_archive/archive1"
    else:
        inputDir = '/home/mansoor4/test-cases/muxShifter128_UCLA/'

elif designName == 'picorv32a':
    number_of_nodes = 28967
    netlist_mode = 0
    dataset_file = ""
    modelPath = ""
    rl_model_path = ""
    gin_model_path = "/scratch/mansoor4/RLAgentDS/gModel/untrained/PICORV32AGJAN1_4_GINEncoder_ed30_encoder_model.pth"
    gsage_model_path = ""
    if integration_mode == '3d':
        inputDir = ""
    else:
        inputDir = '/scratch/mansoor4/test-cases_archive/picorv32a'

from designgines.PLGridSpec import Grid
grid_definition = Grid()
scl_file_path = inputDir + "/" + designName + ".scl"
grid_definition.readSclFile(scl_file_path)
single_cell_height = grid_definition.rows[0].height

from designgines.PLActionsGen import ParametricActionsGen
ag = ParametricActionsGen(mode="paHC")
ag.GenerateActions()

#actions_map.map()
import_num_sites = 'x'
import_num_rows = 'x'

