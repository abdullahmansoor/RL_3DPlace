import datetime
import os
import random
import sys

from designgines.PLGridSpec import Grid
from designgines.PLActionsGen import ParametricActionsGen


class LayoutData(object):
    def __init__(self,
        constData,
        #designName = 'superblue1'
        #designName = '3node',
        designName='RLcase1',
        #designName='muxshifter2'
        #designName='muxshifter3'
        #designName = "muxshifter4",
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
        self.inputDir = None
        if self.designName == 'RLcase1':
            self.number_of_nodes = 18
            self.netlist_mode = 1
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_RLcase1_bin_scheme1_df3_ct3000.csv"
            self.modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/RLcase1_binFolding/df_dataset_RLcase1_bin_scheme1_df3_ct3000_20230930_233959_7311/my_saved_model"
            self.modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/RLcase1_binFolding/df_dataset_RLcase1_bin_scheme1_df3_ct4984_20231001_074757_5607/my_saved_model"
            self.modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/RLcase1_binFolding/df_dataset_RLcase1_bin_scheme6_df3_ct3000_20231001_074813_4902/my_saved_model"
            if constData.integration_mode == '3d':
                self.inputDir = '/home/mansoor4/test-cases/RLcase13d/'
                #inputDir = "/scratch/mansoor4/test-cases_archive/RLcase13d_archive/archive2"
            else:
                self.inputDir = '/home/mansoor4/test-cases/RLcase1/'
        elif self.designName == '3node':
            self.number_of_nodes = 3
            self.netlist_mode = 1
            self.inputDir = '/home/mansoor4/test-cases/3node/'
        elif self.designName == 'muxshifter2':
            self.number_of_nodes = 2
            self.netlist_mode = 1
            self.inputDir = '/home/mansoor4/test-cases/muxShifter2_UCLA/'
        elif self.designName == 'muxshifter3':
            self.number_of_nodes = 3
            self.netlist_mode = 1
            self.inputDir = '/home/mansoor4/test-cases/muxShifter3_UCLA/'
        elif self.designName == 'muxshifter4':
            self.number_of_nodes = 8
            self.netlist_mode = 1
            if constData.integration_mode == '3d':
                self.inputDir = '/home/mansoor4/test-cases/muxShifter4_UCLA3d/'
            else:
                self.inputDir = '/home/mansoor4/test-cases/muxShifter4_UCLA/'
        elif designName == 'muxshifter8':
            self.number_of_nodes = 24
            self.netlist_mode = 0
            self.dataset_file = "/home/mansoor4/docs/cds_m8_mcc_hvz_bF_comV2.csv"
            if constData.integration_mode == '3d':
                self.inputDir = '/home/mansoor4/test-cases/muxShifter8_UCLA3d/'
            else:
                self.inputDir = '/home/mansoor4/test-cases/muxShifter8_UCLA/'
        elif designName == 'muxshifter16':
            self.number_of_nodes = 64
            self.netlist_mode = 0
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter16_scheme1.csv"
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter16_bin_scheme6_df2_ct516.csv"
            #dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter16_bin_scheme6_df3_ct516.csv"
            self.modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter16_paramFolding/df_dataset_muxshifter16_scheme1_20230828_225556_7644/my_saved_model"
            self.modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter16_binFolding/df_dataset_muxshifter16_bin_scheme6_df2_ct516_20231001_074735_2817/my_saved_model"
            #modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter16_binFolding/df_dataset_muxshifter16_bin_scheme6_df3_ct516_20231001_074708_7431/my_saved_model"
            if constData.integration_mode == '3d':
                self.inputDir = '/home/mansoor4/test-cases/muxShifter16_UCLA3d/'
                self.inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter16_UCLA3d_archive/scheme1"
            else:
                self.inputDir = '/home/mansoor4/test-cases/muxShifter16_UCLA/'
        elif self.designName == 'muxshifter16b':
            self.number_of_nodes = 64
            self.netlist_mode = 0
            if constData.integration_mode == '3d':
                self.inputDir = '/home/mansoor4/test-cases/muxShifter16b_UCLA3d/'
                self.inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter16b_UCLA3d_archive/scheme1"
            else:
                self.inputDir = '/home/mansoor4/test-cases/muxShifter16b_UCLA/'
        elif self.designName == 'muxshifter32':
            self.number_of_nodes = 160
            self.netlist_mode = 0
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme3_df4.csv"
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme1.csv"
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme1_df4.csv"
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme1_df4_ct3000.csv"
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme1_df4_ct13285.csv"
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter32_bin_scheme6_df4_ct3000.csv"
            self.modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter32_binFolding/df_dataset_muxshifter32_bin_scheme1_df4_ct3000_20230914_072920_5687/my_saved_model"
            self.modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter32_binFolding/df_dataset_muxshifter32_bin_scheme1_df4_ct13285_20230914_231120_8857/my_saved_model"
            self.modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter32_binFolding/df_dataset_muxshifter32_bin_scheme6_df4_ct3000_20231008_220555_9232/my_saved_model"

            if constData.integration_mode == '3d':
                self.inputDir = '/home/mansoor4/test-cases/muxShifter32_UCLA3d/'
                self.inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter32_UCLA3d_archive/archive1"
            else:
                self.inputDir = '/home/mansoor4/test-cases/muxShifter32_UCLA/'
        elif self.designName == 'muxshifter64':
            self.number_of_nodes = 384
            self.netlist_mode = 0
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter64_bin_scheme6_df4_ct3000.csv"
            self.modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter64_binFolding/df_dataset_muxshifter64_bin_scheme6_df4_ct3000_20231008_220253_3174/my_saved_model"
            if constData.integration_mode == '3d':
                #inputDir = '/home/mansoor4/test-cases/muxShifter64_UCLA3d/'
                self.inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter64_UCLA3d_archive/archive1/"
            else:
                self.inputDir = '/home/mansoor4/test-cases/muxShifter64_UCLA/'
                #self.inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter64_UCLA_archive/archive1/"

        elif designName == 'muxshifter128':
            self.number_of_nodes = 896
            self.netlist_mode = 0
            self.dataset_file = "/home/mansoor4/PDLibs/utils/dataset_muxshifter128_bin_scheme1_df4_ct1779.csv"
            self.modelPath = "/home/mansoor4/PDLibs/parametersTuningLib/muxshifter128_binFolding/df_dataset_muxshifter128_bin_scheme1_df4_ct1779_20231008_215738_83/my_saved_model"
            if constData.integration_mode == '3d':
                #inputDir = '/home/mansoor4/test-cases/muxShifter128_UCLA3d/'
                self.inputDir = "/scratch/mansoor4/test-cases_archive/muxShifter128_UCLA3d_archive/archive1"
            else:
                self.inputDir = '/home/mansoor4/test-cases/muxShifter128_UCLA/'
        elif designName == 'picorv32a':
            self.number_of_nodes = 28967
            self.netlist_mode = 0
            self.dataset_file = ""
            self.modelPath = ""
            if constData.integration_mode == '3d':
                self.inputDir = "/scratch/mansoor4/test-cases_archive/picorv32a"
            else:
                self.inputDir = '/scratch/mansoor4/test-cases_archive/picorv32a'

        self.grid_definition = Grid()
        self.scl_file_path = self.inputDir + "/" + self.designName + ".scl"
        self.grid_definition.readSclFile(self.scl_file_path)


        placement_type = constData.input_placement_type
        sequence_type = constData.sequence_type
        state_method = constData.state_method


        actions_map = None
        action_scheme = [4,5,6,7]
        #action_scheme = [4]
        ag = ParametricActionsGen()
        ag.GenerateActions()


        start = datetime.datetime.now().replace(microsecond=0)
        randomizer=random.randrange(10000)
        run_dir_name = "{}_{}_{}_df{}_tm{}_{}_{}".format(constData.algorithm, designName, constData.integration_mode, constData.bin_size_x, constData.test_mode, start.strftime("%Y%m%d_%H%M%S"),randomizer)
