import argparse
import os
import copy
import sqlite3
import sys
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s -  [%(pathname)s:%(lineno)d] %(message)s'
)

from designgines.PLimportUcla import importUcla
#from PLSummarizer import SummarizePlace
from design_engines.placement_helpers import change_divide_ratio
from designgines.PLGridSpec import BinnedGrid

from design_engines.placement_layout import placement_layout
from design_engines.PLPAHierParametricFolding import PAHierParametricFolding


class DummyLayout(object):
    def __init__(self, layout, gridSpec):
        self.layout = layout
        self.gridSpec = gridSpec


class design_manager_cluster(object):
    def __init__(self, bin_size_x, bin_size_y, layout_data, name=None, path=None, inputPlacementFile=None):

        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.layout_data = layout_data
        self.name = name
        if not self.name:
            self.name=layout_data.designName
        self.path = path
        if not self.path:
            self.path=layout_data.inputDir
        self.inputPlacementFile = inputPlacementFile

        self.binned_layout = None
        self.binnedLayout = None #temporarily for backwoard compatbility
        self.origLayout = None
        self.design_summary = None
        self.cell_bin_mapping = {}

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s -  [%(pathname)s:%(lineno)d] %(message)s'
        )

        self.import_and_bin_design()

    def import_and_bin_design(self):
        # Import the design
        imported_design = importUcla(
            name=self.name,
            path=self.path,
            inputPlacementFile=self.inputPlacementFile
        )

        # Create a binned grid specification
        imported_design_orig = copy.deepcopy(imported_design)
        #print(f"{imported_design_orig.gridDefinition}")

        self.origLayout = DummyLayout(
            imported_design_orig,
            imported_design_orig.gridDefinition
        )
        
        grid_spec = imported_design.gridDefinition
        single_cell_height = grid_spec.rows[1].coordinate - grid_spec.rows[0].coordinate
        
        binned_grid = BinnedGrid.from_grid(
            grid_spec,
            self.bin_size_x,
            self.bin_size_y
        )

        # Create a binned layout
        self.binned_layout = placement_layout(
            grid_spec=binned_grid,
            netlist=imported_design.netlist,
            num_nodes=imported_design.numNodes,
            num_rows=binned_grid.numRows,
            avg_sites_per_row=binned_grid.avg_sites_per_row,
            bins_count=binned_grid.total_sites,
            bin_size_x=self.bin_size_x,
            bin_size_y=self.bin_size_y,
            single_cell_height=single_cell_height
        )

        self.binned_layout.initialize()
        self.binnedLayout = self.binned_layout

        self.verify()

    def verify(self):
        nodes_count = self.origLayout.layout.netlist.numNodesWOTerminals
        self.logger.info(f"number of nodes from netlist={nodes_count}")

        matrix_dict = self.binnedLayout.cell_bin_mapping
        nodes = []
        for k in matrix_dict.values():
            nodes += k

        nodes = [ x for x in nodes if x != None]
        matrix_count = len(nodes)


        if nodes_count != matrix_count:
            netlist_nodes = list(self.origLayout.layout.netlist.nodes.keys())

            missing_nodes = [x for x in netlist_nodes if x not in nodes]

            self.logger.warning(f"missing nodes={missing_nodes}")

            matrix = self.binnedLayout.matrix['cellMatrix']
            shape = f"{len(matrix)}x{len(matrix[0])}x{len(matrix[0][0])}"

            self.logger.warning(f"shape={shape}")
            
        assert(nodes_count == matrix_count), f"netlist nodes coount {nodes_count} != matrix nodes count {matrix_count}"

        
def RunDM():
    from dataModel.PLConstData import ConfigData
    from dataModel.PLLayoutData import LayoutData


    con  = sqlite3.connect("/tmp/test.db")

    numberOfCuts = 1
    direction = 1 # 0 for horizontal bins and 1 for vertical bins
    pattern = [0, 1]
    windowSizeCode = 0
    
    logger.info("Creating folding params")

    foldingParams = PAHierParametricFolding(
        numberOfCuts,
        direction,
        pattern,
        windowSizeCode
    )
    
    confData = ConfigData()

    layoutData = LayoutData(confData)
    foldingParams.GetDivideFactors(layoutData)

    bin_size_x = foldingParams.bin_size_x
    bin_size_y = foldingParams.bin_size_y


    confData = ConfigData(
        bin_size_x=bin_size_x,
        bin_size_y=bin_size_y
    )

    layoutData = LayoutData(confData)
    inputPlacementFile = os.path.join(layoutData.inputDir, f"{layoutData.designName}.pl")
    logger.info(layoutData.inputDir)

    
    logger.info("starting dm")
    dm = design_manager_cluster(
        bin_size_x,
        bin_size_y,
        layoutData,
        inputPlacementFile=inputPlacementFile,
    )
    logger.info("dm loaded")
    nl1 = dm.binnedLayout

    '''
    summarizer = SummarizePlace(
        nl1.netlist,
        con,
        {},
        {},
        ''
    )
    '''
    #summarizer.draw_cv2_image(newLayout.netlist)

def main():
    parser = argparse.ArgumentParser()    
    

    args = parser.parse_args()
    RunDM()

if __name__ == "__main__":
    main()
