import numpy as np
import copy
import argparse
import os
import sys
import sqlite3
import logging
import time
import psutil


from designgines.PLimportUcla import importUcla
from PLSummarizer import SummarizePlace
from dataModel.PLConstData import ConfigData
from dataModel.PLLayoutData import LayoutData
from design_engines.PLPAHierParametricFolding import PAHierParametricFolding
from design_admin.PLdm_clustering import design_manager_cluster
from graph_library.PlacementAwareHC import create_placement_aware_hierarchical_clusters

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s -  [%(pathname)s:%(lineno)d] %(message)s')


class HierLayoutController(object):
    def __init__(self, origLayout, binnedLayout, foldingParams, summarizer):
        self.origLayout = origLayout
        self.binnedLayout = binnedLayout
        self.foldingParams = foldingParams
        self.summarizer = summarizer

        self.newLayout  = None
        self.clusterMap = {}

        self.logger = logging.getLogger(__name__)

    def CreateHierClusters(self, layoutData, mode, orientation=0):
        #self.origLayout.layout.netlist.create_graph()
        if mode == 1:
            clusters = create_placement_aware_hierarchical_clusters(
                self.binnedLayout,
                orientation,
                self.foldingParams.numberOfCuts+1,
            )
        
        else:
            raise ValueError("mode not supported")

        self.Create3DLayout(
            clusters, 
            self.foldingParams.extendedPattern, 
            layoutData
        )

        return self.newLayout.layout

    def Create3DLayout(self, clusters, extendedPattern, layoutData):
        newGridDefinition = copy.deepcopy(self.origLayout.gridSpec)
        #newGridDefinition = copy.deepcopy(layoutData.grid_definition)
        newGridDefinition.numLayers = len(layoutData.constData.layer_values)
        newLayout = copy.deepcopy(self.origLayout)
        self.newLayout = newLayout
        clustersCount = len(set(clusters))
        assert( clustersCount == len(extendedPattern) ), f"clusters count = {clustersCount} != patternSize = {len(extendedPattern)}"

        #create cluster to layer mapping
        uniqueClusters = list(set(clusters))
        clusterToLayersMap = {}
        for i,k  in enumerate(extendedPattern):
            clusterToLayersMap[uniqueClusters[i]] = k

        #assign cluster and layers to nodes
        counter = 0
        for k,v in newLayout.layout.netlist.nodes.items():
            if not v.movable: continue
            v.cluster = clusters[counter]
            v.point_lb.z = float(
                clusterToLayersMap[clusters[counter]]
            )
            self.logger.info(f"node={k}, cluster={clusters[counter]} and layer={v.point_lb.z}")
            counter += 1

        #stack clusters to complete 3D Placement
        self.StackClusters(clusters, layoutData, clusterToLayersMap)
        return

    def UpdateNodesLocation(self, nodes, shift_x, shift_y, layer):
        for name, xl, yl in nodes:
            nodeObj = self.newLayout.layout.netlist.nodes[name]
            #self.logger.info(type(nodeObj.point_lb))
            nodeObj.point_lb.x = xl + shift_x
            nodeObj.point_lb.y = yl + shift_y
            nodeObj.point_lb.z = layer
            #xl += shift_x
            #yl += shift_y
            #self.logger.info(f"name={name}, xl={xl}, yl={yl}")

    def FindLowerLeftCell(self, coordinates):
        # Initialize variables to store the lowest xl and yl values
        lowest_xl = float('inf')  # Initialize with positive infinity to ensure any value will be lower
        lowest_yl = float('inf')

        # Initialize variable to store the name with lowest xl and yl
        name_with_lowest_xl_yl = None

        # Iterate through the list of tuples
        for name, xl, yl in coordinates:
            # Update lowest_xl and lowest_yl if current values are lower
            if xl < lowest_xl:
                lowest_xl = xl
                lowest_yl = yl
                name_with_lowest_xl_yl = name
            elif xl == lowest_xl and yl < lowest_yl:
                lowest_yl = yl
                name_with_lowest_xl_yl = name
        return name_with_lowest_xl_yl

    def FindShiftConstants(self, layoutData, clusterMap):
        shift_x = {}
        shift_y = {}
        for cluster, clusterData in clusterMap.items():
            #Find cell at the lower left of cluster with respect to 2D placement
            name_with_lowest_xl_yl = self.FindLowerLeftCell(clusterData['nodes'])
            self.logger.info(f"cluster={cluster}: lower left cell = {name_with_lowest_xl_yl}")

            #Find shift constatns with respect to cluster lower left
            llCellNode = self.newLayout.layout.netlist.nodes[name_with_lowest_xl_yl]
            self.logger.info(f"node={llCellNode},cluster={cluster}, clusterData={clusterData['point_lb']}")
            shift_x[cluster] = (clusterData['point_lb'][0] - llCellNode.point_lb.x)
            shift_y[cluster] = (clusterData['point_lb'][1] - llCellNode.point_lb.y)
            self.logger.info(f"cluster={cluster}, shift_x={shift_x[cluster]}, shift_y={shift_y[cluster]}")

        return shift_x, shift_y


    def ShiftCells(self, layoutData, clusterMap):
        shift_x, shift_y = self.FindShiftConstants(layoutData, clusterMap)
        for cluster, clusterData in clusterMap.items():
            #Update all cells with respect to shift
            layer = float(clusterData['layer'])
            self.UpdateNodesLocation(
                clusterData['nodes'],
                shift_x[cluster],
                shift_y[cluster],
                layer
            )

        return

    def InitializeLayer(self):
        layerData = {}
        #print(f"{self.newLayout.gridSpec}")
        for row in range(self.newLayout.gridSpec.numRows):
            layerData[row] = {}
            rowObj = self.newLayout.gridSpec.rows[row]
            layerData[row]['height'] = rowObj.height
            layerData[row]['width'] = rowObj.width
            layerData[row]['subRowOrigin'] = rowObj.subRowOrigin
            layerData[row]['coordinate'] = rowObj.coordinate
            layerData[row]['usedWidth'] = 0
            layerData[row]['freePoint'] = rowObj.subRowOrigin
        return layerData

    def StackClusters(self, clusters, layoutData, clusterToLayersMap):
        layoutUsedTable = {}
        
        #initialize cluster Map
        for cluster in list(set(clusters)):
            self.clusterMap[cluster] = {}

        for layer in layoutData.constData.layer_values:
            for cluster in list(set(clusters)):
                #self.logger.info(f"layer={layer} and xx={clusterToLayersMap[cluster]}")
                if int(layer) != int(clusterToLayersMap[cluster]): continue
                xminList = []
                xmaxList = []
                yminList = []
                ymaxList = []
                self.clusterMap[cluster]['nodes'] = []
                for k,v in self.newLayout.layout.netlist.nodes.items():
                    if v.cluster != cluster: continue
                    xminList.append(v.point_lb.x)
                    xmaxList.append(v.point_lb.x + v.width)
                    yminList.append(v.point_lb.y)
                    ymaxList.append(v.point_lb.y + v.height)
                    self.clusterMap[cluster]['nodes'].append([ v.name, v.point_lb.x, v.point_lb.y])
                cluster_width = max(xmaxList) - min(xminList)
                cluster_height =  max(ymaxList) - min(yminList)
                self.logger.info(f"{cluster}: cluster_width={cluster_width}, cluster_height={cluster_height}")

                self.clusterMap[cluster]['layer'] = layer
                self.clusterMap[cluster]['width'] = cluster_width
                self.clusterMap[cluster]['height'] = cluster_height
                self.clusterMap[cluster]['point_lb'] = None
                #self.logger.info(f"layer = {layer},  cluster = {cluster}, clusterMap = {clusterMap}")

        for layer in layoutData.constData.layer_values:
            layoutUsedTable[layer] = self.InitializeLayer()
            for cluster, cData in self.clusterMap.items():
                if int(layer) != int(clusterToLayersMap[cluster]): continue
                rowObj = self.newLayout.gridSpec.rows[0]
                rowHeight = rowObj.height
                rowWidth = rowObj.width * rowObj.numSites
                #clusterRowsCount = cData['height'] // rowHeight
                #self.logger.info(cData)
                cData['point_lb'] = None
                clusterRemaining = cData['height']
                for row, rowD in layoutUsedTable[layer].items():
                    currentRowObj = self.newLayout.gridSpec.rows[row]
                    if clusterRemaining <= 0: break
                    #self.logger.info(f"rowD = {rowD}")
                    if rowD['usedWidth'] >= rowWidth: continue

                    #checking if enough free space
                    freeWidth = rowWidth - rowD['usedWidth']
                    if cData['width'] > freeWidth:
                        self.logger.info(f"Can't fit cluster={cluster} with width {cData['width']} in the layer={layer}, row={row} with freewidth = {freeWidth}")
                        continue
                    rowD['freePoint'] = rowD['freePoint'] + cData['width']
                    rowD['usedWidth'] = rowD['usedWidth'] + cData['width']

                    clusterRemaining -= rowHeight

                    if cData['point_lb'] is not None: continue

                    cData['point_lb'] = [
                        currentRowObj.subRowOrigin, 
                        currentRowObj.coordinate 
                    ]

                    self.logger.info(f"cluster {cluster} point_lb = {cData['point_lb']}")

                #AMTODO: automate finding y origin of the layout
                ylOrigin = self.newLayout.gridSpec.rows[0].coordinate
                assert clusterRemaining <= ylOrigin, f"cluster {cluster} must be placed, remaining={clusterRemaining}. ylOrigin hardcoded to {ylOrigin}"
        #self.logger.info(f"layoutUsedTable={layoutUsedTable}")
        #self.logger.info(f"clusterMap={self.clusterMap}")
        #self.summarizer.draw_cv2_cluster_images(self.origLayout.layout.netlist, self.clusterMap)

        self.ShiftCells(layoutData, self.clusterMap)

        return

def RunHier3DConversion(layoutData, foldingParams, inputPlacementFile):
    con  = sqlite3.connect("/tmp/test.db")

    logger.info(layoutData.inputDir)

    logger.info("Populating map for window size coding based on the design")
    foldingParams.GetDivideFactors(layoutData)
    
    logger.info("Loading design ")
    dm = design_manager_cluster(
        foldingParams.bin_size_x,
        foldingParams.bin_size_y,
        layoutData,
        inputPlacementFile=inputPlacementFile
    )

    nl1 = dm.binnedLayout
    origLayout = dm.origLayout
    
    summarizer = SummarizePlace(
        nl1.netlist,
        con,
        {},
        {},
        ''
    )

    logger.info("Initializing Hier Layout Controller")
    controller = HierLayoutController(
        origLayout,
        nl1,
        foldingParams,
        summarizer
    )

    logger.info("Create Hierarchical clusters")
    newLayout = controller.CreateHierClusters(
        layoutData,
        1, # 0 for agglmoerative clusters and 1 for placement aware clusters
        foldingParams.direction
    )

    #summarizer.draw_cv2_cluster_images(i1.netlist, controller.clusterMap)

    logger.info("Finished Hierarchical clusters!")

    return newLayout, summarizer, dm

def Run():
    import runConfigs.PLconfig_grid as PLconfig_grid
    ag = PLconfig_grid.ag
    action = 135
    numberOfCuts, direction, pattern, windowSizeCode = ag.DecodeAction(action)
    logger.info(f"#########\n"
                f"numberOfCuts={numberOfCuts}\n"
                f"direction={direction}\n"
                f"pattern={pattern}\n"
                f"windowSizeCode={windowSizeCode} \n\n\n\n"
                )

    confData = ConfigData(
    ) #This is dumy class.AMTODO fix things. 

    layoutData = LayoutData(confData)

    logger.info("Creating folding params")

    foldingParams = PAHierParametricFolding(
        numberOfCuts,
        direction,
        pattern,
        windowSizeCode
    )

    inputPlacementFile = os.path.join(layoutData.inputDir, f"{layoutData.designName}.pl")
    
    logger.info("running 2d to 3d using HC")
    newLayout, summarizer, dm = RunHier3DConversion(layoutData, foldingParams, inputPlacementFile)

    nl1 = newLayout
    
    twl = nl1.netlist.twl(nl1.avg_sites_per_row, foldingParams.bin_size_x, 2)[0]
    last_cost = dm.origLayout.layout.netlist.twl_pahc_2d(dm.origLayout.layout.avg_sites_per_row, 1)[0]
    twlDelta = twl - last_cost
    #twlDelta = twl = last_cost = 0
    logger.info(f"Plane Locations twlDelt={twlDelta}, new_cost={twl}, last_cost={last_cost}")

    nl1.netlist.change_location_type()
    dm.origLayout.layout.netlist.change_location_type()
    twl = nl1.netlist.twl(nl1.avg_sites_per_row, foldingParams.bin_size_x, 2)[0]
    last_cost = dm.origLayout.layout.netlist.twl_pahc_2d(dm.origLayout.layout.avg_sites_per_row, 1)[0]
    twlDelta = twl - last_cost
    #twlDelta = twl = last_cost = 0
    logger.info(f"Grid Locations twlDelt={twlDelta}, new_cost={twl}, last_cost={last_cost}")

    cf = nl1.netlist.calculate_cost_function(nl1.avg_sites_per_row, foldingParams.bin_size_x, 2)
    last_cf = dm.origLayout.layout.netlist.calculate_cost_function(
        dm.origLayout.layout.avg_sites_per_row, 1, 1
    )
    logger.info(f"Grid Location new cf={cf}, last_cf={last_cf}, delta cf = {cf-last_cf}")
    
    logger.info("drawing images")
    #episode=0
    #summarizer.draw_image(newLayout, episode)
    #summarizer.draw_cv2_image(newLayout.netlist)
    #self.logger.info(newLayout.nodes)
    #for k,v in newLayout.netlist.nodes.items():        
    #    self.logger.info(type(v.point_lb))

def main():
    parser = argparse.ArgumentParser()    
    
    parser.add_argument("-exportName", action="store", dest="exportName",
                        help="design name",
                        required=False, type=str)    
    
    parser.add_argument("-exportDir", action="store", dest="exportDir",
                        help="folder of UCLA design files",
                        required=False, type=str)    

    parser.add_argument("-inputPlacementFile", action="store", 
                        dest="sInputPlacementFile",
                        help="folder of input UCLA design files",
                        required=False, type=str)

    args = parser.parse_args()

    process = psutil.Process(os.getpid())

    # Measure initial memory usage
    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    start_time = time.time()  # Start time tracking

    # Run the function
    Run()

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
