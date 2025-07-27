import argparse
import sys
import time
import psutil
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parents[2] / "PDLibs"))
from designgines.PLimportUcla import importUcla

sys.path.append(str(Path(__file__).resolve().parents[2] / "rl_3dplace"))
from dataModel.PLConstData import ConfigData
from dataModel.PLLayoutData import LayoutData


def compute_hpwl(netlist):
    """
    Compute Half-Perimeter Wire Length (HPWL) in the XY plane only.
    Assumes all node.point_lb are PlaneLocation.
    Works on 2-pin edges only (not hypernets).
    """
    total_hpwl = 0.0

    for edge in netlist.edges.values():
        p1 = edge.v1.point_lb
        p2 = edge.v2.point_lb

        if not hasattr(p1, "x") or not hasattr(p2, "x"):
            raise TypeError("HPWL calculation expects PlaneLocation with .x and .y")

        dx = abs(p2.x - p1.x)
        dy = abs(p2.y - p1.y)
        total_hpwl += dx + dy

    return total_hpwl

def compute_area(netlist):
    """
    Compute Half-Perimeter Wire Length (HPWL) in the XY plane only.
    Assumes all node.point_lb are PlaneLocation.
    Works on 2-pin edges only (not hypernets).
    """
    total_hpwl = 0.0
    xlist = []
    ylist = []
    for edge in netlist.edges.values():
        p1 = edge.v1.point_lb
        p2 = edge.v2.point_lb

        if not hasattr(p1, "x") or not hasattr(p2, "x"):
            raise TypeError("HPWL calculation expects PlaneLocation with .x and .y")

        xlist.append(p2.x)
        xlist.append(p1.x)
        ylist.append(p2.y)
        ylist.append(p1.y)
    xmin = min(xlist)
    xmax = max(xlist)
    ymin = min(ylist)
    ymax = max(ylist)
    area = (xmax - xmin + 1)*(ymax - ymin + 1)

    return area

def compute_miv_count_from_edges(netlist, miv_weight=1.0):
    miv_count = 0
    for edge in netlist.edges.values():
        p1, p2 = edge.v1.point_lb, edge.v2.point_lb
        if hasattr(p1, 'z') and hasattr(p2, 'z'):
            if p1.z != p2.z:
                miv_count += 1
    return miv_weight * miv_count

def Run(args):
    import runConfigs.PLconfig_grid as PLconfig_grid

    if args.sDesignName:
        designName = args.sDesignName
    else:
        designName = PLconfig_grid.designName


    confData = ConfigData()
    layoutData = LayoutData(confData, designName=designName)

    if args.sInputPlacementFile:
        inputPlacementFile = args.sInputPlacementFile
    else:
        inputPlacementFile = os.path.join(layoutData.inputDir, f"{layoutData.designName}.pl")
    
    layout = importUcla(
        name=designName,
        path=layoutData.inputDir,
        inputPlacementFile=inputPlacementFile
    )
    layout.readNetsFile(f"{layoutData.inputDir}/{layoutData.designName}.nets")
    netlist = layout.netlist

    hpwl = compute_hpwl(netlist)
    miv_count = compute_miv_count_from_edges(netlist)
    miv_cost = compute_miv_count_from_edges(netlist, 0.1)
    twl = hpwl + miv_cost
    area = compute_area(netlist)
    alpha = 0.4
    beta = 0.6
    cost = alpha * twl + beta * area
    units = "pdk"
    if layoutData.category in ['small', 'medium']:
        netlist.change_location_type()
        units = "grid"
    print(f"Design: {designName}")
    print(f"Units {units}")
    print(f"HPWL (X+Y):         {hpwl:.2f}")
    print(f"MIV Count:          {miv_count}")
    print(f"Area:          {area}")
    print(f"Total WL:        {twl:.2f}")
    print(f"Total cost:        {cost:.2f}")    
    
def main():

    parser = argparse.ArgumentParser()    
    
    parser.add_argument("-designName", action="store", 
                        dest="sDesignName",
                        help="Name of design for DHCARL flow",
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


