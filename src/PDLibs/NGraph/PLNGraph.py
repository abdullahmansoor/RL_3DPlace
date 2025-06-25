import os
import sys
import networkx as nx
#import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import random
import math

# Add PDLibs to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "PDLibs"))
import runConfigs.PLconfig_grid as PLconfig_grid
from designgines.PLLocationConversion import GridLocation, BinLocation, PlaneLocation, Bin2Grid, ThreeD2Grid, ThreeDLocation

def max_min_landmarks_by_location(G, L):
    """
    Select L landmarks using max-min distance based on node placement (x, y).
    Assumes each node has: G.nodes[n]['node_object'].point_lb.{x, y}
    """
    nodes = list(G.nodes)
    n = len(nodes)

    # Extract (x, y) locations for all nodes
    coords = np.array([
        [G.nodes[n]['point_lb_x'], G.nodes[n]['point_lb_y']]
        for n in nodes
    ])
    
    # Choose the first landmark randomly
    selected = [random.choice(range(n))]  # use indices
    landmarks = [coords[selected[0]]]       # actual node IDs

    # Precompute distance matrix (n x n is too big → do pairwise on-the-fly)
    for _ in range(1, L):
        # Compute distance from each node to its closest current landmark
        dists = np.min([
            np.linalg.norm(coords - coords[i], axis=1)
            for i in selected
        ], axis=0)
        next_idx = np.argmax(dists)
        selected.append(next_idx)
        landmarks.append(coords[next_idx])
        

    print(landmarks)
    return landmarks

def compute_embedding_by_xy(G, landmark_xy):
    # Get node locations
    node_coords = np.array([
        [G.nodes[n]['point_lb_x'], G.nodes[n]['point_lb_y']]
        for n in G.nodes
    ])  # shape [n, 2]

    # Compute Euclidean distances to each landmark
    dist_matrix = np.linalg.norm(
        node_coords[:, None, :] - landmark_xy[None, :, :], axis=-1
    )  # shape [n_nodes, L]

    # Normalize and invert (closer = higher score)
    #dist_matrix = np.clip(dist_matrix, 0, 100) / 100
    #dist_matrix = 1.0 - dist_matrix

    return dist_matrix


def load_landmarks(file_path):
    return np.load(file_path)['landmarks']

def save_landmarks(landmarks, out_file):
    np.savez_compressed(out_file, 
                        #dist=dist_matrix, 
                        #node_ids=np.array(node_ids), 
                        landmarks=np.array(landmarks))


def compute_landmark_count(n_nodes, mode='sqrt', max_landmarks=256, k=3, mem_cap_mb=100):
    """
    Computes number of landmarks (L) based on graph size and heuristic.
    
    Parameters:
        n_nodes: int – number of nodes in the graph
        mode: str – one of ['sqrt', 'log', 'cap', 'ram']
        max_landmarks: int – upper cap on L for large graphs
        k: float – multiplier for log mode
        mem_cap_mb: int – memory budget in megabytes (used in 'ram' mode)
        
    Returns:
        L: int – number of landmarks
    """
    if mode == 'sqrt':
        return int(min(math.ceil(math.sqrt(n_nodes)), max_landmarks))
    elif mode == 'log':
        return int(min(math.ceil(math.log2(n_nodes) * k), max_landmarks))
    elif mode == 'cap':
        return int(min(math.ceil(math.sqrt(n_nodes)), max_landmarks))
    elif mode == 'ram':
        # Estimate max L for memory budget: n × L × 4 bytes (float32)
        L_ram = int((mem_cap_mb * 1024**2) / (n_nodes * 4))
        return min(L_ram, max_landmarks)
    else:
        raise ValueError(f"Unknown mode: {mode}")

class NGraph(object):
    def __init__(self, netlist, landmark_count_mode = 'sqrt'):
        #input variable
        self.netlist = netlist
        self.landmark_count_mode = landmark_count_mode

        #internal variables
        self.graph = None
        self.attributes_content_list = []
        self.attributes_header_list = [
            #'width',
            #'height',
            #'hierarchy',
            #'movable',
            #'terminalType',
            'point_lb_x',
            'point_lb_y'
        ]

        #output variables
        self.adjacency_matrix = None
        self.properties_matrix = None
        self.distance_matrix = None
        self.landmarks = compute_landmark_count(self.netlist.numNodes)
        self.landmarks_out = PLconfig_grid.inputDir / f"{PLconfig_grid.designName}_{self.landmark_count_mode}{self.landmarks}_landmark.npz"
        print("NGraph is               being initialiiiiiiiiiiiiiiiiiiiiized")
        self.initialize()
        print("NGraph init done!")

    def initialize(self):
        self.update_graph_props()
        #self.update_adjacency_matrix()
        #self.update_attributes_matrix()
        #print('properties_matrix', self.properties_matrix.shape, 'ajdacency_matrix', self.adjacency_matrix.shape)
        #disabling saving garph due to sbatch
        #self.save_graph_drawing()

    def update_graph(self):
        self.update_graph_props()
        #self.update_adjacency_matrix()
        #self.update_attributes_matrix()

    def update_graph_props(self):
        self.graph = nx.Graph()
        self.attributes_content_list = []
        for node_object in self.netlist.nodes.values():
            x = None
            y = None
            if isinstance(node_object.point_lb, GridLocation):
                grid_location = node_object.point_lb
                x = grid_location.xgrid
                y = grid_location.ygrid
            #if hasattr(node_object.point_lb, 'bin_number') and hasattr(node_object.point_lb, 'yrow'):
            elif isinstance(node_object.point_lb, BinLocation):
                grid_location = Bin2Grid(node_object.point_lb).grid_location
                x = grid_location.xgrid
                y = grid_location.ygrid
            elif isinstance(node_object.point_lb, ThreeDLocation):
                grid_location = ThreeD2Grid(node_object.point_lb).grid_location
                x = grid_location.xgrid
                y = grid_location.ygrid
            elif isinstance(node_object.point_lb, PlaneLocation):
                x = node_object.point_lb.x
                y = node_object.point_lb.y
            else:
                raise ValueError("The location type is not supported {}".format(type(node_object.point_lb)))
            attributes = {
                'name': node_object.name,
                #'width': node_object.width,
                #'height': node_object.height,
                #'hierarchy': node_object.hierarchy,
                #'movable': node_object.movable,
                #'terminalType': node_object.terminalType,
                #attributes['pins'] = node_object.pin
                'point_lb_x': x,
                'point_lb_y': y
                 }
            self.attributes_content_list.append(
                [
                    #node_object.width,
                    #node_object.height,
                    #node_object.hierarchy,
                    #node_object.movable,
                    #node_object.terminalType,
                    x,
                    y
                ]
            )
            node_info = (node_object.name, attributes)
            self.graph.add_nodes_from([node_info])
            '''
            name = int(node_object.name[1:])
            self.graph.add_node(name)
            self.graph.nodes[name]['name'] = node_object.name
            '''

        #print(self.attributes_content_list, len(self.attributes_content_list), len(self.graph.nodes))

        # Compute distance matrix
        #distance_matrix = []
        #for x1, y1 in self.attributes_content_list:
        #    row = []
        #    for x2, y2 in self.attributes_content_list:
        #        manhattan_distance = y2-y1 + x2-x1
        #        row.append(manhattan_distance)
        #    distance_matrix.append(row)

        # Convert to a NumPy array if needed
        #self.distance_matrix = np.array(distance_matrix)

        for edge_object in self.netlist.edges.values():
            self.graph.add_edge(edge_object.v1.name, edge_object.v2.name)

        del self.netlist
        import gc
        gc.collect()
        
    def setup_landmarks(self):
        G = self.graph
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        landmarks = max_min_landmarks_by_location(G, self.landmarks)
        print(f"Selected {self.landmarks} landmarks={landmarks}")
        
        save_landmarks(landmarks, self.landmarks_out)
        print(f"Saved {len(landmarks)} landmarks to {self.landmarks_out}")
        
    def get_landmark_distance_matrix(self):
        G = self.graph
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        landmarks = load_landmarks(self.landmarks_out)  # shape [L]

        print("Computing distance matrix...")
        dist_matrix= compute_embedding_by_xy(G, landmarks)

        return dist_matrix

    def update_adjacency_matrix(self):
        self.adjacency_matrix = nx.adjacency_matrix(self.graph)
        #print(type(self.adacency_matrix), self.adacency_matrix)
        #print(node_object.data)

    def update_attributes_matrix(self):
        '''
        prop = NodeAttributesTable(
            self.attributes_header_list,
            self.attributes_content_list
        )
        self.properties_matrix = prop.attributes_matrix.todense()
        print(self.properties_matrix)
        '''
        self.properties_matrix = self.distance_matrix
    '''
    def save_graph_drawing(self, name="netlist_graph"):
        nx.draw_spectral(self.graph, with_labels = True)
        plt.title("Netlist Graph Drawing")
        file_name = name + "_graph.png" 
        plt.savefig(file_name)
        plt.close()
    '''

def Run(args):
    if args.sDesignName:
        designName = args.sDesignName
    else:
        designName = PLconfig_grid.designName
    inputDir = PLconfig_grid.inputDir
    confData = ConfigData()
    layoutData = LayoutData(confData, designName=designName)

    layout = importUcla(
        name=layoutData.designName,
        path=layoutData.inputDir,
        #inputPlacementFile=inputPlacementFile
    )
    layout.readNetsFile(f"{layoutData.inputDir}/{layoutData.designName}.nets")

    #layout.netlist.create_graph()
    g1 = NGraph(layout.netlist, args.landmark_count_mode)
    g1.setup_landmarks()

    dist_matrix = g1.get_landmark_distance_matrix()
    print(dist_matrix)


def main(args):
    Run(args)

if __name__ == "__main__":
    from designgines.PLimportUcla import importUcla

    # Add PDLibs to sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "rl_3dplace"))
    
    from dataModel.PLConstData import ConfigData
    from dataModel.PLLayoutData import LayoutData

    parser = argparse.ArgumentParser()    
    

    parser.add_argument("-inputPlacement", action="store", 
                        dest="sInputPlacement",
                        help="Placement file for evaluation",
                        required=False, type=str)
    parser.add_argument("-designName", action="store", 
                        dest="sDesignName",
                        help="Name of design for DHCARL flow",
                        required=False, type=str)
    parser.add_argument("-landmark_count_mode", action="store", 
                        dest="landmark_count_mode",
                        help="landmark_count_mode",
                        required=False, default="sqrt",  type=str)
    args = parser.parse_args()    
    main(args)
