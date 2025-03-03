import os
import sys
import networkx as nx
#import matplotlib.pyplot as plt
import numpy as np

from designgines.PLLocationConversion import GridLocation, BinLocation, PlaneLocation, Bin2Grid, ThreeD2Grid, ThreeDLocation

class NGraph(object):
    def __init__(self, netlist):
        #input variable
        self.netlist = netlist

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

        self.initialize()

    def initialize(self):
        self.update_graph_props()
        self.update_adjacency_matrix()
        self.update_attributes_matrix()
        #print('properties_matrix', self.properties_matrix.shape, 'ajdacency_matrix', self.adjacency_matrix.shape)
        #disabling saving garph due to sbatch
        #self.save_graph_drawing()

    def update_graph(self):
        self.update_graph_props()
        self.update_adjacency_matrix()
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
                'width': node_object.width,
                'height': node_object.height,
                'hierarchy': node_object.hierarchy,
                'movable': node_object.movable,
                'terminalType': node_object.terminalType,
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
        distance_matrix = []
        for x1, y1 in self.attributes_content_list:
            row = []
            for x2, y2 in self.attributes_content_list:
                manhattan_distance = y2-y1 + x2-x1
                row.append(manhattan_distance)
            distance_matrix.append(row)

        # Convert to a NumPy array if needed
        self.distance_matrix = np.array(distance_matrix)

        for edge_object in self.netlist.edges.values():
            self.graph.add_edge(edge_object.v1.name, edge_object.v2.name)

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

def check_3DBinLayout(designName, inputDir):
    from design_manager.PLdm import design_manager
    dm = design_manager()
    nl1 = dm.layout_controller.newLayout
    nl1.netlist.create_graph()
    g1 = nl1.g_object
    g1.save_graph_drawing(designName)

def main():
    inputDir = os.path.abspath(PLconfig_grid.inputDir)
    designName = PLconfig_grid.designName

    check_3DBinLayout(designName, inputDir)

if __name__ == "__main__":
    import runConfigs.PLconfig_grid as PLconfig_grid
    import runConfigs.PLconfig_const as PLconfig_const
    main()
