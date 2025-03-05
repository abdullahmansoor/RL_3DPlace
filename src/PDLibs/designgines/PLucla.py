import numpy as np
import networkx as nx
import copy
from networkx.utils import cuthill_mckee_ordering
from scipy.spatial.distance import cityblock

from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.cluster import KMeans

import torch
from torch_geometric.data import Data

from NGraph.PLNGraph import NGraph
from designgines.PLGeometricLib import Point, BBox
from designgines.PLLocationConversion import PlaneLocation, GridLocation, BinLocation, ThreeDLocation, Bin2Grid, ThreeD2Grid
from designgines.PLLocationConversion import Grid2Plane, Plane2Grid

#from spektral.data import Graph
#from spektral.data import DisjointLoader

import runConfigs.PLconfig_grid as PLconfig_grid

class Netlist(object):
    def __init__(self, **kwargs):
        self.data = {'nodes' : {}, 'nets' : {}, 'edges' : {}, 'numNodes' : None, 'numTerminals' : None, 'numNets' : None, 'numPins' : None}

        self.g_object = None
        self.graph = None
        self.adjacency_matrix = None
        self.properties_matrix = None
        self.rcm = None
        self.inputs = None
        self.placement_lists = []
        self.placement1_array = None
        self.placement2_array = None
        self.numNodesWOTerminals = None

        self.features = {
            'area' : None,
            'area1' : None,
            'area2' : None,
            'density' : None,
            'density1' : None,
            'density2' : None,
            'com' : None,
            'com1x' : None,
            'com1y' : None,
            'com2x' : None,
            'com2y' : None,
            #'moix' : None,
            #'moix1' : None,
            #'moix2' : None,
            #'moiy' : None,
            #'moiy1' : None,
            #'moiy2' : None,
            #'ent' : None,
            #'ent1' : None,
            #'ent2' : None,
            #'spct' : None,
            #'spct1' : None,
            #'spct2' : None,
        }
        

    @property
    def nodes(self):
        return self.data['nodes']
    @property
    def nets(self):
        return self.data['nets']
    @property
    def edges(self):
        return self.data['edges']
    @property
    def numNodes(self):
        return self.data['numNodes']
    @property
    def numTerminals(self):
        return self.data['numTerminals']
    @property
    def numNets(self):
        return self.data['numNets']
    @property
    def numPins(self):
        return self.data['numPins']

    @nodes.setter
    def nodes(self, v1):
        self.data['nodes']=v1
    @nets.setter
    def nets(self, v1):
        self.data['nets']=v1
    @edges.setter
    def edges(self, v1):
        self.data['edges']=v1
    @numNodes.setter
    def numNodes(self, v1):
        self.data['numNodes']=v1
    @numTerminals.setter
    def numTerminals(self, v1):
        self.data['numTerminals']=v1
    @numNets.setter
    def numNets(self, v1):
        self.data['numNets']=v1

    @numPins.setter
    def numPins(self, v1):
        self.data['numPins']=v1

    def addNode(self,v1):
        self.data['nodes'][v1.name]= v1

    def addEdge(self,v1):
        self.data['edges'][v1.name]=v1

    def addNet(self,v1):
        # Assume nodes are loaded before nets
        self.data['nets'][v1.name]=v1
        #Turning off code to update nodes with pins as the searching in list is taking lot of time
        # this work will be done in the dictionary
        for pin in v1.pins:
            #WIP logic to update pins in respect nodes and created edges
            node1 = self.nodes[pin.node]
            if node1:
                node1.addpin(pin)
            else:
                raise ValueError("Can't find node related to pin, %s" % pin)
        #Turning off code to add edges  as the searching in list is taking lot of time
        # this work will be done with dictionary
        for i in range(len(v1.nodes)-1):
            n1Name = v1.nodes[i].node
            n1 = self.nodes[n1Name]
            n2Name = v1.nodes[i+1].node
            n2 = self.nodes[n2Name]
            name1 = n1Name + "-" + n2Name
            self.addEdge(Edge(v1=n1, v2=n2, name = name1, netName= v1.name))

    def nodeIndexAtCellName(self, leafNodeName):
        # return pointer of node object matching Name
        for node in self.nodes:
            if node.name == leafNodeName:
                return node
        raise ValueError("Couldn't find leaf node, %s" % leafNodeName)

    def dump_graph(self, file_name):
        fh = open(file_name, 'w')
        graph = ''
        sum1 = 0
        for key, node in self.nodes.items():
            name = node.name[1:]
            net_names  = [x.net for x in node.pins]
            nets = [ self.nets[x] for x in net_names]
            nodes = [ node.name[1:] ]
            for net in nets:
                for pin in net.pins:
                    nodes.append(pin.node[1:])
                    nodes_uniq = list(set(nodes))
                    degree = len(nodes_uniq)
                    sum1 += degree
                    line = str(degree) + "\t" + "\t".join(nodes_uniq) + "\n"
                    fh.write(line)
        header = "0\n"
        header += "%s\t%s\n" % (self.numNodes, sum1)
        header += "1\t000\n"
        fh.write(header)
        fh.close()
        print(f"done dumping graph={file_name}")


    def generate_encoding(self, sp1):
        #HVZ-Seuence
        #result = [ [int(sq.name[1:])] for sq in sp1.pos_sequence ] + [ [int(sq.name[1:])] for sq in sp1.neg_sequence ] + [ [int(sq.name[1:])] for sq in sp1.z_sequence ]

        #H-Sequence
        result = [ [int(sq.name[1:])] for sq in sp1.pos_sequence ]
        return np.array(result)

        #AvgPosHVZ
        result = []
        pos_sequence = [sq.name for sq in sp1.pos_sequence]
        neg_sequence = [sq.name for sq in sp1.neg_sequence]
        z_sequence = [sq.name for sq in sp1.z_sequence]
        #print("####AMDEBUG", pos_sequence, neg_sequence, z_sequence)
        for node, nodeObj in self.nodes.items():
            node_indices = []
            node_indices.append(pos_sequence.index(node))
            node_indices.append(neg_sequence.index(node))
            node_indices.append(z_sequence.index(node))
            max_value = len(pos_sequence)
            #result.append( [np.sum(node_indices)*1] )
            #result.append( node_indices )
        return np.array(result)

    def change_location_type(self):
        for v in self.nodes.values():
            location = v.point_lb
            if isinstance(location, PlaneLocation):
                plane_to_grid = Plane2Grid(location)
                grid_location = plane_to_grid.grid_location
                v.point_lb = grid_location

    def calculate_cost_function(self, avg_sites_per_row, divide_factor_x, number_of_layers):
        twl, twl_xy, twl_z, xlist, ylist, nodes_visited, offset = self.twl(avg_sites_per_row, divide_factor_x, number_of_layers)
        
        area = self.area(xlist, ylist, nodes_visited, offset)

        alpha = 0.4
        beta = 0.6
        return alpha * twl + beta * area

    def twl_pahc_2d(self, avg_sites_per_row, divide_factor_x):
        for v in self.nodes.values():
            location = v.point_lb
            #if isinstance(location, GridLocation):
            #    grid_to_plane = Grid2Plane(location)
            #    plane_location = grid_to_plane.plane_location
            #    v.point_lb = plane_location
        twl, twl_xy, twl_z, xlist, ylist, nodes_visited, offset = self.twl(avg_sites_per_row, divide_factor_x, 1)
        return twl, twl_xy, twl_z, xlist, ylist, nodes_visited, offset

    def check_placement(self):
        locations = set()
        for v in self.nodes.values():
            location = v.point_lb
            if not hasattr(location, 'z'):
                location.z=0
            locations.add((location.x, location.y, location.z))
        assert(self.numNodes == len(locations)), f"numer of unique nodes {self.numNodes} and locations mismatch, {len(locations)}, {locations}"

    def twl(self, avg_sites_per_row, divide_factor_x, number_of_layers):
        debug = False
        #self.check_placement()
        twl=0
        twl_z = 0
        twl_xy = 0
        xlist = []
        ylist = []
        nodes_visited = []
        offset = (avg_sites_per_row * divide_factor_x)//number_of_layers
        for edgeObj in self.edges.values():
            name = edgeObj.name
            nodes_visited += name.split('-')
            p1 = copy.deepcopy(edgeObj.v1.point_lb)
            p2 = copy.deepcopy(edgeObj.v2.point_lb)
            if isinstance(p1, GridLocation):
                xlist.append(p2.xgrid)
                xlist.append(p1.xgrid)
                ylist.append(p2.ygrid)
                ylist.append(p1.ygrid)
                #print(f"{p2}")
                twl += abs(p2.xgrid-p1.xgrid) + abs(p2.ygrid-p1.ygrid)
            elif isinstance(p1, PlaneLocation):
                xlist.append(p2.x)
                xlist.append(p1.x)
                ylist.append(p2.y)
                ylist.append(p1.y)
                #print(f"{p1}-{p2}")
                twl += abs(p2.x-p1.x) + abs(p2.y-p1.y)
                if p1.z != p2.z:
                    twl += 0.1
            elif isinstance(p1, BinLocation):
                b2g1 = Bin2Grid(p1)
                b2g2 = Bin2Grid(p2)
                xlist.append(b2g2.grid_location.xgrid)
                xlist.append(b2g1.grid_location.xgrid)
                ylist.append(b2g2.grid_location.ygrid)
                ylist.append(b2g1.grid_location.ygrid)
                twl += abs(b2g2.grid_location.xgrid-b2g1.grid_location.xgrid) + abs(b2g2.grid_location.ygrid-b2g1.grid_location.ygrid)
            elif isinstance(p1, ThreeDLocation):
                '''
                Logic:
                Convert Global bin number to PerLayer local bin number
                Then convert PerLayer bin number to Grid number
                TWL is caclulated based on Grid and equavalent to true 3D values
                '''
                global_bin1 = p1.bin_number
                global_bin2 = p2.bin_number

                layer1, perlayer_bin1 = PLconfig_grid.folded_bins_map.global_to_perlayer_bin(global_bin1)
                layer2, perlayer_bin2 = PLconfig_grid.folded_bins_map.global_to_perlayer_bin(global_bin2)
                p1_layer = p1.zlayer = layer1 
                p2_layer = p2.zlayer = layer2

                b2g1 = ThreeD2Grid(p1)
                b2g2 = ThreeD2Grid(p2)
                p1_x = None
                p2_x = None
                if p2_layer == 1:
                    p2_x = b2g2.grid_location.xgrid - offset
                else:
                    p2_x = b2g2.grid_location.xgrid
                xlist.append(p2_x)
                if p1_layer == 1:
                    p1_x = b2g1.grid_location.xgrid - offset
                else:
                    p1_x = b2g1.grid_location.xgrid
                xlist.append(p1_x)
                ylist.append(b2g2.grid_location.ygrid)
                ylist.append(b2g1.grid_location.ygrid)
                wl_xy = abs(p2_x-p1_x) + abs(b2g2.grid_location.ygrid-b2g1.grid_location.ygrid)
                twl_xy += wl_xy
                wl_z = abs(b2g2.grid_location.zgrid-b2g1.grid_location.zgrid)*0.1 
                twl_z += wl_z
                wl = wl_xy + wl_z
                twl += wl
                if debug:
                    print('\n',edgeObj.name)
                    print('p1=', p1)
                    print('p2=', p2)
                    print('wl=', wl)
                    print('wl_xy=', wl_xy)    
                    print('wl_z=', wl_z)
                    print('xlist=', p2_x, ',', p1_x)
                    print('ylist=', b2g2.grid_location.ygrid,', ',  b2g1.grid_location.ygrid)
            else:
                raise TypeError("Cannot calculate WL due to datatype error")
        #print(twl, twl_xy, twl_z, xlist, ylist, nodes_visited, offset)
        return twl, twl_xy, twl_z, xlist, ylist, nodes_visited, offset

    def area(self, xlist, ylist, nodes_visited, offset):
        for node, nodeObj in self.nodes.items():
            if node in nodes_visited: continue
            #print("missed node ={}".format(node))
            p1 = copy.deepcopy(nodeObj.point_lb)
            if isinstance(p1, GridLocation):
                xlist.append(p1.xgrid)
                ylist.append(p1.ygrid)
            elif isinstance(p1, BinLocation):
                b2g1 = Bin2Grid(p1)
                xlist.append(b2g1.grid_location.xgrid)
                ylist.append(b2g1.grid_location.ygrid)
            elif isinstance(p1, ThreeDLocation):
                global_bin1 = p1.bin_number

                layer1, perlayer_bin1 = PLconfig_grid.folded_bins_map.global_to_perlayer_bin(global_bin1)
                p1_layer = p1.zlayer = layer1 

                b2g1 = ThreeD2Grid(p1)
                p1_x = None
                if p1_layer == 1:
                    p1_x = b2g1.grid_location.xgrid - offset
                else:
                    p1_x = b2g1.grid_location.xgrid
                xlist.append(p1_x)
                ylist.append(b2g1.grid_location.ygrid)
        xmin = min(xlist)
        xmax = max(xlist)
        ymin = min(ylist)
        ymax = max(ylist)
        area = (xmax - xmin + 1)*(ymax - ymin + 1)
        return area

    def generate_features(self):
        self.create_np_placement_array()
        self.calculate_area_density()
        self.calculate_com_moi() #com = Center of mass, moi = Moment of inertia

        #AMTODO:understand and fix Maximal Independent Grp based symmetry
        #self.calculate_mig_symm()

        #AMTODO: undersnd and fix entropy based randomness measure
        #self.calculate_ent_symm()

        #self.calculate_spct_symm()
        #print(self.features)

    def create_np_placement_array(self):
        placement1_array_list = []
        placement2_array_list = []
        for _, nodeObj in self.nodes.items():
            p1 = nodeObj.point_lb
            if isinstance(p1, GridLocation):
                point_lb = p1
                placement1_array_list.append([point_lb.xgrid, point_lb.ygrid])
            elif isinstance(p1, BinLocation):
                b2g1 = Bin2Grid(p1)
                point_lb = b2g1.grid_location
                if point_lb.zgrid == 1:
                    placement2_array_list.append([point_lb.xgrid, point_lb.ygrid])
                #####AMTOO: Make it generic to all location types
                else:
                    placement1_array_list.append([point_lb.xgrid, point_lb.ygrid])
            elif isinstance(p1, ThreeDLocation):
                gridLocObject = ThreeD2Grid(p1)
                point_lb = gridLocObject.grid_location
                #print(point_lb)
                if point_lb.zgrid == 1:
                    placement2_array_list.append([point_lb.xgrid, point_lb.ygrid])
                #####AMTOO: Make it generic to all location types
                else:
                    placement1_array_list.append([point_lb.xgrid, point_lb.ygrid])
            else:
                raise ValueError("Loaction type is not supported")
        self.placement1_array = np.array(placement1_array_list) if placement1_array_list else np.array([[0,0]])
        self.placement2_array = np.array(placement2_array_list) if placement2_array_list else np.array([[0,0]])
        self.placement_lists = [self.placement1_array, self.placement2_array ]

    def calculate_area_density(self):
        areaList = []
        densityList = []
        for placement_array in self.placement_lists:
            # Calculate the bounding box of the placement
            min_x, min_y = np.min(placement_array, axis=0)
            max_x, max_y = np.max(placement_array, axis=0)
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            # Calculate the area of the placement
            area = width * height
            
            # Calculate the density of cells as the number of cells divided by the area
            num_cells = placement_array.shape[0]
            density = num_cells / area if area else 0.0
            
            areaList.append(area)
            densityList.append(density)

            #print("Density of cells in the placement:", density)
        self.features['area1'] = float(areaList[0])
        self.features['area2'] = float(areaList[1])
        self.features['area'] = float(max(areaList))

        self.features['density1'] = float(densityList[0])
        self.features['density2'] = float(densityList[1])
        self.features['density'] = float(max(densityList))


    def calculate_com_moi(self):
        comList = []
        moixList = []
        moiyList = []
        for placement in self.placement_lists:
            # Calculate the center of mass of the cells
            center_of_mass = np.mean(placement, axis=0)
            comList.append(center_of_mass)

            # Calculate the distances of each cell from the center of mass
            distances = placement - center_of_mass

            # Calculate the moment of inertia around the x-axis
            Ix = np.sum(distances[:, 1]**2)
            moixList.append(Ix)

            # Calculate the moment of inertia around the y-axis
            Iy = np.sum(distances[:, 0]**2)
            moiyList.append(Iy)

            #print("com={}, moix={}, moiy={}:".format(center_of_mass, Ix, Iy))

        self.features['com1x'] = comList[0][0]
        self.features['com1y'] = comList[0][1]
        self.features['com2x'] = comList[1][0]
        self.features['com2y'] = comList[1][1]
        self.features['com'] = 1/(cityblock(comList[0],comList[1])+0.01)

        '''
        self.features['moix1'] = moixList[0]
        self.features['moix2'] = moixList[1]
        self.features['moix'] = sum(moixList)

        self.features['moiy1'] = moiyList[0]
        self.features['moiy2'] = moiyList[1]
        self.features['moiy'] = sum(moiyList)
        '''

    def calculate_mig_symm(self): #entropy-based measure of symmetry
        entList = []
        for placement in self.placement_lists:
            # Define a threshold for considering two cells as symmetrically equivalent
            #print(placement)
            if len(placement) == 1:
                entList.append(0)
                continue

            epsilon = 1

            # Build a placement graph using the networkx library
            G = nx.Graph()
            for i in range(placement.shape[0]):
                for j in range(i+1, placement.shape[0]):
                    #print(placement[i], placement[j])
                    if np.allclose(placement[i], placement[j], atol=epsilon):
                        G.add_edge(i, j)
            #print('len(edges)', len(G.edges), 'edges', G.edges)
            # Find the maximal independent symmetry groups using the networkx library
            misgs = list(nx.maximal_independent_set(G))
            #print('misgs=', misgs)
            # Print the size of the largest MISG
            if len(misgs) > 0:
                max_misg_size = max(misgs)
                entList.append(max_misg_size)
                print("Size of the largest MISG:", max_misg_size)
            else:
                print("No MISGs found.")
                entList.append(0)
        self.features['ent1'] = entList[0]
        self.features['ent2'] = entList[1]
        self.features['ent'] = sum(entList)

    def calculate_ent_symm(self):
        entList = []
        for placement in self.placement_lists:
            # Define a resolution for the entropy grid
            resolution = 3

            # Calculate the range of x and y coordinates in the placement
            x_min, y_min = np.min(placement, axis=0)
            x_max, y_max = np.max(placement, axis=0)

            # Calculate the number of cells in the placement
            num_cells = placement.shape[0]
            #print("num_cells", num_cells)
            # Create a grid of points with the specified resolution
            x_grid = np.arange(x_min, x_max+resolution, resolution)
            y_grid = np.arange(y_min, y_max+resolution, resolution)
            X, Y = np.meshgrid(x_grid, y_grid)

            # Count the number of cells in each grid cell
            counts, _, _ = np.histogram2d(placement[:, 0], placement[:, 1], bins=[x_grid, y_grid])

            # Calculate the probability distribution of cells in the grid
            p = counts / num_cells
            #print("probability distribution of cells", p, 'counts=', counts)

            # Calculate the entropy of the placement
            entropy = -np.sum(p * np.log2(p))

            # Calculate the normalized entropy of the placement
            norm_entropy = entropy / np.log2(num_cells)
            entList.append(norm_entropy)

            #print("Entropy of the placement:", entropy)
            #print("Normalized entropy of the placement:", norm_entropy)
        self.features['ent1'] = entList[0]
        self.features['ent2'] = entList[1]
        self.features['ent'] = sum(entList)

    def calculate_spct_symm(self):
        spctList = []
        for placement in self.placement_lists:
            if len(placement) == 1:
                spctList.append(0)
                continue

            # Calculate the pairwise distances between cells
            dist_matrix = squareform(pdist(placement,  'cityblock'))
            #print("pdist(placement)", pdist(placement, 'cityblock'))
            #print("squareform(pdist(placement))", squareform(pdist(placement)))
            # Calculate the Laplacian matrix of the distance graph
            degree_matrix = np.diag(np.sum(dist_matrix, axis=0))
            laplacian_matrix = degree_matrix - dist_matrix

            # Calculate the eigenvectors and eigenvalues of the Laplacian matrix
            eigenvalues, eigenvectors = eigh(laplacian_matrix)

            # Find the k-means clustering of the eigenvectors with k=2
            kmeans = KMeans(n_clusters=2, random_state=0).fit(eigenvectors[:,1:3])

            # Calculate the symmetry score as the percentage of cells in the smaller cluster
            cluster_sizes = np.bincount(kmeans.labels_)
            symmetry_score = np.min(cluster_sizes) / placement.shape[0]
            spctList.append(symmetry_score)
            #print("Symmetry score of the placement:", symmetry_score)
        self.features['spct1'] = spctList[0]
        self.features['spct2'] = spctList[1]
        self.features['spct'] = sum(spctList)


    def generate_encoding(self, sp1):
        #HVZ-Seuence
        #result = [ [int(sq.name[1:])] for sq in sp1.pos_sequence ] + [ [int(sq.name[1:])] for sq in sp1.neg_sequence ] + [ [int(sq.name[1:])] for sq in sp1.z_sequence ]

        #H-Sequence
        result = [ [int(sq.name[1:])] for sq in sp1.pos_sequence ]
        return np.array(result)

        #AvgPosHVZ
        result = []
        pos_sequence = [sq.name for sq in sp1.pos_sequence]
        neg_sequence = [sq.name for sq in sp1.neg_sequence]
        z_sequence = [sq.name for sq in sp1.z_sequence]
        #print("####AMDEBUG", pos_sequence, neg_sequence, z_sequence)
        for node, nodeObj in self.nodes.items():
            node_indices = []
            node_indices.append(pos_sequence.index(node))
            node_indices.append(neg_sequence.index(node))
            node_indices.append(z_sequence.index(node))
            max_value = len(pos_sequence)
            #result.append( [np.sum(node_indices)*1] )
            #result.append( node_indices )
        return np.array(result)

    def get_torch_data(self):
        self.create_graph()

        a = self.adjacency_matrix
        x = self.g_object.distance_matrix
        row, col = np.nonzero(a.todense())

        # Step 2: Create edge_index tensor
        edge_index = torch.tensor([row, col], dtype=torch.long)

        data = Data(
            x = torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,  # Edge index
            #y=torch.tensor([y], dtype=torch.float)  # Target (graph-level attribute)
        )

        return data
        
    def create_graph(self):
        self.g_object = NGraph(self)
        self.graph = self.g_object.graph
        self.adjacency_matrix = self.g_object.adjacency_matrix
        self.properties_matrix = self.g_object.properties_matrix

    '''
    def create_dataset_example(self, hvz_sequences):
        test=False
        self.create_graph()
        self.adjacency_matrix = self.g_object.adjacency_matrix
        self.properties_matrix = self.g_object.properties_matrix
        x = np.matrix(self.properties_matrix.toarray()) # for spektral
        if test:
            x = self.properties_matrix = self.generate_encoding(hvz_sequences)
            size = len(x)
            a = self.adjacency_matrix = np.ones(shape=(size, size))
        a = self.adjacency_matrix
        g = Graph( x=x, a=a, e=None,y=[456] )
        self.rcm = list(cuthill_mckee_ordering(self.graph))
        ds = [g]
        design_name = PLconfig_grid.designName
        dsr = CircuitDataset2(design_name, ds, False, path="/home/mansoor4/CircuitAttributePrediction/dataset")
        #dsr = CircuitDataset('RLcase1', ds)
        loader = DisjointLoader(dsr, batch_size=1)
        inputs, target = loader.__next__()
        self.inputs = inputs
    '''
 
    def update_graph(self):
        self.g_object.update_graph()
        self.adjacency_matrix = self.g_object.adjacency_matrix
        self.properties_matrix = self.g_object.properties_matrix
        #x = np.matrix(self.properties_matrix.toarray()) # for spektral
        #a = self.adjacency_matrix
        #g = Graph( x=x, a=a, e=None,y=[456] )
        #ds = [g]
        #design_name = PLconfig_grid.designName
        #dsr = CircuitDataset2(design_name, ds, False, path="/home/mansoor4/CircuitAttributePrediction/dataset")
        #loader = DisjointLoader(dsr, batch_size=1)
        #inputs, target = loader.__next__()
        #self.inputs = inputs
        #self.inputs = ds

    def __str__(self):
        str1="{ \"nodes\" : {\n "
        for key,cell in self.data['nodes'].items():
                str1 += "{}".format(cell)
        str1 = str1[:-2] + "\n},\n\"nets\" : { \n "
        for k,net in self.data['nets'].items():
                str1 += "{}".format(net)
        str1 = str1[:-2] + "\n },\n\"edges\" : {\n "
        for k,edge in self.data['edges'].items():
                str1 += "{}".format(edge)
        str1 = str1[:-2] + "\n} }"
        return str1

class Pin(object):
	def __init__(self, **kwargs):
            self.data = { 'node' : None, 'net' : None, 'direction' : None, 'xOffset' : None, 'yOffset' : None}
            self.data['node'] = kwargs['node']
            self.data['direction'] = kwargs['direction']
            self.data['xOffset'] = kwargs['xOffset']
            self.data['yOffset'] = kwargs['yOffset']
            self.data['net'] = kwargs['net']
		
		
	@property
	def node(self):
		return self.data['node']
	@property
	def net(self):
		return self.data['net']
	@property
	def direction(self):
		return self.data['direction']
	@property
	def xOffset(self):
		return self.data['xOffset']
	@property
	def yOffset(self):
		return self.data['yOffset']

	@node.setter
	def node(self):
		return self.data['node']
	@net.setter
	def net(self):
		return self.data['net']
	@direction.setter
	def direction(self):
		return self.data['direction']
	@xOffset.setter
	def xOffset(self):
		return self.data['xOffset']
	@yOffset.setter
	def yOffset(self):
		return self.data['yOffset']	
 
	def __str__(self):
		return "[\"{}\" , \"{}\" , \"{}\" , {} , {} ],".format(self.net, self.node, self.direction, self.xOffset, self.yOffset)

class Net(object):
	def __init__(self, **kwargs):
		self.data = { 'name' : None, 'pins' : [], 'nodes' : [] ,'degree' : None}
		self.data['name'] = kwargs['name']
		self.data['degree'] = kwargs['degree']
		self.data['pins'] = kwargs['pins'] if 'pins' in kwargs else []
		self.data['nodes'] = kwargs['nodes'] if 'nodes' in kwargs else []
		
	@property
	def name(self):
		return self.data['name']
	@property
	def pins(self):
		return self.data['pins']
	@property
	def nodes(self):
		return self.data['pins']
	@property
	def degree(self):
		return self.data['degree']

	@name.setter
	def name(self, v1):
		self.data['name']=v1
	@pins.setter
	def pins(self, v1):
		self.data['pins']=v1
	@degree.setter
	def degree(self, v1):
		self.data['degree']=v1				

	def addpin(self,v1):
		self.data['pins'].append(v1)
		self.addNode(v1.node)
		
	def addNode(self,v1):
		self.data['nodes'].append(v1)

	def __str__(self):
		str1=""
		for pin in self.pins:
			str1+= "{}".format(pin)
		return "\"{}\" : [ {}, {{ \"pins\" : [{}] }} ] ,\n".format(self.name, self.degree, str1[:-1])

			
class Edge(object):
	def __init__(self, **kwargs):
		self.data = { 'v1' : None, 'v2': None, 'name' : None, 'netName' : None, 'location' : None} 
		self.data['v1'] = kwargs['v1']
		self.data['v2'] = kwargs['v2']
		self.data['name'] = kwargs['name']
		self.data['netName'] = kwargs['netName']
		self.data['location'] = kwargs['location'] if 'location' in kwargs else None
	@property
	def v1(self):
		return self.data['v1']
	@property
	def v2(self):
		return self.data['v2']
	@property
	def name(self):
		return self.data['name']
	@property
	def netName(self):
		return self.data['netName']
	@property
	def location(self):
		return self.data['location']

	@v1.setter
	def v1(self, vx):
		self.data['v1']=vx
	@v2.setter
	def v2(self, vx):
		self.data['v2']=vx
	@name.setter
	def name(self, vx):
		self.data['name']=vx
	@netName.setter
	def netName(self, vx):
		self.data['netName']=vx
	@location.setter
	def location(self, vx):
		self.data['location']=vx
		
	def __str__(self):
		return "\"{}\" : [ \"{}\", \"{}\", \"{}\", \"{}\" ],\n".format(self.name, self.v1.name, self.v2.name,  self.netName, self.location)

class Node(object):
    def __init__(self, **kwargs):
        self.data = { 'name' : None, 'width' : None, 'height' : None, 'hierarchy' : None, 'movable' : None, 'terminalType' : None, 'pins' : [], 'point_lb' : Point(0,0)}
        self.data['name'] = kwargs['name'] 
        self.data['width'] = kwargs['width']
        self.data['height'] = kwargs['height']
        self.data['movable'] = kwargs['movable'] if 'movable' in kwargs else True
        self.data['terminalType'] = kwargs['terminalType'] if 'terminalType' in kwargs else None
        self.data['hierarchy'] = kwargs['hierarchy'] if 'hierarchy' in kwargs else None
        self.data['pins'] = kwargs['pins'] if 'pins' in kwargs else []
        self.data['point_lb'] = kwargs['point_lb'] if 'point_lb' in kwargs else Point(0,0)
        self.bbox = None
        self.color = 'blue'
        self.cluster = None #AM 3/9/2024 to support hierarchical clustering
        self.update_bbox()

    @property
    def name(self):
            return self.data['name']
    @property
    def width(self):
            return self.data['width']
    @property
    def height(self):
            return self.data['height']
    @property
    def movable(self):
            return self.data['movable']
    @property
    def terminalType(self):
            return self.data['terminalType']
    @property
    def hierarchy(self):
            return self.data['hierarchy']
    @property
    def pins(self):
            return self.data['pins']
    @property
    def point_lb(self):
            return self.data['point_lb']

    @name.setter
    def name(self, v1):
            self.data['name']=v1
    @width.setter
    def width(self, v1):
            self.data['width']=v1
    @height.setter
    def height(self, v1):
            self.data['height']=v1
    @movable.setter
    def movable(self, v1):
            self.data['movable']=v1
    @terminalType.setter
    def terminalType(self,v1):
            self.data['terminalType']=v1
    @hierarchy.setter
    def hierarchy(self, v1):
            self.data['hierarchy']=v1
    @pins.setter
    def pins(self, v1):
            self.data['pins']=v1
    @point_lb.setter
    def point_lb(self, v1):
        self.data['point_lb']=v1
        self.update_bbox()

    @property
    def isterminal(self):
        if self.terminalType:
                return False
        else:
                return True

    def addpin(self, v1):
        self.data['pins'].append(v1)

    def update_bbox(self):
        if isinstance(self.point_lb, PlaneLocation) or isinstance(self.point_lb, Point):
                ub = PlaneLocation(float(self.point_lb.x + self.width-1), float(self.point_lb.y + self.height/PLconfig_grid.single_cell_height-1), 0.0)
                self.bbox = BBox(lb=self.point_lb, ub=ub, name=self.name)
        elif isinstance(self.point_lb, GridLocation):
                ub = GridLocation(int(self.point_lb.xgrid + self.width-1), int(self.point_lb.ygrid + self.height/PLconfig_grid.single_cell_height-1), 0)
                self.bbox = BBox(lb=self.point_lb, ub=ub, name=self.name)
        elif isinstance(self.point_lb, BinLocation):
                #assume self.point_lb is Grid Location 
                #Add check
                ub = BinLocation(int(self.point_lb.bin_number), int(self.point_lb.yrow + self.height/PLconfig_grid.single_cell_height), int(self.point_lb.xcolumn + self.width), 0)
                self.bbox = BBox(lb=self.point_lb, ub=ub, name=self.name)
        elif isinstance(self.point_lb, ThreeDLocation):
                #assume input is Bin Location
                #Add check
                ub = ThreeDLocation(int(self.point_lb.bin_number), int(self.point_lb.yrow + self.height/PLconfig_grid.single_cell_height), int(self.point_lb.xcolumn + self.width), self.point_lb.zlayer)
                self.bbox = BBox(lb=self.point_lb, ub=ub, name=self.name)
        else:
                raise TypeError("Node point_lb is not valid data type ", type(self.point_lb))

    def __str__(self):
        str1=""
        for pin in self.pins:
                str1 += "{}".format(pin)
        str2="[ "
        for item in self.hierarchy:
                str2 += "\"" + item + "\","
        str2 = str2[:-1] + "]" 
        return "\"{}\" : [{}, {}, {}, {}, \"{}\", {}, {{ \"pins\" : [ {} ] }}],\n".format(self.name, self.point_lb, self.width, self.height, int(self.movable), self.terminalType, str2, str1[:-1])
