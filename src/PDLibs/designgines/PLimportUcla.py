import time
import json
import sys
import os

from designgines.PLucla import Netlist
from designgines.PLucla import Node
from designgines.PLucla import Net
from designgines.PLucla import Pin
from designgines.PLGeometricLib import Point
from designgines.PLGridSpec import Grid
from designgines.PLGridSpec import Row
from designgines.PLLocationConversion import PlaneLocation, BinLocation, Bin2Plane
from designgines.PLLocationConversion import GridLocation, Grid2Plane
from designgines.PLLocationConversion import ThreeDLocation, ThreeD2Plane
from NGraph.PLNGraph import NGraph

from mystrings import string


class importUcla(object):
    def __init__(self, **kwargs):
        self.data = {'path' : None, 'name' : None,'netlist' : Netlist(), 'gridDefinition' : Grid()}
        self.data['name'] = kwargs['name']
        self.data['path'] = kwargs['path']
        self.data['netlist'] = kwargs['netlist'] if 'netlist' in kwargs else Netlist()
        self.inputPlacementFile = kwargs['inputPlacementFile'] if 'inputPlacementFile' in kwargs else None
        self.data['gridDefinition'] = kwargs['gridDefinition'] if 'gridDefinition' in kwargs else Grid()
        self.d1 = self.data['netlist']
        self.g1 = self.data['gridDefinition']
        self.numNodes = None
        self.numTerminals = None
        self.numNets = None
        self.numPins = None
        self.numRows = None
        self.total_sites = None
        self.avg_sites_per_row = None
        print("***************Reading Nodes file***************")
        self.readNodefile(self.data['path'] / f"{self.data['name']}.nodes")
        print("***************Reading Nets file***************")
        self.readNetsFile(self.data['path'] / f"{self.data['name']}.nets")
        print("***************Reading Placement file***************")
        if self.inputPlacementFile:
            self.readPlFile(self.inputPlacementFile)
        else:
            self.readPlFile(self.data['path'] / f"{self.data['name']}.pl")
        print("***************Reading Grid definition file***************")
        self.readSclFile(self.data['path'] / f"{self.data['name']}.scl")
        #self.data['netlist'].create_graph()
        #Graph initialization moved to PLLayout updateParameters function
        self.updateNodesWithoutTerminals()

    @property
    def name(self):
            return self.data['name']
    @property
    def path(self):
            return self.data['path']
    @property
    def netlist(self):
            return self.data['netlist']
    @property
    def gridDefinition(self):
            return self.data['gridDefinition']

    @name.setter
    def name(self,v1):
            self.data['name']=v1
    @path.setter
    def path(self,v1):
            self.data['path']=v1
    @netlist.setter
    def netlist(self,v1):
            self.data['netlist']=v1
    @gridDefinition.setter
    def gridDefinition(self,v1):
            self.data['gridDefinition']=v1


    def updateNodesWithoutTerminals(self):
        count = 0
        for node, nodeObj in self.d1.nodes.items():
            if nodeObj.movable: count += 1
        self.d1.numNodesWOTerminals = count

        
    def parseNodeName(self,name):
        list1=name.split('/')
        index1= len(list1)-2
        hier1= []
        cellName=list1[index1+1]
        if index1 == 0:
            hier1.append(list1[0])
        else:
            hier1 = list1[:index1]
        return cellName, hier1

    def readNodefile(self,filename):
        fnode=open(filename, "r")
        numNodes=0
        numTerminals=0

        for line in fnode:
            s1=string(line)
            flag1, list1 = s1.compare(string(r'UCLA nodes 1.0'))
            flag2, list2 = s1.compare(string(r'^NumNodes\s+:\s+(\d+)'))
            flag3, list3 = s1.compare(string(r'^NumTerminals\s+:\s+(\d+)'))
            flag4, list4 = s1.compare(string(r'^\s+(\S+)\s+(\d+)\s+(\d+)'))
            flag5, list5 = s1.compare(string(r'^\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)'))
            flag12, list12 = s1.compare(string(r'^#'))

            if flag1:
                pass
                #print(s1)
            elif flag2:
                numNodes=int(list2[0])
                self.d1.numNodes=numNodes
                #print("NumNodes={}".format(numNodes))
            elif flag3:
                numTerminals=int(list3[0])
                self.d1.numTerminals=numTerminals
                #print("NumTerminals={}".format(numTerminals))
            elif flag5:
                nodeName, width, height, terminalType =list5[0],int(list5[1]),int(list5[2]),list5[3]
                cellName, hierarchy=self.parseNodeName(nodeName)
                self.d1.addNode(Node(name=cellName, width=width, height=height, movable=False, terminalType= terminalType, hierarchy=hierarchy))
            elif flag4:
                nodeName, width, height =list4[0],int(list4[1]),int(list4[2])
                cellName, hierarchy=self.parseNodeName(nodeName)
                self.d1.addNode(Node(name=cellName, width=width, height=height, hierarchy=hierarchy))
            elif flag12:
                pass
            else:
                print(s1)

        self.numNodes = numNodes
        self.numTerminals = numTerminals

    def readNetsFile(self,filename):
        fnets=open(filename, "r")
        numNets=0
        numPins=0
        N1=None
        for line in fnets:
            s1=string(line)
            flag1, list1 = s1.compare(string(r'UCLA nets 1.0'))
            #flag12, list12 = s1.compare(string(r'^#'))
            if flag1:
                    pass
                    #print(s1)
            flag2, list1 = s1.compare(string(r'^NumNets\s+:\s+(\d+)'))
            if flag2:
                    numNets=int(list1[0])
                    self.d1.numNets=numNets
                    #print("NumNets={}".format(numNets))
            flag3, list1 = s1.compare(string(r'^NumPins\s+:\s+(\d+)'))
            if flag3:
                    numPins=int(list1[0])
                    self.d1.numPins=numPins
                    #print("NumPins={}".format(numPins))
            flag4, list1 = s1.compare(string(r'^NetDegree\s+:\s+(\d+)\s+([\d\w_/]+)'))
            if flag4:
                    degree, netName =int(list1[0]), list1[1]
                    if N1:
                            self.d1.addNet(N1)
                            #print("done with net={}".format(N1.name))
                    N1 = Net(name=netName, degree=degree)
            flag5, list1 = s1.compare(string(r'^\s+([\d\w/]+)\s+(\w)\s+:\s+([\d.-]+)\s+([\d.-]+)'))
            if flag5:
                    nodeName, pinDirection, pinXOffset, pinYOffset =list1[0],list1[1],float(list1[2]),float(list1[3])
                    vertex, hierarchy=self.parseNodeName(nodeName)
                    N1.addpin(Pin(node=vertex, net=netName, direction= pinDirection, xOffset = pinXOffset, yOffset = pinYOffset))
        self.d1.addNet(N1)
        self.numNets = numNets
        self.numPins = numPins

    def readPlFile(self,filename):
        fnode=open(filename, "r")
        for line in fnode:
            s1=string(line)
            flag1, list1 = s1.compare(string(r'UCLA pl 1.0'))
            flag4, list4 = s1.compare(string(r'^\s+(\S+)\s+(\S+)\s+(\S+)\s+:\s+(\w)'))
            flag4a, list4a = s1.compare(string(r'^\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+:\s+(\w)'))

            flag5, list5 = s1.compare(string(r'^\s+(\S+)\s+(\S+)\s+(\S+)\s+:\s+(\w)\s+(\S+)'))

            flag6, list6 = s1.compare(string(r'^\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+:\s+(\w)\s+([/\w]+)'))

            flag7, list7 = s1.compare(string(r'^(\S+)\s+(\S+)\s+(\S+)\s+:\s+(\S+)'))
            flag12, list12 = s1.compare(string(r'^#'))


            if flag1:
                pass
                #print(s1)
            elif flag6:
                nodeName, x, y, z, dir1, mtype =list5[0],float(list5[1]),float(list5[2]),float(list5[3]),list5[4],list5[5]
                cellName, hierarchy=self.parseNodeName(nodeName)
                self.d1.nodes[cellName].point_lb=PlaneLocation(x, y, z)

            elif flag5:
                nodeName, x, y, dir1, mtype =list5[0],float(list5[1]),float(list5[2]),list5[3],list5[4]
                cellName, hierarchy=self.parseNodeName(nodeName)
                self.d1.nodes[cellName].point_lb=PlaneLocation(x, y, 0.0)
            elif flag7:
                nodeName, x, y, dir1 =list7[0],float(list7[1]),float(list7[2]),list7[3]
                cellName, hierarchy=self.parseNodeName(nodeName)
                #if cellName not in self.d1.nodes.keys(): continue
                self.d1.nodes[cellName].point_lb=PlaneLocation(x, y, 0.0)
            elif flag4a:
                nodeName, x, y, z, dir1 =list4a[0],float(list4a[1]),float(list4a[2]),float(list4a[3]),list4a[4]
                cellName, hierarchy=self.parseNodeName(nodeName)
                self.d1.nodes[cellName].point_lb=PlaneLocation(x, y, z)
            elif flag4:
                nodeName, x, y, dir1 =list4[0],float(list4[1]),float(list4[2]),list4[3]
                cellName, hierarchy=self.parseNodeName(nodeName)
                self.d1.nodes[cellName].point_lb=PlaneLocation(x, y, 0.0)
            elif flag12:
                pass
            else:
                #pass
                print(s1)

    def read_bin_plFile(self,filename):
        fnode=open(filename, "r")
        for line in fnode:
            s1=string(line)
            flag5, list5 = s1.compare(string(r'^\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$'))
            flag6, list6 = s1.compare(string(r'^\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$'))
            flag12, list12 = s1.compare(string(r'^#'))

            if flag6:
                    nodeName, bin1, ycol, xrow, zlayer = list6[0],int(list6[1]),int(list6[2]),int(list6[3]), int(list6[4])
                    cellName, hierarchy=self.parseNodeName(nodeName)
                    bin_location = BinLocation(bin1, ycol, xrow, zlayer)
                    plane_location = Bin2Plane(bin_location)
                    #print(nodeName, bin1, ycol, xrow)
                    #print(plane_location)
                    self.d1.nodes[cellName].point_lb=plane_location.plane_location
            elif flag5:
                    nodeName, bin1, ycol, xrow = list5[0],int(list5[1]),int(list5[2]),int(list5[3])
                    cellName, hierarchy=self.parseNodeName(nodeName)
                    bin_location = BinLocation(bin1, ycol, xrow, 0)
                    plane_location = Bin2Plane(bin_location)
                    #print(nodeName, bin1, ycol, xrow)
                    #print(plane_location)
                    self.d1.nodes[cellName].point_lb=plane_location.plane_location
            elif flag12:
                    pass
            else:
                    print(s1)

    def readSclFile(self,filename):
        self.g1.readSclFile(filename)
        self.numRows = self.g1.numRows
        self.total_sites = self.g1.total_sites
        self.avg_sites_per_row = self.g1.avg_sites_per_row

    def __str__(self):
        msg = ''
        msg += "numLayers={}\n".format(self.g1.number_of_layers)
        msg += "numNodes={}\n".format(self.numNodes)
        msg += "numTerminals={}\n".format(self.numTerminals)
        msg += "numNets={}\n".format(self.numNets)
        msg += "numPins={}\n".format(self.numPins)
        msg += "numRows={}\n".format(self.numRows)
        msg += "avg_sites_per_row={}\n".format(self.avg_sites_per_row)
        return msg

class exportUcla(object):
    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.path = kwargs['path']
        self.netlist = kwargs['netlist'] if 'netlist' in kwargs else Netlist()
        self.gridDefinition = kwargs['gridDefinition'] if 'gridDefinition' in kwargs else Grid()
        self.dimensions = kwargs['dimensions'] if 'dimensions' in kwargs else 'plane_location'
        self.numNodes = None
        self.numTerminals = None
        self.numNets = None
        self.numPins = None
        self.numRows = None
        self.total_sites = None
        self.avg_sites_per_row = None
        self.writeNodefile(self.path+"/"+self.name+".nodes")
        self.writeNetsFile(self.path+"/"+self.name+".nets")
        self.writePlFile(self.path+"/"+self.name+".pl")
        self.writeSclFile(self.path+"/"+self.name+".scl")

    def writeNodefile(self, filename):
        pass

    def writeNetsFile(self, filename):
        pass

    def writePlFile(self, filename):
        fh = open(filename, 'w')
        for node, nodeObj in self.netlist.nodes.items():
            if self.dimensions == 'bin_location':
                if isinstance(nodeObj.point_lb, BinLocation):
                    b1 = nodeObj.point_lb.bin_number
                    r1 = nodeObj.point_lb.yrow
                    c1 = nodeObj.point_lb.xcolumn
                    name = nodeObj.name
                    line = "\t" + name + "\t" + str(b1) + "\t" + str(r1) + "\t" + str(c1) + '\t:\tN\n'
                else:
                    raise ValueError("Write BinLocation Placement file is not supported. Exiting!")
                fh.write(line)
            elif self.dimensions == 'grid_location':
                line = ''
                if isinstance(nodeObj.point_lb, GridLocation):
                    y = nodeObj.point_lb.ygrid
                    x = nodeObj.point_lb.xgrid
                    name = nodeObj.name
                    line = "\t" + name + "\t" + str(x) + "\t" + str(y) + '\t:\tN\n'
                else:
                    raise ValueError('Write GridLocation Placement file is not supported. Exiting!')
                fh.write(line)
            elif self.dimensions == 'plane_location':
                if isinstance(nodeObj.point_lb, PlaneLocation):
                    y = nodeObj.point_lb.y
                    x = nodeObj.point_lb.x
                    z = nodeObj.point_lb.z
                    name = nodeObj.name
                    line = "\t" + name + "\t" + str(x) + "\t" + str(y)  + "\t" + str(z) + '\t:\tN\n'
                elif isinstance(nodeObj.point_lb, GridLocation):
                    g2p_location =  Grid2Plane(nodeObj.point_lb)
                    plane_location = g2p_location.plane_location
                    y = plane_location.y
                    x = plane_location.x
                    z = plane_location.z
                    name = nodeObj.name
                    line = "\t" + name + "\t" + str(x) + "\t" + str(y)  + "\t" + str(z) + '\t:\tN\n'
                elif isinstance(nodeObj.point_lb, BinLocation):
                    b2p_location  = Bin2Plane(nodeObj.point_lb)
                    plane_location = b2p_location.plane_location
                    y = plane_location.y
                    x = plane_location.x
                    z = plane_location.z
                    name = nodeObj.name
                    line = "\t" + name + "\t" + str(x) + "\t" + str(y)  + "\t" + str(z) + '\t:\tN\n'
                elif isinstance(nodeObj.point_lb, ThreeDLocation):
                    b2p_location  = ThreeD2Plane(nodeObj.point_lb)
                    plane_location = b2p_location.plane_location
                    y = plane_location.y
                    x = plane_location.x
                    z = plane_location.z
                    name = nodeObj.name
                    line = "\t" + name + "\t" + str(x) + "\t" + str(y)  + "\t" + str(z) + '\t:\tN\n'
                else:
                    raise ValueError("Write PlaneLocation Placement file is not supported. Exiting!")
                fh.write(line)
            else:
                raise ValueError('dimensions=%s is not supported. Exiting!' % self.dimensions)
        fh.close()

    def writeNodefile(self, filename):
            pass

    def writeSclFile(self, filename):
        fh = open(filename, 'w')
        line = "NumRows 	:	{}\n".format(self.gridDefinition.numRows)
        fh.write(line)
        for row in self.gridDefinition.rows.values():
            line = "CoreRow Horizontal\n"
            line += "	Coordinate	:	{}\n".format(row.coordinate)
            line += "	Height	:	{}\n".format(row.height)
            line += "	Sitewidth	:	{}\n".format(row.width)
            line += "	Sitespacing	:	{}\n".format(row.spacing)
            line += "	Siteorient	:	{}\n".format(row.orient)
            line += "	Sitesymmetry	:	{}\n".format(row.symmetry)
            line += "	SubrowOrigin	:	{}	NumSites	:	{}\n".format(row.subRowOrigin, row.numSites)
            line += "End\n"
            fh.write(line)
        fh.close()

class Sim_to_ucla(object):
    def __init__(self, **kwargs):
        self.data = {'path' : None, 'name' : None,'netlist' : Netlist(), 'gridDefinition' : Grid()}
        self.data['name'] = kwargs['name']
        self.data['path'] = kwargs['path']
        self.input_file = kwargs['input_file']
        self.data['netlist'] = kwargs['netlist'] if 'netlist' in kwargs else Netlist()
        self.data['gridDefinition'] = kwargs['gridDefinition'] if 'gridDefinition' in kwargs else Grid()
        self.g1 = self.data['gridDefinition']
        self.numNodes = None
        self.numTerminals = None
        self.numNets = None
        self.numPins = None
        self.numRows = None
        self.total_sites = None
        self.avg_sites_per_row = None
        self.cell_height = None
        self.subRowOrigin = None
        print("***************Creating Nodes file***************")
        self.create_nodefile(self.data['path']+"/"+self.data['name']+".nodes", self.input_file)
        print("\n\n***************Creating Nets file***************")
        self.create_netsFile(self.data['path']+"/"+self.data['name']+".nets", self.input_file)
        print("\n\n***************Creating  Grid definition file***************")
        self.create_sclFile(self.data['path']+"/"+self.data['name']+".scl", self.input_file)
        print("\n\n***************Creating Placement file***************")
        self.create_plFile(self.data['path']+"/"+self.data['name']+".pl", self.input_file)

    @property
    def name(self):
            return self.data['name']
    @property
    def path(self):
            return self.data['path']
    @property
    def netlist(self):
            return self.data['netlist']
    @property
    def gridDefinition(self):
            return self.data['gridDefinition']

    @name.setter
    def name(self,v1):
            self.data['name']=v1
    @path.setter
    def path(self,v1):
            self.data['path']=v1
    @netlist.setter
    def netlist(self,v1):
            self.data['netlist']=v1
    @gridDefinition.setter
    def gridDefinition(self,v1):
            self.data['gridDefinition']=v1

    def parseNodeName(self,name):
        list1=name.split('/')
        index1= len(list1)-2
        hier1= []
        cellName=list1[index1+1]
        if index1 == 0:
                hier1.append(list1[0])
        else:
                hier1 = list1[:index1]
        return cellName, hier1

    def create_nodefile(self, output_filename, input_filename):
        fspice = open(input_filename, 'r')
        fnode=open(output_filename, "w")
        num_nodes=0
        num_terminals=0

        header = 'UCLA nodes 1.0\n'
        header += '# Created  :  Mar 14 2020\n'
        header += '# User     :  Mansoor, Abdullah,  (abdullahmansoor@gmail.com)\n'
        fnode.write(header)

        body = ''
        for line in fspice:
                s1=string(line)
                #X20 d0 d1 s2 o0l2 mux2
                flag4, list4 = s1.compare(string(r'^(X\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)'))
                if flag4:
                        num_nodes += 1
                        node_name = list4[0]
                        width = 1
                        height = 9
                        cellName = node_name
                        hierarchy= 'top'
                        body += "    {}    {}    {}\n".format(cellName, width, height)
                        self.netlist.addNode(Node(name=cellName, width=width, height=height, hierarchy=hierarchy))
                else:
                        flag5, list5 = s1.compare(string(r'^\*interface\s+(\S+)\s+orientacao=(\S+)\s+(\S+)$'))
                        if (not flag5):
                                if not line:
                                        print(s1)
        num_nodes = "NumNodes : {}\n".format(num_nodes)
        num_terminals = "NumTerminals : {}\n".format(num_terminals)

        fnode.write(num_nodes)
        fnode.write(num_terminals)
        fnode.write(body)

        self.numNodes = num_nodes
        self.numTerminals = num_terminals
        fnode.close()

    def create_netsFile(self, output_filename, input_filename):
        fnets=open(output_filename, "w")
        N1=None
        nets = set()
        fspice=open(input_filename, "r")
        for line in fspice.readlines():
                s1=string(line)
                flag4, list4 = s1.compare(string(r'^(X\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)'))
                if flag4:
                        nets.add(list4[1])
                        nets.add(list4[2])
                        nets.add(list4[3])
                        nets.add(list4[4])
                else:
                        flag4, list4 = s1.compare(string(r'^\*interface\s+(\S+)\s+orientacao=(\S+)\s+(\S+)'))
                        if not flag4:
                                if not line:
                                        print(s1)
        nets = sorted(list(nets))
        pins = {}
        for net in nets:
                pins[net] = []
                N1 = Net(name=net, degree=0)
                fspice=open(input_filename, "r")
                for line in fspice.readlines():
                        s1=string(line)
                        flag4, list4 = s1.compare(string(r'^(X\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$'))
                        if flag4:
                                if net in list4:
                                        nodeName = list4[0]
                                        pins[net].append(nodeName)
                                        pinDirection = 'I'
                                        pinXOffset = 0
                                        pinYOffset = 0
                                        N1.addpin(Pin(node=nodeName, net=net, direction= pinDirection, xOffset = pinXOffset, yOffset = pinYOffset))
                        else:
                                flag5, list5 = s1.compare(string(r'^\*interface\s+(\S+)\s+orientacao=(\S+)\s+(\S+)$'))
                                if (not flag5):
                                        if not line:
                                                print(s1)
                N1.degree = len(pins[net])
                self.netlist.addNet(N1)
        #print(self.netlist)
        numNets=len(nets)
        self.netlist.numNets=numNets
        numPins=0
        self.netlist.numPins=numPins
        self.numNets = numNets
        self.numPins = numPins
        header = 'UCLA nets 1.0\n'
        header += '# Created  :  Mar 14 2020\n'
        header += '# User     :  Mansoor, Abdullah,  (abdullahmansoor@gmail.com)\n'
        header += "NumNets : {}\n".format(numNets)
        header += "NumPins : {}\n".format(numPins)
        fnets.write(header)
        body = ''
        for k,v in self.netlist.nets.items():
                degree = v.degree
                net_name = v.name
                body += "NetDegree : {} {}\n".format(degree, net_name)
                for pin in v.pins:
                    body += "    {} {} : {} {}\n".format(pin.node, pin.direction, pin.xOffset, pin.yOffset)
        fnets.write(body)
        fnets.close()

    def create_sclFile(self, output_filename, input_filename):
        self.g1.create_sclFile(
            output_filename,
            runConfigs.PLconfig_grid.import_num_rows,
            runConfigs.PLconfig_grid.import_num_sites
        )
        self.numRows = self.g1.numRows
        self.total_sites = self.g1.total_sites
        self.avg_sites_per_row = self.g1.avg_sites_per_row
        row_item = self.g1.rows[0]
        self.cell_height = row_item.height
        self.subRowOrigin = row_item.subRowOrigin

    def create_plFile(self, output_filename, input_filename):
        fnode=open(output_filename, "w")
        body = ''
        site_count = 0
        row_count = 0
        for k, v in self.netlist.nodes.items():
                cellName = v.name
                x = float(self.subRowOrigin + (site_count % self.avg_sites_per_row))
                y = self.subRowOrigin + float(int(site_count/self.avg_sites_per_row)*self.cell_height)
                #movable = None
                #terminalType = None
                v.point_lb=PlaneLocation(x, y, 0.0)
                body += "    {}    {}    {}    :    {}\n".format(cellName, x, y, 'N')
                site_count += 1
        header = 'UCLA pl 1.0\n'
        header += '# Created  :  Mar 14 2020\n'
        header += '# User     :  Mansoor, Abdullah,  (abdullahmansoor@gmail.com)\n'
        fnode.write(header)
        fnode.write(body)
        fnode.close()

    def __str__(self):
        msg = ''
        msg += "numNodes={}\n".format(self.numNodes)
        msg += "numTerminals={}\n".format(self.numTerminals)
        msg += "numNets={}\n".format(self.numNets)
        msg += "numPins={}\n".format(self.numPins)
        msg += "numRows={}\n".format(self.numRows)
        msg += "avg_sites_per_row={}\n".format(self.avg_sites_per_row)
        return msg

def create_graph_ucla(inputDir, designName):
    i1= importUcla(name=designName, path=inputDir)

    graph_file = inputDir + "/" + designName + ".grf"

    i1.netlist.dump_graph(graph_file)

def sim_to_pldm(inputDir, designName, input_file):
    i1= Sim_to_ucla(name=designName, path=inputDir, input_file=input_file)
    #i1= importUcla(name=designName, path=inputDir)
    print(i1)
	
def ucla_to_pldm(inputDir, designName):
    i1= importUcla(name=designName, path=inputDir)
    items = i1.netlist.twl(1432, 1, 1)
    print(items[0])
    g1 = NGraph(i1.netlist)

def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-name", action="store", dest="designName",
                                            help="design name",
                                            required=True, type=str)

    parser.add_argument("-dir", action="store", dest="inputDir",
                                            help="folder of UCLA design files",
                                            required=True, type=str)

    parser.add_argument("-spice", action="store", dest="spice_file",
                                            help="name of spice file",
                                            required=False, type=str)

    parser.add_argument("-graph", action="store_true", help="check if dump graph",
                                            required=False, default=False)

    args = parser.parse_args()

    inputDir = os.path.abspath(args.inputDir)
    designName = args.designName

    spice_file = None
    if args.spice_file:
            spice_file = os.path.abspath(inputDir + "/" + args.spice_file)
            sim_to_pldm(inputDir, designName, spice_file)
    else:
            ucla_to_pldm(inputDir, designName)

    if args.graph:
            create_graph_ucla(inputDir, designName)


if __name__ == "__main__":
	main()
