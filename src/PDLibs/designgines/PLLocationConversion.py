from designgines.PLGeometricLib import Point
from designgines.PLGridSpec import Grid, BinnedGrid, ThreeDBinnedGrid
import runConfigs.PLconfig_grid #defines global variables

class PlaneLocation(object):
    """Location of layout objects in real x,y,z plane"""
    def __init__(self, x, y, z):
        self.data = {'x': None, 'y': None, 'z': None}
        self.data['x'] = x
        self.data['y'] = y
        self.data['z'] = z
        self.check()

    @property
    def x(self):
        return self.data['x']
    @property
    def y(self):
        return self.data['y']
    @property
    def z(self):
        return self.data['z']

    @x.setter
    def x(self, v1):
        self.data['x'] = v1
        self.check()
    @y.setter
    def y(self, v1):
        self.data['y'] = v1
        self.check()
    @z.setter
    def z(self, v1):
        self.data['z'] = v1
        self.check()

    def check(self):
        if not isinstance(self.x, float):
            raise TypeError("Plane location x value is not float")
        if not isinstance(self.y, float):
            raise TypeError("Plane location y value is not float")
        if not isinstance(self.z, float):
            raise TypeError("Plane location z value is not float")

    def __str__(self):
        return "x={:,}, y={:,}, z={:,}".format(self.x, self.y, self.z)

class GridLocation(object):
    """Location of layout objects in terms of UCLA grid hortizontal sites and vertical rows"""
    def __init__(self, xgrid, ygrid, zgrid):
        self.data = {'xgrid': None, 'ygrid': None, 'zgrid': None}
        self.data['xgrid'] = xgrid
        self.data['ygrid'] = ygrid
        self.data['zgrid'] = zgrid
        self.check()

    @property
    def xgrid(self):
        return self.data['xgrid']
    @property
    def ygrid(self):
        return self.data['ygrid']
    @property
    def zgrid(self):
        return self.data['zgrid']

    @xgrid.setter
    def xgrid(self, v1):
        self.data['xgrid'] = v1
        self.check()
    @ygrid.setter
    def ygrid(self, v1):
        self.data['ygrid'] = v1
        self.check()
    @zgrid.setter
    def zgrid(self, v1):
        self.data['zgrid'] = v1
        self.check()

    def check(self):
        if not isinstance(runConfigs.PLconfig_grid.grid_definition, Grid):
            print('grid_definitino is not of type grid')
            raise TypeError("PLconfig_grid.grid_definition is not type Grid")
        if not isinstance(self.xgrid, int):
            print('xgrid location is not type int')
            raise TypeError("Grid location xgrid value is not int")
        if not isinstance(self.ygrid, int):
            print('ygrid location is not type int')
            raise TypeError("Grid location ygrid value is not int")
        if not isinstance(self.zgrid, int):
            print('zgrid location is not type int')
            raise TypeError("zgrid location zgrid value is not int")
        if self.xgrid < 0 or self.xgrid > runConfigs.PLconfig_grid.grid_definition.avg_sites_per_row:
            print('number of columns=',runConfigs.PLconfig_grid.grid_definition.avg_sites_per_row)
            raise ValueError("Grid location xgrid value is not valid. Found", self.xgrid, "Expected less than ", runConfigs.PLconfig_grid.grid_definition.avg_sites_per_row)
        if self.ygrid < 0 or self.ygrid > runConfigs.PLconfig_grid.grid_definition.numRows:
            print('numRows=',runConfigs.PLconfig_grid.grid_definition.numRows)
            #raise ValueError("Grid location ygrid value is not valid. Found", self.ygrid, "Expected less than ", runConfigs.PLconfig_grid.grid_definition.numRows)
        if not self.zgrid in runConfigs.PLconfig_grid.layer_values:
            pass
            #raise ValueError("Grid location zgrid value is not valid. Found", self.zgrid, "Expected one of ", PLconfig_grid.layer_values)

    def __str__(self):
        return "xgrid={:,}, ygrid={:,}, zgrid={:,}".format(self.xgrid, self.ygrid, self.zgrid)


class BinLocation(object):
    """Location of layout objects in terms of Binned grid dimensions"""
    def __init__(self, bin_number, yrow, xcolumn, zlayer):
        self.data = {'bin_number': None, 'yrow': None, 'xcolumn': None, 'zlayer': None}
        self.data['bin_number'] = bin_number
        self.data['yrow'] = yrow
        self.data['xcolumn'] = xcolumn
        self.data['zlayer'] = zlayer
        self.check()

    @property
    def bin_number(self):
        return self.data['bin_number']
    @property
    def yrow(self):
        return self.data['yrow']
    @property
    def xcolumn(self):
        return self.data['xcolumn']
    @property
    def zlayer(self):
        return self.data['zlayer']

    @bin_number.setter
    def bin_number(self, v1):
        self.data['bin_number'] = v1
        self.check()
    @yrow.setter
    def yrow(self, v1):
        self.data['yrow'] = v1
        self.check()
    @xcolumn.setter
    def xcolumn(self, v1):
        self.data['xcolumn'] = v1
        self.check()
    @zlayer.setter
    def zlayer(self, v1):
        self.data['zlayer'] = v1
        self.check()

    def check(self):
        if not isinstance(runConfigs.PLconfig_grid.binned_grid_definition, BinnedGrid):
            raise TypeError("Bin location grid_definition value is not BinnedGrid")
        if not isinstance(self.bin_number, int):
            raise TypeError("Bin location zbin value is not int")
        if not isinstance(self.yrow, int):
            raise TypeError("Bin location ycolumn value is not int")
        if not isinstance(self.xcolumn, int):
            raise TypeError("Bin location xrow value is not int")
        if not isinstance(self.zlayer, int):
            raise TypeError("Bin location zlayer value is not int")
        if self.bin_number >= runConfigs.PLconfig_grid.binned_grid_definition.numBins:
            raise ValueError("Bin location bin_number value is not valid Found", self.bin_number, "Expected less than or equal to ", runConfigs.PLconfig_grid.binned_grid_definition.numBins)
        rowOBj = runConfigs.PLconfig_grid.binned_grid_definition.rows[0]
        if self.yrow < 0 and self.yrow > rowOBj.height+1:
            raise ValueError("Bin location ycolumn value is not valid. Found", self.yrow, "Expected less than ", rowOBj.height+1)
        if self.xcolumn < 0 and self.xcolumn > rowOBj.width+1:
            raise ValueError("Bin location xrow value is not valid. Found", self.xcolumn, "Expected less than ", rowOBj.width+1)
        if not self.zlayer in runConfigs.PLconfig_grid.layer_values:
            raise ValueError("Bin location zlayer value is not valid. Found", self.zlayer, "Expected one of ", runConfigs.PLconfig_grid.layer_values)

    def __str__(self):
        return "bin_number={:,}, yrow={:,}, xcolumn={:,}, zgrid={:,}".format(self.bin_number, self.yrow, self.xcolumn,  self.zlayer)

class ThreeDLocation(object):
    """Location of 3D layout objects in terms of Binned grid dimensions"""
    def __init__(self, bin_number, yrow, xcolumn, zlayer):
        self.data = {'bin_number': None, 'yrow': None, 'xcolumn': None, 'zlayer': None}
        self.data['bin_number'] = bin_number
        self.data['yrow'] = yrow
        self.data['xcolumn'] = xcolumn
        self.data['zlayer'] = zlayer
        self.check()

    @property
    def bin_number(self):
        return self.data['bin_number']
    @property
    def yrow(self):
        return self.data['yrow']
    @property
    def xcolumn(self):
        return self.data['xcolumn']
    @property
    def zlayer(self):
        return self.data['zlayer']

    @bin_number.setter
    def bin_number(self, v1):
        self.data['bin_number'] = v1
        self.check()
    @yrow.setter
    def yrow(self, v1):
        self.data['yrow'] = v1
        self.check()
    @xcolumn.setter
    def xcolumn(self, v1):
        self.data['xcolumn'] = v1
        self.check()
    @zlayer.setter
    def zlayer(self, v1):
        self.data['zlayer'] = v1
        self.check()

    def check(self):
        if not isinstance(runConfigs.PLconfig_grid.threeD_binned_grid_definition, BinnedGrid):
            raise TypeError("Bin location grid_definition value is not BinnedGrid")
        if not isinstance(self.bin_number, int):
            raise TypeError("Bin location zbin value is not int")
        if not isinstance(self.yrow, int):
            raise TypeError("Bin location ycolumn value is not int")
        if not isinstance(self.xcolumn, int):
            raise TypeError("Bin location xrow value is not int")
        if not isinstance(self.zlayer, int):
            raise TypeError("Bin location zlayer value is not int")
        if self.bin_number >= runConfigs.PLconfig_grid.threeD_binned_grid_definition.numBins:
            raise ValueError("Bin location bin_number value is not valid Found", self.bin_number, "Expected less than or equal to ", runConfigs.PLconfig_grid.threeD_binned_grid_definition.numBins)
        rowOBj = runConfigs.PLconfig_grid.threeD_binned_grid_definition.rows[0]
        if self.yrow < 0 and self.yrow > rowOBj.height+1:
            raise ValueError("Bin location ycolumn value is not valid. Found", self.yrow, "Expected less than ", rowOBj.height+1)
        if self.xcolumn < 0 and self.xcolumn > rowOBj.width+1:
            raise ValueError("Bin location xrow value is not valid. Found", self.xcolumn, "Expected less than ", rowOBj.width+1)
        if not self.zlayer in runConfigs.PLconfig_grid.layer_values:
            raise ValueError("Bin location zlayer value is not valid. Found", self.zlayer, "Expected one of ", runConfigs.PLconfig_grid.layer_values)

    def __str__(self):
        return "bin_number={:,}, yrow={:,}, xcolumn={:,}, zgrid={:,}".format(self.bin_number, self.yrow, self.xcolumn,  self.zlayer)

class Grid2Plane(object):
    def __init__(self, grid_location):
        self.grid_location = grid_location
        self.plane_location = None
        self.update_plane_location()

    def update_plane_location(self):
        if not isinstance(self.grid_location, GridLocation):
            raise TypeError("Grid2Plan input is not of type GridLocation, %s", self.grid_location)
        if not isinstance(runConfigs.PLconfig_grid.grid_definition, Grid):
            raise TypeError("PLconfig_grid.grid_definition is not of type Grid")
        num_rows = runConfigs.PLconfig_grid.grid_definition.numRows
        num_sites = runConfigs.PLconfig_grid.grid_definition.rows[0].numSites
        if self.grid_location.ygrid < 0 and self.grid_location.ygrid > num_rows:
            raise TypeError("Grid_location ygrid is not valid. Found {} and expected less than or equal to {}".format(self.grid_location.ygrid, num_rows))
        if self.grid_location.xgrid < 0 and self.grid_location.xgrid > num_sites:
            raise TypeError("Grid_location xgrid is not valid. Found {} and expected less than or equal to {}".format(self.grid_location.xgrid, num_sites))
        if self.grid_location.ygrid == num_rows:
            row_object = runConfigs.PLconfig_grid.grid_definition.rows[self.grid_location.ygrid-1]
            y = float(row_object.coordinate+row_object.height)
            x = float(row_object.subRowOrigin + row_object.width*self.grid_location.xgrid)
        else:
            row_object = runConfigs.PLconfig_grid.grid_definition.rows[self.grid_location.ygrid]
            y = float(row_object.coordinate)
            x = float(row_object.subRowOrigin + row_object.width*self.grid_location.xgrid)

        #AM TODO: z transformation to be defined
        z=0.0
        self.plane_location = PlaneLocation(x,y,z)

    def __str__(self):
        return "input(grid_location) => {} and output(plane_location)={}".format(self.grid_location, self.plane_location)

class Plane2Grid(object):
    def __init__(self, plane_location):
        self.plane_location = plane_location
        self.grid_location = None
        self.update_grid_location()

    def update_grid_location(self):
        if not isinstance(self.plane_location, PlaneLocation):
            raise TypeError("Plane2Grid input is not of type PlaneLocation, %s", self.plane_location)
        if not isinstance(runConfigs.PLconfig_grid.grid_definition, Grid):
            raise TypeError("PLconfig_grid.grid_definition is not of type Grid")
        #AM TODO: replace the logic by finding actual row object based on the range of coordiantes
        row_object = runConfigs.PLconfig_grid.grid_definition.rows[0]
        ygrid = int((self.plane_location.y - row_object.coordinate) / runConfigs.PLconfig_grid.single_cell_height)
        xgrid = int(self.plane_location.x - row_object.subRowOrigin)
        #AM TODO: z transformation to be defined
        zgrid=0
        self.grid_location = GridLocation(xgrid, ygrid, zgrid)

    def __str__(self):
        return "input(plane_location) => {} and output(grid_location)={}".format(self.plane_location, self.grid_location)

class Bin2Plane(object):
    def __init__(self, bin_location):
        self.bin_location = bin_location
        self.plane_location = None
        self.update_plane_location()

    def update_plane_location(self):
        if not isinstance(runConfigs.PLconfig_grid.binned_grid_definition, BinnedGrid):
            raise TypeError("PLconfig_grid.binned_grid_definition is not of type BinnedGrid")
        if not isinstance(runConfigs.PLconfig_grid.grid_definition, Grid):
            raise TypeError("PLconfig_grid.grid_definition is not of type Grid")
        b2g = Bin2Grid(self.bin_location)
        g2p = Grid2Plane(b2g.grid_location)
        self.plane_location = g2p.plane_location

    def __str__(self):
        return "input(bin_location) => {} and output(plane_location)={}".format(self.bin_location, self.plane_location)

class Plane2Bin(object):
    def __init__(self, plane_location):
        self.bin_location = None
        self.plane_location = plane_location
        self.update_bin_location()

    def update_bin_location(self):
        if not isinstance(runConfigs.PLconfig_grid.binned_grid_definition, BinnedGrid):
            raise TypeError("PLconfig_grid.binned_grid_definition is not of type BinnedGrid")
        if not isinstance(runConfigs.PLconfig_grid.grid_definition, Grid):
            raise TypeError("PLconfig_grid.grid_definition is not of type Grid")
        p2g = Plane2Grid(self.plane_location)
        g2b = Grid2Bin(p2g.grid_location)
        self.bin_location = g2b.bin_location

    def __str__(self):
        return "input(plane_location) => {} and output(bin_location)={}".format(self.plane_location, self.bin_location)

class Bin2Grid(object):
    def __init__(self, bin_location):
        self.bin_location = bin_location
        self.grid_location = None
        self.update_grid_location()

    def update_grid_location(self):
        if not isinstance(self.bin_location, BinLocation):
            raise TypeError("Plane2Grid input is not of type PlaneLocation, %s", self.bin_location)
        if not isinstance(runConfigs.PLconfig_grid.binned_grid_definition, BinnedGrid):
            raise TypeError("PLconfig_grid.binned_grid_definition is not of type BinnedGrid")
        if not isinstance(runConfigs.PLconfig_grid.grid_definition, Grid):
            raise TypeError("PLconfig_grid.grid_definition is not of type Grid")
        bbox = runConfigs.PLconfig_grid.binned_grid_definition.squares.squares[self.bin_location.bin_number]
        xgrid = bbox.lb.x + self.bin_location.xcolumn
        ygrid = bbox.lb.y + self.bin_location.yrow
        #AM TODO: z transformation to be defined
        zgrid=0
        self.grid_location = GridLocation(xgrid,ygrid,zgrid)

    def __str__(self):
        return "input(bin_location) => {} and output(grid_location)={}".format(self.bin_location, self.grid_location)

class Grid2Bin(object):
    def __init__(self, grid_location):
        self.grid_location = grid_location
        self.bin_location = None
        self.update_bin_location()

    def update_bin_location(self):
        if not isinstance(self.grid_location, GridLocation):
            raise TypeError("Grid2Bin input is not of type GridLocation, %s", self.grid_location)
        if not isinstance(runConfigs.PLconfig_grid.binned_grid_definition, BinnedGrid):
            raise TypeError("PLconfig_grid.binned_grid_definition is not of type BinnedGrid")
        if not isinstance(runConfigs.PLconfig_grid.grid_definition, Grid):
            raise TypeError("PLconfig_grid.grid_definition is not of type Grid")
        b1 = runConfigs.PLconfig_grid.binned_grid_definition.squares.findBin(Point(self.grid_location.xgrid, self.grid_location.ygrid))
        print(runConfigs.PLconfig_grid.binned_grid_definition.squares)
        print(runConfigs.PLconfig_grid.binned_grid_definition)
        c1=self.grid_location.xgrid%runConfigs.PLconfig_grid.binned_grid_definition.divide_factor_x
        r1=self.grid_location.ygrid%runConfigs.PLconfig_grid.binned_grid_definition.divide_factor_y
        #AM TODO: z transformation to be defined
        zgrid=0
        self.bin_location = BinLocation(b1, r1, c1 , zgrid)

    def __str__(self):
        return "input(grid_location) => {} and output(bin_location)={}".format(self.grid_location, self.bin_location)

class Bin2ThreeD(object):
    def __init__(self, bin_location):
        self.bin_location = bin_location
        self.threeD_location = None
        self.update_threeD_location()

    def update_threeD_location(self):
        if not isinstance(self.bin_location, BinLocation):
            raise TypeError("Bin2ThreeD input is not of type BinLocation, %s", self.bin_location)
        if not isinstance(runConfigs.PLconfig_grid.threeD_binned_grid_definition, ThreeDBinnedGrid):
            raise TypeError("PLconfig_grid.trheeDbinned_grid_definition is not of type ThreeDBinnedGrid")
        if not isinstance(runConfigs.PLconfig_grid.binned_grid_definition, BinnedGrid):
            raise TypeError("PLconfig_grid.grid_definition is not of type Grid")
        threeD_bin_number = int(runConfigs.PLconfig_grid.folded_bins_map.twoD_to_threeD(self.bin_location.bin_number))

        self.threeD_location = ThreeDLocation(
            threeD_bin_number, 
            self.bin_location.yrow,
            self.bin_location.xcolumn, 
            self.bin_location.zlayer
        )

    def __str__(self):
        return "input(2dbin_location) =>\n {} and\n output(3dbin_location)=\n{}".format(self.bin_location, self.threeD_location)

class ThreeD2Bin(object):
    def __init__(self, threeD_bin_location):
        self.bin_location = None
        self.threeD_location = threeD_bin_location
        self.update_twoD_location()

    def update_twoD_location(self):
        if not isinstance(self.threeD_location, ThreeDLocation):
            raise TypeError("ThreeD2Bin input is not of type ThreeDLocation, %s", self.threeD_bin_location)
        if not isinstance(runConfigs.PLconfig_grid.threeD_binned_grid_definition, ThreeDBinnedGrid):
            raise TypeError("PLconfig_grid.binned_grid_definition is not of type ThreeDBinnedGrid")
        if not isinstance(runConfigs.PLconfig_grid.binned_grid_definition, BinnedGrid):
            raise TypeError("PLconfig_grid.grid_definition is not of type BinnedGrid")
        twoD_bin_number = runConfigs.PLconfig_grid.folded_bins_map.threeD_to_twoD(self.threeD_location.bin_number)
        self.bin_location = BinLocation(
            twoD_bin_number, 
            self.threeD_location.yrow,
            self.threeD_location.xcolumn, 
            int(self.threeD_location.zlayer)
        )

    def __str__(self):
        return "input(3dbin_location) => {} and output(2dbin_location)={}".format(self.threeD_location, self.bin_location)

class ThreeD2Grid(object):
    def __init__(self, threeD_bin_location):
        """
        The concept of ThreeD2Plan is different from other conversios.
        ThreeDLocation is not equivalent to BinLocation.
        The coordinate system and location in 2D systems are changed.
        The Bin2ThreeD location are created using Folding algorithm. 
        However, the main idea of ThreeD2Plane conversion is Global Bin
        to plane geometric conversion so that sequence pair generation is 
        representative of 3D layout rather than original 2D layout. 
        """
        self.grid_location = None
        self.threeD_location = threeD_bin_location
        self.update_grid_location()

    def update_grid_location(self):
        if not isinstance(self.threeD_location, ThreeDLocation):
            raise TypeError("ThreeD2Bin input is not of type ThreeDLocation, %s", self.threeD_bin_location)
        if not isinstance(runConfigs.PLconfig_grid.threeD_binned_grid_definition, ThreeDBinnedGrid):
            raise TypeError("PLconfig_grid.binned_grid_definition is not of type ThreeDBinnedGrid")
        if not isinstance(runConfigs.PLconfig_grid.binned_grid_definition, BinnedGrid):
            raise TypeError("PLconfig_grid.grid_definition is not of type BinnedGrid")
        bbox = runConfigs.PLconfig_grid.threeD_binned_grid_definition.squares.squares[self.threeD_location.bin_number]
        #print("bbox=", bbox, " ******zgrid=", self.threeD_location.zlayer)
        xgrid = bbox.lb.x + self.threeD_location.xcolumn
        ygrid = bbox.lb.y + self.threeD_location.yrow

        class glocation(object):
            def __init__(self, xgrid, ygrid, zgrid):
                self.xgrid = xgrid
                self.ygrid = ygrid
                self.zgrid = zgrid
            def __str__(self):
                return 'xgrid={},ygrid={},zgrid={}'.format(self.xgrid, self.ygrid, self.zgrid)


        #print('x=',xgrid,'y=',ygrid)
        self.grid_location = glocation(xgrid,ygrid,int(self.threeD_location.zlayer))

    def __str__(self):
        return "input(3dbin_location) => {} and output(grid_location)={}".format(self.threeD_location, self.grid_location)

class ThreeD2Plane(object):
    def __init__(self, threeD_bin_location):
        """
        The concept of ThreeD2Plan is different from other conversios.
        ThreeDLocation is not equivalent to BinLocation.
        The coordinate system and location in 2D systems are changed.
        The Bin2ThreeD location are created using Folding algorithm. 
        However, the main idea of ThreeD2Plane conversion is Global Bin
        to plane geometric conversion so that sequence pair generation is 
        representative of 3D layout rather than original 2D layout. 
        """
        self.plane_location = None
        self.threeD_location = threeD_bin_location
        self.update_plane_location()

    def update_plane_location(self):
        if not isinstance(self.threeD_location, ThreeDLocation):
            raise TypeError("ThreeD2Bin input is not of type ThreeDLocation, %s", self.threeD_bin_location)
        if not isinstance(runConfigs.PLconfig_grid.threeD_binned_grid_definition, ThreeDBinnedGrid):
            raise TypeError("PLconfig_grid.binned_grid_definition is not of type ThreeDBinnedGrid")
        if not isinstance(runConfigs.PLconfig_grid.binned_grid_definition, BinnedGrid):
            raise TypeError("PLconfig_grid.grid_definition is not of type BinnedGrid")
        
        mapd = runConfigs.PLconfig_grid.folded_bins_map
        layer, perlayer_bin_number = mapd.global_to_perlayer_bin(
            self.threeD_location.bin_number
        )
        #print(f"perlayer_bin_number={perlayer_bin_number}")
        #bbox = runConfigs.PLconfig_grid.threeD_binned_grid_definition.squares.squares[perlayer_bin_number]
        bbox = runConfigs.PLconfig_grid.threeD_binned_grid_definition.squares.squares[self.threeD_location.bin_number]
        #print(bbox)
        xgrid = bbox.lb.x + self.threeD_location.xcolumn
        #xgrid = bbox.lb.x + perlayer_bin_number
        ygrid = bbox.lb.y + self.threeD_location.yrow
        z = float(layer)
        #z = 0.0
        num_rows = runConfigs.PLconfig_grid.grid_definition.numRows
        #print('numRows=',num_rows,'xgrid=', xgrid,'ygrid=', ygrid)
        #origin = runConfigs.PLconfig_grid.grid_definition.rows[0]
        if ygrid == num_rows:
            row_object = runConfigs.PLconfig_grid.grid_definition.rows[ygrid-1]
            y = float(row_object.coordinate+row_object.height)
            x = float(row_object.subRowOrigin + row_object.width*xgrid)
        else:
            row_object = runConfigs.PLconfig_grid.grid_definition.rows[ygrid]
            y = float(row_object.coordinate)
            x = float(row_object.subRowOrigin + row_object.width*xgrid)
        #threed_to_bin = ThreeD2Bin(self.threeD_location)
        #bin_location = threed_to_grid.bin_location
        #print('threeD_bin_number=', self.threeD_location.bin_number, ' ,twoD_bin_number=', twoD_bin_number, ' ,perlayer_bin_number=', perlayer_bin_number)
        #print('x=',xgrid,'y=',ygrid)
        self.plane_location = PlaneLocation(x,y,z)

    def __str__(self):
        return "input(3dbin_location) => {} and output(plane_location)={}".format(self.threeD_location, self.plane_location)

class Plane2ThreeD(object):
    def __init__(self, plane_location):
        """
        The concept of ThreeD2Plan is different from other conversios.
        ThreeDLocation is not equivalent to BinLocation.
        The coordinate system and location in 2D systems are changed.
        The Bin2ThreeD location are created using Folding algorithm. 
        However, the main idea of ThreeD2Plane conversion is Global Bin
        to plane geometric conversion so that sequence pair generation is 
        representative of 3D layout rather than original 2D layout. 
        """
        self.plane_location = plane_location
        self.threeD_location = None
        self.update_threed_bin_location()

    def update_threed_bin_location(self):
        if not isinstance(self.plane_location, PlaneLocation):
            raise TypeError("PlaneLocation input is not of type PlaneLocation, %s", self.plane_location)
        if not isinstance(runConfigs.PLconfig_grid.threeD_binned_grid_definition, ThreeDBinnedGrid):
            raise TypeError("PLconfig_grid.threeD_binned_grid_definition is not of type ThreeDBinnedGrid")
        row_object = runConfigs.PLconfig_grid.threeD_binned_grid_definition.rows[0]
        number_of_layers = runConfigs.PLconfig_grid.threeD_binned_grid_definition.number_of_layers
        divide_factor = runConfigs.PLconfig_grid.divide_factor

        ygrid = int((self.plane_location.y - row_object.coordinate) / runConfigs.PLconfig_grid.single_cell_height)

        #AM Bugfix 9/9/23 
        #zplane = int(self.plane_location.z) #Folding-based 3D layotus have zplane 0 for cells on upper layer.


        #AM 9/9/23: calculating zplane value based on position. layer 1 cells always be on the right half side (x values greater than points corresponding to row_object.numSites)
        zplane = int(self.plane_location.x - row_object.subRowOrigin) // row_object.numSites
        #print(f"x={self.plane_location.x}, numSites={row_object.numSites}, zplane={zplane}")
        xgrid=None
        if zplane == 1:
            xgrid = int(self.plane_location.x - row_object.subRowOrigin)
            #xgrid -= ( row_object.numSites // number_of_layers ) * divide_factor #AM Bugfix 9/9/2023: This to resolve 3D layouts loading issues due to coordinate errors
        else:
            xgrid = int(self.plane_location.x - row_object.subRowOrigin)
        zgrid=0

        #print(f"xgrid={xgrid}, ygrid={ygrid}")

        b1 = runConfigs.PLconfig_grid.threeD_binned_grid_definition.squares.findBin(Point(xgrid, ygrid))
        c1=xgrid%runConfigs.PLconfig_grid.threeD_binned_grid_definition.divide_factor
        r1=ygrid%runConfigs.PLconfig_grid.threeD_binned_grid_definition.divide_factor
        self.threeD_location = ThreeDLocation(b1, r1, c1 , zgrid)

    def __str__(self):
        return "input(plane_location) => {} and output(threeD_location)={}".format(self.plane_location, self.threeD_location)

def test_plane_location_conversions():
    location_list = []
    location_list.append(PlaneLocation(18.0, 18.0, 0.0))
    location_list.append(PlaneLocation(25.0, 91.0, 0.0))
    location_list.append(PlaneLocation(23.0, 36.0, 0.0))

    for plane_location in location_list:
        p2g = Plane2Grid(plane_location)
        print(p2g)
        
        
        g2b = Grid2Bin(p2g.grid_location)
        print(g2b)
        
        b2g = Bin2Grid(g2b.bin_location)
        print(b2g)
        
        g2p = Grid2Plane(b2g.grid_location)
        print(g2p)

def test_grid_location_conversion():
    bin_location_list = []
    bin_location_list.append(GridLocation(xgrid=4, ygrid=0, zgrid=0))
    for bin_location in bin_location_list:
        b2p = Grid2ThreeD(bin_location)
        print(grid_location,'=>converted to 3D=>',b2p)

def test_bin_location_conversion():
    bin_location_list = []
    bin_location_list.append(BinLocation(bin_number=2, xcolumn=1, yrow=1, zlayer=0))
    for bin_location in bin_location_list:
        b2p = Bin2Plane(bin_location)
        b2threed = Bin2ThreeD(bin_location)
        print(b2threed)


def main():
    """
    parser = argparse.ArgumentParser()    
    
    parser.add_argument("-name", action="store", dest="designName",
                        help="design name",
                        required=True, type=str)    
    
    parser.add_argument("-dir", action="store", dest="inputDir",
                        help="folder of UCLA design files",
                        required=True, type=str)    
    
    
    args = parser.parse_args()
    
    inputDir = os.path.abspath(args.inputDir)
    designName = args.designName
    """
    #test_plane_location_conversions()
    test_bin_location_conversion()
    #test_grid_location_conversion()

if __name__ == "__main__":
    main()
