import copy
from designgines.PLGeometricLib import Point, BBox, Polygons
from mystrings import string

class Row(object):
	def __init__(self, **kwargs):
            self.data = { "coordinate" : None, "height" : None, "width" : None, "spacing" : None, "orient" : None, "symmetry" : None, "subRowOrigin" : None, "numSites" : None }
            self.data['coordinate'] = kwargs['coordinate'] # y value
            self.data['height'] = kwargs['height']
            self.data['width'] = kwargs['width']
            self.data['spacing'] = kwargs['spacing']
            self.data['orient'] = kwargs['orient']
            self.data['symmetry'] = kwargs['symmetry']
            self.data['subRowOrigin'] = kwargs['subRowOrigin'] # x value
            self.data['numSites'] = kwargs['numSites']

	@property
	def coordinate(self):
		return self.data['coordinate']
	@property
	def height(self):
		return self.data['height']
	@property
	def width(self):
		return self.data['width']
	@property
	def spacing(self):
		return self.data['spacing']
	@property
	def orient(self):
		return self.data['orient']
	@property
	def symmetry(self):
		return self.data['symmetry']
	@property
	def subRowOrigin(self):
		return self.data['subRowOrigin']
	@property
	def numSites(self):
		return self.data['numSites']

	@coordinate.setter
	def coordinate(self,v1):
		self.data['coordinate'] = v1
	@height.setter
	def height(self, v1):
		 self.data['height'] = v1
	@width.setter
	def width(self, v1):
		 self.data['width'] = v1
	@spacing.setter
	def spacing(self, v1):
		 self.data['spacing'] = v1
	@orient.setter
	def orient(self, v1):
		 self.data['orient'] = v1
	@symmetry.setter
	def symmetry(self, v1):
		 self.data['symmetry'] = v1
	@subRowOrigin.setter
	def subRowOrigin(self, v1):
		 self.data['subRowOrigin'] =v1 
	@numSites.setter
	def numSites(self, v1):
		 self.data['numSites'] = v1
	
	def lowerBound(self):
		return Point(self.subRowOrigin, self.coordinate)

	def upperBound(self):
		return Point(self.subRowOrigin+(self.numSites*self.width), self.coordinate+self.height)
		
	def __str__(self):
            return "[ {}, {}, {}, {}, {}, {}, {}, {} ],".format(self.coordinate, self.height, self.width, self.spacing, self.orient, self.symmetry, self.subRowOrigin, self.numSites)

class Grid(object):
    def __init__(self, **kwargs):
        self.data = { "numRows" : None, "rows" : {}, "numSites" : None } 
        self.data['numRows'] = kwargs['numRows'] if 'numRows' in kwargs else None
        self.data['rows'] = kwargs['rows'] if 'rows' in kwargs else {}
        self.data['avg_sites_per_row'] = kwargs['avg_sites_per_row'] if 'avg_sites_per_row' in kwargs else None
        self.data['total_sites'] = kwargs['total_sites'] if 'total_sites' in kwargs else 0
        self.numBins=kwargs['numBins'] if 'numBins' in kwargs else None
        self.number_of_layers = kwargs['number_of_layers'] if 'number_of_layers' in kwargs else 1

        ###AM Bugfix 9/9/23: 
        #self.divide_factor = 1  #removed hardcoding of divide_factor. This was causing error in ThreedBinnedGrid grid definition due to inheritance. I was advised not to use inheritance for better code readability
        self.divide_factor=kwargs['divide_factor'] if 'divide_factor' in kwargs else 1

    @property
    def numRows(self):
            return self.data['numRows']
    @property
    def rows(self):
            return self.data['rows']
    @property
    def avg_sites_per_row(self):
            return self.data['avg_sites_per_row']
    @property
    def total_sites(self):
            return self.data['total_sites']

    @numRows.setter
    def numRows(self, v1):
            self.data['numRows'] = v1
    @rows.setter
    def rows(self, v1):
            self.data['rows'] = v1
    @avg_sites_per_row.setter
    def avg_sites_per_row(self, v1):
            self.data['avg_sites_per_row'] = v1
    @total_sites.setter
    def total_sites(self, v1):
            self.data['total_sites'] = v1

    def addRow(self, v1, v2):
            self.data['rows'][v1]= v2

    def make_even_dimensions(self):
        flag_avg_sites_per_row = False
        if self.numRows % 2 == 1:
            new = self.rows[self.numRows] = copy.deepcopy(self.rows[self.numRows-1])
            self.numRows = self.numRows + 1
            new.coordinate = new.coordinate+new.height
        total_sites = 0
        for row_key, row_obj in self.rows.items():
            if row_obj.numSites % 2 == 1:
                flag_avg_sites_per_row = True
                row_obj.numSites = row_obj.numSites + 1
                total_sites += row_obj.numSites
        if flag_avg_sites_per_row:
            self.total_sites = total_sites
            self.avg_sites_per_row = int (self.total_sites / self.numRows)

    def readSclFile(self,filename):
        fscl=open(filename, "r")
        numRows=0
        count=0
        flag=None
        R1=None
        sites = set()
        for line in fscl:
            s1=string(line)
            flag1, list1 = s1.compare(string(r'UCLA scl 1.0'))
            flag2, list2 = s1.compare(string(r'^NumRows\s+:\s+(\d+)'))
            flag2b, list2b = s1.compare(string(r'^NumLayers\s+:\s+(\d+)'))
            flag3, list3 = s1.compare(string(r'^CoreRow\s+Horizontal'))
            flag4, list4 = s1.compare(string(r'^\s+Coordinate\s+:\s+(\d+)'))
            flag5, list5 = s1.compare(string(r'^\s+Height\s+:\s+(\d+)'))
            flag6, list6 = s1.compare(string(r'^\s+Sitewidth\s+:\s+(\d+)'))
            flag7, list7 = s1.compare(string(r'^\s+SiteSpacing\s+:\s+(\d+)'))
            flag8, list8 = s1.compare(string(r'^\s+SiteOrient\s+:\s+(\w)'))
            flag9, list9 = s1.compare(string(r'^\s+Sitesymmetry\s+:\s+(\w)'))
            flag10, list10 = s1.compare(string(r'^\s+SubrowOrigin\s+:\s+(\d+)\s+NumSites\s+:\s+([\d]+)'))
            flag11, list11 = s1.compare(string(r'^End'))
            flag12, list12 = s1.compare(string(r'^#'))
            if flag1:
                    pass
                    #print(s1)
            elif flag3:
                    if flag:
                        R1= Row(coordinate=coordinate, height= height, width= sitewidth, spacing = sitespacing, orient= siteorient, symmetry = sitesymmetry, subRowOrigin = subroworigin, numSites = numsites )
                        self.addRow(count, R1)
                        count += 1
                    flag=1
            elif flag2:
                    numRows=int(list2[0])
                    self.numRows=numRows
                    #print("NumRows={}".format(numRows))
            elif flag2b:
                    self.number_of_layers=int(list2b[0])
                    #print("Number of Layers={}".format(self.number_of_layers))
            elif flag4:
                    coordinate =int(list4[0])
            elif flag5:
                    height =int(list5[0])
            elif flag6:
                    sitewidth =int(list6[0])
            elif flag7:
                    sitespacing =int(list7[0])
            elif flag8:
                    siteorient = list8[0]
            elif flag9:
                    sitesymmetry =list9[0]
            elif flag10:
                    subroworigin, numsites = int(list10[0]), int(list10[1])
                    sites.add(numsites)
                    self.total_sites += numsites
            elif flag11:
                    pass
            elif flag12:
                    pass
            else:
                    print(s1)
        R1= Row(coordinate=coordinate, height= height, width= sitewidth, spacing = sitespacing, orient= siteorient, symmetry = sitesymmetry, subRowOrigin = subroworigin, numSites = numsites )
        self.addRow(count, R1)
        self.numRows = numRows
        self.avg_sites_per_row = int(self.total_sites / self.numRows)
        self.numBins = self.numRows * self.avg_sites_per_row
        if not len(sites) == 1:
                raise ValueError("Grid spec is not regular. Not supported")

    def create_sclFile(self,filename, number_of_rows, number_of_sites):
        fscl=open(filename, "w")

        body = ''
        for row_i in range(number_of_rows):
            coordinate = 18 + 9 * row_i # y value
            height = 9
            sitewidth = 1
            sitespacing = 1
            siteorient = 'N'
            sitesymmetry = 'Y'
            subroworigin = 18  # x value
            numsites = number_of_sites
            self.total_sites += numsites
            R1= Row(coordinate=coordinate, height= height, width= sitewidth, spacing = sitespacing, orient= siteorient, symmetry = sitesymmetry, subRowOrigin = subroworigin, numSites = numsites )
            self.addRow(row_i, R1)
            body += "CoreRow Horizontal\n"
            body += "\tCoordinate\t:\t{}\n".format(coordinate)
            body += "\tHeight\t:\t{}\n".format(height)
            body += "\tSitewidth\t:\t{}\n".format(sitewidth)
            body += "\tSiteSpacing\t:\t{}\n".format(sitespacing)
            body += "\tSiteOrient\t:\t{}\n".format(siteorient)
            body += "\tSitesymmetry\t:\t{}\n".format(sitesymmetry)
            body += "\tSubrowOrigin\t:\t{}\tNumSites\t:\t{}\n".format(subroworigin, numsites)
            body += "End\n"
        header = "UCLA scl 1.0\n"
        header += "NumRows : {}\n".format(number_of_rows)
        header += "NumLayers : {}\n".format(self.number_of_layers)
        fscl.write(header)
        fscl.write(body)
        fscl.close()
        self.numRows = number_of_rows
        self.avg_sites_per_row = int(self.total_sites / self.numRows)


    def convert_to_3d(self):
        for row in self.rows.values():
            row.numSites = row.numSites*self.number_of_layers
        self.avg_sites_per_row = self.avg_sites_per_row * self.number_of_layers
        self.total_sites = self.total_sites*self.number_of_layers

    def __str__(self):
            str1 = "[ {{ numRows : {} }},\n ".format(self.numRows)
            str1 += "[ {{ numLayers : {} }},\n ".format(self.number_of_layers)
            str1 += "[ {{ avg_sites_per_row : {} }},\n ".format(self.avg_sites_per_row)
            for k, row in self.rows.items():
                    str1+= "{{  {} : {} }},\n".format(k, row)
            str1 = str1[:-2] + "]"
            return str1

class BinnedGrid(Grid):
    def __init__(self, **kwargs):
        self.divide_factor=kwargs['divide_factor'] if 'divide_factor' in kwargs else None
        self.squares=kwargs['squares'] if 'squares' in kwargs else Polygons()
        self.divide_factor_y=kwargs['divide_factor_y'] if 'divide_factor_y' in kwargs else 4
        self.divide_factor_x=kwargs['divide_factor_x'] if 'divide_factor_x' in kwargs else 1
        super().__init__(**kwargs)
        self.updateSquares()
        number_of_squares = len(self.squares.squares.keys())
        if number_of_squares != self.numBins:
            print('# of squares = %s ' % number_of_squares)
            print('# of bins = %s ' % self.numBins)
            raise ValueError(
                    'The number of bins do not match number of squares'
            )

    @staticmethod
    def is_regular(original_grid, divide_factor_x, divide_factor_y):
        flag1 = original_grid.numRows % divide_factor_y == 0
        flag2 = original_grid.avg_sites_per_row % divide_factor_x == 0
        return flag1 and flag2

    @classmethod
    def from_grid(cls, original_grid, divide_factor_x, divide_factor_y):
        origNumRows = original_grid.numRows
        origNumBins = original_grid.numBins
        number_of_layers = original_grid.number_of_layers
        newNumRows= int(origNumRows/divide_factor_y)
        newNumSites = 0
        numSites = None
        newGridDefinition = Grid(numRows = newNumRows )
        new_rows = {}
        for i in range(newNumRows):
                rowObj = original_grid.rows[0]
                """
                BIG ASSUMPTION HERE THAT BINNED LAYOUT IS REGULAR.
                """
                coordinate = rowObj.coordinate
                #need double checking

                #currently bins are created based on width and spacing. which is not rivht.
                #ideally it should be based on x and y divide factors
                #
                #height = rowObj.height*divide_factor_y
                height = divide_factor_y #AMTODO: Not sure if this is right
                coordinate = rowObj.coordinate + i*height
                width = rowObj.width*divide_factor_x
                spacing = rowObj.spacing*divide_factor_x
                orient = rowObj.orient
                symmetry = rowObj.symmetry
                subRowOrigin = rowObj.subRowOrigin
                numSites = int(int(rowObj.numSites)/divide_factor_x)
                newNumSites += numSites
                new_rows[i] = Row(
                        coordinate=coordinate,
                        height=height,
                        width=width,
                        spacing=spacing,
                        orient=orient,
                        symmetry=symmetry,
                        subRowOrigin=subRowOrigin,
                        numSites=numSites
                )
        numBins = newNumRows * numSites
        avg_sites_per_row = numSites
        #print("old bins_count = %s, New bins_count %s" % (origNumBins, numBins))
        return cls(
                numRows=newNumRows,
                total_sites=newNumSites,
                number_of_layers=number_of_layers,
                avg_sites_per_row=avg_sites_per_row,
                numBins=numBins,
                rows=new_rows,
                divide_factor_x=divide_factor_x,
                divide_factor_y=divide_factor_y,
        )

    def updateSquares(self):
        self.squares = Polygons()
        xCuts = []
        yCuts = []
        count=0
        #self.divide_factor = 2 ###AMTODO 9/9/23: Temp hack remove it ASAP
        #the nusmites logic below to be uddated for non-regular cells
        for x in range(self.rows[0].numSites*self.divide_factor_x):
                if x % self.divide_factor_x == 0:
                        xCuts.append(x)
        for y in range(self.numRows*self.divide_factor_y):
                if y % self.divide_factor_y == 0:
                        yCuts.append(y)
        pointlb = Point(0,0)
        pointub = Point(0,0)
        pointnext = Point(0,0)
        #print('xCuts=',xCuts,'\nyCuts', yCuts, '\nbin_size_x', self.divide_factor_x, '\nbin_size_y', self.divide_factor_y)
        for r in yCuts:
            pointlb.x = 0
            for c in xCuts:
                pointub = Point(c+self.divide_factor_x, r+self.divide_factor_y)
                pointnext = Point(pointub.x, pointlb.y)
                bb=BBox(lb=pointlb, ub=pointub)
                #print('bb======',bb)
                self.squares.addSquare(bb)
                pointlb = copy.deepcopy(pointnext)
            pointlb.y = r + self.divide_factor_y
        #print("squares:\n{}".format(self.squares))

    def convert_to_3d(self):
        for row in self.rows.values():
            row.numSites = row.numSites*self.number_of_layers
        self.avg_sites_per_row = self.avg_sites_per_row * self.number_of_layers
        self.total_sites = self.total_sites*self.number_of_layers

class ThreeDBinnedGrid(BinnedGrid):
    def __init__(self, **kwargs):
        self.divide_factor=kwargs['divide_factor']
        super().__init__(**kwargs)
        number_of_squares = len(self.squares.squares.keys())
        if number_of_squares != self.numBins:
            print('# of squares = %s ' % number_of_squares)
            print('# of bins = %s ' % self.numBins)
            raise ValueError('The number of bins do not match number of squares')

    @classmethod
    def from_grid(cls, original_grid, divide_factor):
        '''
        class function to create grid object for 3d binned grid
        from 2d binned grid
        '''
        origNumRows = original_grid.numRows
        origNumBins = original_grid.numBins
        #number_of_layers = int(original_grid.number_of_layers)
        number_of_layers = 2
        newNumRows= int(origNumRows)
        newNumSites = 0
        numSites = None
        newGridDefinition = Grid(numRows = newNumRows )
        new_rows = {}
        for i in range(newNumRows):
            rowObj = original_grid.rows[0]
            """
            BIG ASSUMPTION HERE THAT BINNED LAYOUT IS REGULAR.
            """
            coordinate = rowObj.coordinate
            #need double checking
            height = rowObj.height
            coordinate = rowObj.coordinate + i*height
            width = rowObj.width
            spacing = rowObj.spacing
            orient = rowObj.orient
            symmetry = rowObj.symmetry
            subRowOrigin = rowObj.subRowOrigin
            numSites = int(rowObj.numSites)*number_of_layers
            newNumSites += numSites
            new_rows[i] = Row(coordinate=coordinate,height=height,width=width,spacing=spacing,orient=orient,symmetry=symmetry,subRowOrigin=subRowOrigin,numSites=numSites)
        numBins = newNumRows * numSites
        avg_sites_per_row = numSites
        #print("old bins_count = %s, New bins_count %s" % (origNumBins, numBins))
        return cls(
                numRows=newNumRows,
                total_sites=newNumSites,
                number_of_layers=number_of_layers,
                avg_sites_per_row=avg_sites_per_row,
                numBins=numBins,
                rows=new_rows,
                divide_factor=divide_factor
        )

    @classmethod
    def fromSclFile(cls, filename, divide_factor):
        fscl=open(filename, "r")
        numRows=0
        count=0
        flag=None
        R1=None
        sites = set()
        total_sites=0
        number_of_layers=avg_sites_per_row=numBins=None
        rows={}
        numsites=None
        for line in fscl:
            s1=string(line)
            flag1, list1 = s1.compare(string(r'UCLA scl 1.0'))
            flag2, list2 = s1.compare(string(r'^NumRows\s+:\s+(\d+)'))
            flag2b, list2b = s1.compare(string(r'^NumLayers\s+:\s+(\d+)'))
            flag3, list3 = s1.compare(string(r'^CoreRow\s+Horizontal'))
            flag4, list4 = s1.compare(string(r'^\s+Coordinate\s+:\s+(\d+)'))
            flag5, list5 = s1.compare(string(r'^\s+Height\s+:\s+(\d+)'))
            flag6, list6 = s1.compare(string(r'^\s+Sitewidth\s+:\s+(\d+)'))
            flag7, list7 = s1.compare(string(r'^\s+SiteSpacing\s+:\s+(\d+)'))
            flag8, list8 = s1.compare(string(r'^\s+SiteOrient\s+:\s+(\w)'))
            flag9, list9 = s1.compare(string(r'^\s+Sitesymmetry\s+:\s+(\w)'))
            flag10, list10 = s1.compare(string(r'^\s+SubrowOrigin\s+:\s+(\d+)\s+NumSites\s+:\s+([\d]+)'))
            flag11, list11 = s1.compare(string(r'^End'))
            flag12, list12 = s1.compare(string(r'^#'))
            if flag1:
                    pass
                    #print(s1)
            elif flag3:
                    if flag:
                        R1= Row(
                                coordinate=coordinate,
                                height= height,
                                width= sitewidth,
                                spacing = sitespacing,
                                orient= siteorient,
                                symmetry = sitesymmetry,
                                subRowOrigin = subroworigin,
                                numSites = numsites )
                        rows[count] =  R1
                        count += 1
                    flag=1
            elif flag2:
                    numRows=int(list2[0])//divide_factor
                    #print("NumRows={}".format(numRows))
            elif flag2b:
                    number_of_layers=int(list2b[0])
                    #print("Number of Layers={}".format(self.number_of_layers))
            elif flag4:
                    coordinate =int(list4[0])
            elif flag5:
                    height =int(list5[0])*divide_factor
            elif flag6:
                    sitewidth =int(list6[0])*divide_factor
            elif flag7:
                    sitespacing =int(list7[0])*divide_factor
            elif flag8:
                    siteorient = list8[0]
            elif flag9:
                    sitesymmetry =list9[0]
            elif flag10:
                    subroworigin, numsites = int(list10[0]), int(list10[1])
                    #print(f"number_of_layers={number_of_layers}")
                    #print(f"numsites={numsites}")
                    #print(f"divide_factor={divide_factor}")
                    numsites = (number_of_layers*numsites)//divide_factor
                    avg_sites_per_row = numsites
                    sites.add(numsites)
                    total_sites += numsites
            elif flag11:
                    pass
            elif flag12:
                    pass
            else:
                    print(s1)
        R1 = Row(
                coordinate=coordinate,
                height= height,
                width= sitewidth,
                spacing = sitespacing,
                orient= siteorient,
                symmetry = sitesymmetry,
                subRowOrigin = subroworigin,
                numSites = numsites)
        rows[count] = R1
        avg_sites_per_row = numsites
        numBins = numRows * avg_sites_per_row
        if not len(sites) == 1:
                raise ValueError("Grid spec is not regular. Not supported")
        updatedRows = {}
        for count in range(numRows):
            updatedRows[count] = rows[count]
        return cls(
                numRows=numRows,
                total_sites=total_sites,
                number_of_layers=number_of_layers,
                avg_sites_per_row=avg_sites_per_row,
                numBins=numBins,
                rows=updatedRows,
                divide_factor=divide_factor)

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
	import runConfigs.PLconfig_grid as PLconfig_grid
	g1 = Grid()
	scl_file_path = PLconfig_grid.inputDir + "/" + PLconfig_grid.designName + ".scl"
	g1.readSclFile(scl_file_path)
	print(g1)

if __name__ == "__main__":
    main()
