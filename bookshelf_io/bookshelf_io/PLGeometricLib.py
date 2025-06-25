class Point(object):
	def __init__(self, x=None, y=None):
		self._x = x
		self._y = y
	
	@property
	def x(self):
		return self._x
	@property
	def y(self):
		return self._y

	@x.setter
	def x(self, v1):
		self._x = v1
	@y.setter
	def y(self, v1):
		self._y = v1

	def value(self, v1 = None, v2 = None):
		if v1 and v2: 
			self._x = v1
			self._y = v2
		return (self._x, self._y)
	

	def __str__(self):
		return "x={}, y={}".format(self.x, self.y)

class BBox():
	def __init__(self, **kwargs):
		self.data = { 'lb' : Point(), 'ub' : Point()}
		self.data['lb']=kwargs['lb'] if 'lb' in kwargs else Point()
		self.data['ub']=kwargs['ub'] if 'ub' in kwargs else Point()
		self.name = kwargs['name'] if 'name' in kwargs else ''

	@property
	def lb(self):
		return self.data['lb']
	@property
	def ub(self):
		return self.data['ub']

	@lb.setter	
	def lb(self,v1):
		self.data['lb']=v1
	@ub.setter	
	def ub(self,v1):
		self.data['ub']=v1

	def contains(self,v1):
		xl=self.lb.x
		yl=self.lb.y
		xh=self.ub.x
		yh=self.ub.y
		ix= v1.x
		iy=v1.y
		if ix >= xl and ix < xh:
			if iy >= yl and iy < yh:
				return True
		return False 
	
	def length_x(self):
		return self.ub.x - self.lb.x

	def length_y(self):
		return self.ub.y - self.lb.y

	def scale_up(self, multiple):
		self.ub.x = self.ub.x*multiple
		self.ub.y = self.ub.y*multiple
		self.lb.x = self.lb.x*multiple
		self.lb.y = self.lb.y*multiple

	def ub_right(self):
		return Point(self.lb.x, self.ub.y)

	def lb_left(self):
		return Point(self.ub.x, self.lb.y)

	def __str__(self):
		return "{},{}".format(self.lb, self.ub)

class Square(BBox):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)

class Squares(object):
    def __init__(self, **kwargs):
        self.data = { 'squares' : {}, 'index' : 0}
        self.data['squares']=kwargs['squares'] if 'squares' in kwargs else {}
        self.data['index']=kwargs['index'] if 'index' in kwargs else 0
        self.avg_sites_per_row = None
        self.number_of_layers = None
        self.divide_factor = None

    @property
    def squares(self):
        return self.data['squares']

    @property
    def index(self):
        return self.data['index']

    @squares.setter	
    def squares(self,v1):
        self.data['squares']=v1

    @index.setter	
    def index(self,v1):
        self.data['index']=v1

    def addSquare(self,v1):
        self.squares[self.index]=v1
        self.index+=1

    def addPolygon(self,v1):
        self.squares[self.index]=v1
        self.index+=1

    def add_unit_square(self, point, name=''):
        self.addSquare(Square(lb=Point(point.x, point.y), ub=Point(point.x+1, point.y+1), name=name))

    def findBin(self,p1):
        for k,v in self.squares.items():
            check = v.contains(p1)
            if check:
                return k
        raise ValueError("Couldn't find bin for point=%s with squares\n%s" % (p1, self))

    def __str__(self):
        str1=""
        for k,v in self.squares.items():
            str1+= "{}, {}\n".format(k, v)
            return str1
	
class Polygons(Squares):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
