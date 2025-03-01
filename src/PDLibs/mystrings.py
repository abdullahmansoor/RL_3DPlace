import re

class string(object):
	def __init__(self, str1):
		self._str = str1
		self._flag = None
		self._list = []
	
	def compare(self, other):
		flag = re.IGNORECASE
		p = re.compile(other._str, flag)
		ans = p.search(self._str)
		
		# print (pattern, flag, ans)
		
		if ans is not None:
			self._flag, self._list = (True, ans.groups())
		else:
			self._flag, self._list = (False, None)
		return self._flag, self._list
	
	def __eq__(self, other):
		flag = re.IGNORECASE
		p = re.compile(other._str, flag)
		ans = p.search(self._str)
		
		# print (pattern, flag, ans)
		
		if ans is not None:
			self._flag, self._list = (True, ans.groups())
			return True
		else:
			self._flag, self._list = (False, None)
			return False
		return self._flag, self._list
	def __str__(self):
		return "{}".format(self._str)
	
