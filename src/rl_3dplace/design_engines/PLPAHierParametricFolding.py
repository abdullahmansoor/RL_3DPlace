import sys
import numpy as np
import networkx as nx
from scipy import interpolate
import math
from sympy import divisors
import itertools
import math

from design_engines.placement_helpers import change_divide_ratio
import runConfigs.PLconfig_grid as PLconfig_grid


def myround(x):
    '''
    round function maps 0.5 to 0 and >0.5 to 1. 
    this function maps 0.5 >= to 1
    '''

    val = None
    if (float(x) % 1 ) >= 0.5:
        val = math.ceil(x)
    else:
        val = round(x)
    return val


def log_scaled_steps(n, steps=8):
    return [int(round(n ** (i / steps))) for i in range(1, steps)]

def fractional_root_candidates(n, k_start=1.5, k_stop=5.0, step=0.25, min_val=2):
    roots = []
    k = k_start
    while k <= k_stop:
        r = int(round(n ** (1 / k)))
        if r > min_val and r < n:
            roots.append(r)
        k += step
    return sorted(set(roots))

class PAHierParametricFolding(object):
    def __init__(self, numberOfCuts, direction, pattern, windowSizeCode):
        self.numberOfCuts = numberOfCuts
        self.direction = direction
        self.pattern = pattern
        self.windowSizeCode = windowSizeCode
        self.extendedPattern = self.ExtendPattern()


        #update metricsMap for rectangular windwo sizes
        self.bin_size_x = None
        self.bin_size_y = None
        self.metricsMap = { 
            0 : "length",
        }

        self.sortedWindowSizes = None

    def ExtendPattern(self):
        #S1: Find total number of folds and sections
        totalFolds = self.numberOfCuts+1

        #S2: Find pattern length
        patternLength = len(self.pattern)

        #S3: Repeat pattern to fill the total number of folds depending on the mirror parameter
        x = np.linspace(0, patternLength - 1, patternLength)  # Adjusted x range
        y = np.array(list(self.pattern))
        #what if pattern size is bigger then cuts???
        #print(x,y)
        f = interpolate.interp1d(x, y)
        xnew = np.linspace(0, patternLength - 1, totalFolds)  # Adjusted x range
        ynew = f(xnew)
        transferredPattern = list(map(myround, ynew))
        return transferredPattern

    def GetWindowSize(self, action):
        #use windowSize to find x and y divide factors
        bin_size_x = self.sortedWindowSizes[action][0] 
        bin_size_y = self.sortedWindowSizes[action][1]
        return bin_size_x, bin_size_y

    def SetWindowSizeCode(self, col_size, row_size):
        for i, dims in enumerate(self.sortedWindowSizes):
            x,y = dims
            if x == col_size and y == row_size:
                self.bin_size_x = x
                self.bin_size_y = y
                return
        raise ValueError("couldn't find dimensions")

    def GetDivideFactors(self, layout_data):
        if layout_data.scheme == "scheme4":
            self.GetDivideFactors_scheme4(layout_data)
        elif layout_data.scheme == "scheme5":
            self.GetDivideFactors_scheme5(layout_data)

    def GetDivideFactors_scheme4(self, layout_data):

        gridSpec = layout_data.grid_definition
        #create map of action decoder based on layout
        numColumns = gridSpec.avg_sites_per_row
        numRows = gridSpec.numRows

        #find divisors of the given dimension
        xDivisors = divisors(numColumns)
        #xDivisors = list(reversed(log_scaled_steps(numColumns)))

        yDivisors = divisors(numRows)
        #yDivisors = list(reversed(log_scaled_steps(numRows)))
        
        print(f"divisors for {numColumns} are {xDivisors} using from sympy import divisors")
        print(f"divisors for {numRows} are {yDivisors} using from sympy import divisors")

        sXDivisors = xDivisors[:-1] if PLconfig_grid.designName in [ "picorv32a" , "tate", "jpeg"] else xDivisors
        print(f"sXDivisors={sXDivisors}, xDivisors={xDivisors}")
        #create all possible combinations
        windowSizes = list(itertools.product(
            sXDivisors,
            yDivisors[:-1]
            )
        )

        print(f"windowSizes={windowSizes}")

        # Sorting logic
        sortedWindowSizes = sorted(
            windowSizes,
            key=lambda x: (-sum(x), -x[0])
        )

        maxIndex = len(sortedWindowSizes) - 1
        index = self.windowSizeCode
        print(f"windowSizeCode={index}, the max window sizes={maxIndex}")
        if index > maxIndex:
            raise ValueError(f"windowSizeCode={index} exceeds the maxIndex={maxIndex}")

        #use windowSize to find x and y divide factors
        self.bin_size_x = sortedWindowSizes[index][0] 
        self.bin_size_y = sortedWindowSizes[index][1]

        #update divide ratio in constant file
        change_divide_ratio(
            layout_data,
            self.bin_size_x,
            self.bin_size_y
        )

        self.sortedWindowSizes = sortedWindowSizes

    def GetDivideFactors_scheme5(self, layout_data):

        gridSpec = layout_data.grid_definition
        #create map of action decoder based on layout
        numColumns = gridSpec.avg_sites_per_row
        numRows = gridSpec.numRows

        #find divisors of the given dimension
        #xDivisors = divisors(numColumns)
        #xDivisors = list(reversed(log_scaled_steps(numColumns)))
        xDivisors = [numColumns, numColumns//2, numColumns//4]
        
        #yDivisors = divisors(numRows)
        yDivisors = list(reversed(log_scaled_steps(numRows)))
        yDivisors = log_scaled_steps(numRows)
        yDivisors = [1, 2, 3, 4, 8, 16]
        
        #print(f"divisors for {numColumns} are {xDivisors} using from sympy import divisors")
        print(f"divisors for {numRows} are {yDivisors} using from sympy import divisors")

        #sXDivisors = xDivisors[:-1] if PLconfig_grid.designName in [ "picorv32a" , "tate", "jpeg"] else xDivisors
        #print(f"sXDivisors={sXDivisors}, xDivisors={xDivisors}")
        #create all possible combinations
        windowSizes = list(itertools.product(
            xDivisors,
            yDivisors[:-1]
            )
        )

        print(f"windowSizes={windowSizes}")

        # Sorting logic
        sortedWindowSizes = windowSizes
        '''
        sortedWindowSizes = sorted(
            windowSizes,
            key=lambda x: (-sum(x), -x[0])
        )
        '''

        maxIndex = len(sortedWindowSizes) - 1
        index = self.windowSizeCode
        print(f"windowSizeCode={index}, the max window sizes={maxIndex}")
        if index > maxIndex:
            raise ValueError(f"windowSizeCode={index} exceeds the maxIndex={maxIndex}")

        #use windowSize to find x and y divide factors
        self.bin_size_x = sortedWindowSizes[index][0] 
        self.bin_size_y = sortedWindowSizes[index][1]

        #update divide ratio in constant file
        change_divide_ratio(
            layout_data,
            self.bin_size_x,
            self.bin_size_y
        )

        self.sortedWindowSizes = sortedWindowSizes
