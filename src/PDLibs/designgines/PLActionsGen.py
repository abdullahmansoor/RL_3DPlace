import numpy as np
import os
import sys
import copy
import itertools


class ParametricActionsGen:
    def __init__(self,
                 cutValues=np.array([1, 2, 3, 3, 4, 5, 6, 7]),
                 directions=np.array([0, 1]),
                 patterns=np.array([0, 1, 2, 3, 4]),
                 mirrorValues=np.array([0,1]),
                 windowSizeCodes=np.array([0,1,2,3,4,5]),
                 distanceMatrixCodes=np.array([0,1,2,3,4,5]),
                 mode="default",
                 maps = {
                     'directions' : {'ver': 0, 'hor' : 1},
                     'mirrorValues' : {'True' : 0, 'False' : 1},
                     'patterns' : {"01" : 0, "10" : 1, "101" : 2, 
                                   "010" : 3, "100" : 4, "001" : 5,
                                    "110": 6, "011" : 7
                                   },
                     'mode' : {'default' : 0, 'aggHC' : 1, 'paHC' : 2 },
                 },
    ):
        self.cutValues = cutValues
        self.directions = directions
        self.patterns = patterns
        self.mirrorValues = mirrorValues
        self.distanceMatrixCodes = distanceMatrixCodes
        self.windowSizeCodes = windowSizeCodes
        self.mode = mode
        self.maps = maps
        
        #output variable
        self.actionsValues = None

    def GenerateActions(self):
        if self.maps['mode'][self.mode] == 0:
            self.actionsValues = list(itertools.product(
                self.cutValues,
                self.directions,
                self.patterns,
                self.mirrorValues
                )
            )
        elif self.maps['mode'][self.mode] == 1:
            self.actionsValues = list(itertools.product(
                self.cutValues,
                self.patterns,
                self.mirrorValues,
                self.distanceMatrixCodes
                )
            )
        elif self.maps['mode'][self.mode] == 2:
            self.actionsValues = list(itertools.product(
                self.cutValues,
                self.directions,
                self.patterns,
                self.windowSizeCodes,
                )
            )
        else:
            raise(f"Value Error, mode= {self.mode} not supported!")

        #print(f"actions={self.actionsValues}")
        print(f"Number of DHA actions are {len(self.actionsValues)}")

    def GetValue(self, section, target):
        for k,v in self.maps[section].items():
            if v == target:
                return k
        raise ValueError(f"key for {target} not found!")

    def DecodeAction(self, actionCode):
        if self.maps['mode'][self.mode] == 0:
            cutValue, directionCode, patternCode, mirrorCode = self.actionsValues[actionCode]
            direction = self.GetValue("directions", directionCode)
            pattern = self.GetValue("patterns", patternCode)
            mirror = self.GetValue("mirrorValues", mirrorCode)

            #print("cutValue, direction, pattern, mirror")
            #print(cutValue, direction, pattern, mirror)
            return cutValue, direction, pattern, mirror
        elif self.maps['mode'][self.mode] == 1:
            cutValue, patternCode, mirrorCode, distanceMatrixCode = self.actionsValues[actionCode]
            pattern = self.GetValue("patterns", patternCode)
            mirror = self.GetValue("mirrorValues", mirrorCode)

            #print("cutValue, direction, pattern, mirror")
            #print(cutValue, direction, pattern, mirror)
            return cutValue, patternCode, mirrorCode, distanceMatrixCode
        elif self.maps['mode'][self.mode] == 2:
            cutValue, directionCode, patternCode, windowSizeCode = self.actionsValues[actionCode]
            direction = self.GetValue("directions", directionCode)
            pattern = self.GetValue("patterns", patternCode)
 
            #print("cutValue, direction, pattern, mirror")
            #print(cutValue, direction, pattern, mirror)
            return cutValue, directionCode, pattern, windowSizeCode
        else:
            raise(f"Value Error, mode= {self.mode} not supported!")



def testSchemeCoding():
    from utils.DatasetStateFoldingActionGen import FindFoldingSchemeNumber
    ag = PLconfig_grid.ag
    schemeNumber = FindFoldingSchemeNumber()
    print(f"Scheme number is {schemeNumber}")

def main():
    '''
    import argparse
    parser = argparse.ArgumentParser()    
    
    parser.add_argument("-exportName", action="store", dest="exportName",
                        help="design name",
                        required=False, type=str)    
    
    parser.add_argument("-exportDir", action="store", dest="exportDir",
                        help="folder of UCLA design files",
                        required=False, type=str)    
        
    args = parser.parse_args()
    '''

    #testSchemeCoding()
    #exit()

    inputDir = os.path.abspath(PLconfig_grid.inputDir)
    designName = PLconfig_grid.designName

    ag = ParametricActionsGen(mode="paHC")
    ag.GenerateActions()
    
    print(len(ag.actionsValues))

    exit()
    
    for action in range(len(ag.actionsValues)):
        cutValue, direction, pattern, mirror = ag.DecodeAction(action)

        folding_params = {
            'number_of_cuts' : cutValue,
            'direction' : direction,
            'pattern' : pattern,
            'mirror': mirror
        }

        fm = FoldedBinsMap(
            PLconfig_grid.binned_grid_definition,
            PLconfig_grid.threeD_binned_grid_definition,
            2,
            folding_params
        )
        #layer2, perlayer_bin2 = fm.global_to_perlayer_bin(12)
        #bin_number = fm.threeD_to_twoD(71)
        #print('bin number=', bin_number)

        #fm.DrawFoldingChart()

    print("maps=\n",ag.maps)
    testSchemeCoding()

if __name__ == "__main__":
    main()
