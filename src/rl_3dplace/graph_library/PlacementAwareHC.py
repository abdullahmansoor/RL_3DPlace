import scipy.cluster.hierarchy as shc
#import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)

def create_clusters_cut(clusters, number_of_cuts):
    logger.info(f"clusters={clusters}")
    uniquified_clusters = []
    for cluster in clusters:
        if cluster not in uniquified_clusters:
            uniquified_clusters.append(cluster)
 
    splits = np.array_split(uniquified_clusters, number_of_cuts)
    renameMap = {}
    counter = 0
    for split in splits:
        for item in split:
            renameMap[item] = counter
        counter += 1

    simplified_clusters = []
    for item in clusters:
        simplified_clusters.append(renameMap[item]) 
               
    logger.info(f"uniquified_clusters={uniquified_clusters}\n"
                f"renameMap={renameMap}\n"
                f"splits={splits}\n"
                )
    logger.info(f"simplified_clusters={simplified_clusters}")
    return simplified_clusters

def create_placement_aware_linkage(layout, orientation, number_of_cuts):
    cellsFromBin = lambda x: layout.get_cells_in_bin(x)
    #logger.info(orientation, layout.grid_spec.numRows, layout.grid_spec.avg_sites_per_row, layout.gri_spec.total_sites)
    if orientation == 0:
      cellsSequence = [] #array of cells obtained from bins traversed horizontally
      clusterSequence = [] #array of cluster mask for cellsSequence
      for x in range(layout.grid_spec.numBins):
          array = cellsFromBin(x)
          array = [i for i in array if i != None]
          logger.info(array)
          #append bin-level cells Sequence to overall sequence
          cellsSequence += list(array)

          #create mask for clusterSequence based on bin number (x) 
          barray = [x for y in range(len(array))]

          clusterSequence += barray
    else:
      cellsSequence = [] #array of cells obtained from bins traversed vertically
      clusterSequence = [] #array of cluster mask for cellsSequence

      logger.info(f"numBins = {layout.grid_spec.numBins}")
      logger.info(f"columns={layout.grid_spec.avg_sites_per_row}")
      logger.info(f"rows={layout.grid_spec.numRows}")

      for column in range(layout.grid_spec.avg_sites_per_row):
          for row in range(layout.grid_spec.numRows):
              binNumber = column + row*layout.grid_spec.avg_sites_per_row 
              #logger.info(column, row, row*layout.grid_spec.numRows, binNumber)
              array = cellsFromBin(binNumber)

              logger.info(f"array={array}")
              array = [i for i in array if i != None]
              logger.info(f"cellsSequence={cellsSequence}, array={array}")

              #append bin-level cells Sequence to overall sequence
              cellsSequence += list(array)

              #create mask for clusterSequence based on bin number (binNumber) 
              barray = [binNumber for y in range(len(array))]
              clusterSequence += barray

    countCellsSequence = len(cellsSequence)
    countClusterSequence = len(clusterSequence)
    assert( countCellsSequence == countClusterSequence), f"cellsSequence count doesn't match clusterSequence Count, {len(cellsSequence)} != {len(clusterSequence)}"

    logger.info(f"cellsSequence={cellsSequence} and length={countCellsSequence}")

    logger.info(f"clusterSequence={clusterSequence} and length={countClusterSequence}")
    clusterSequence = create_clusters_cut(clusterSequence, number_of_cuts)
    countClusterSequence = len(clusterSequence)
    logger.info(f"clusterSequence after cut={clusterSequence} and length={countClusterSequence}")

    clusters = [ None for x in range(countClusterSequence) ]

    nodesList = [x.name for x in layout.netlist.nodes.values() if x.movable]
    countNodes = layout.netlist.numNodesWOTerminals
    assert(countClusterSequence == countNodes ), f"length clusters {countClusterSequence} doesnt match number of netlist cells={countNodes}"
    for i in range(countClusterSequence):
        nodeName = cellsSequence[i]
        clusterNumber = clusterSequence[i]
        gIndex = nodesList.index(nodeName)
        logger.info(f"gIndex={gIndex}")
        clusters[gIndex] = clusterNumber
    logger.info(f"clusters={clusters}")

    assert(len(cellsSequence) == len(clusters)), f"cellsSequence count doesn't match clusters Count, {len(cellsSequence)} != {len(clusters)}"

    return clusters

    
def create_placement_aware_hierarchical_clusters(layout, orientation, number_of_cuts):
    
    clusters = create_placement_aware_linkage(layout, orientation, number_of_cuts)

    
    '''
    #perform hierarchical clustering on the distance matrix
    linkage = shc.linkage(distance_array, method='ward')
    #logger.info("linkage=\n",linkage)
    # linkage_matrix is the result of shc.linkage()
    clusters = shc.fcluster(linkage, t=number_of_cuts, criterion='maxclust')
    #clusters = shc.fcluster(linkage, t=number_of_cuts, criterion='maxclust_monocrit')
    logger.info("clusters=", clusters)
    # creae dendrogram
    dendrogram = shc.dendrogram(linkage, no_plot=False)

    # Save the dendrogram to a file (e.g., PNG format)
    plt.savefig('dendrogram.png')
    '''
    return clusters
