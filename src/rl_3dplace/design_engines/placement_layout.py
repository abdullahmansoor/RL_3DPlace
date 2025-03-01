import logging

class placement_layout:
    def __init__(self, grid_spec, netlist, num_nodes, num_rows, avg_sites_per_row, bins_count, bin_size_x, bin_size_y, single_cell_height):
        """Initialize the simplified layout with binning support and built-in location conversions.

        Args:
            grid_spec: The grid specification object.
            netlist: Dictionary representing the netlist with node information.
            num_nodes: Total number of nodes in the layout.
            num_rows: Number of rows in the grid.
            avg_sites_per_row: Average sites per row in the grid.
            bins_count: Total number of bins.
            bin_size_x: Number of columns per bin.
            bin_size_y: Number of rows per bin.
        """
        self.grid_spec = grid_spec
        self.netlist = netlist
        self.num_nodes = num_nodes
        self.num_rows = num_rows
        self.avg_sites_per_row = avg_sites_per_row
        self.bins_count = bins_count
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.single_cell_height = single_cell_height

        self.matrix = None
        self.cell_bin_mapping = None

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.WARN,
            format='%(asctime)s - %(levelname)s -  [%(pathname)s:%(lineno)d] %(message)s'
        )
        
    def initialize(self):
        """Initialize the layout by creating and updating the matrix."""
        self.logger.info(f" \nnum_nodes={self.num_nodes},\n"
              f" num_rows={self.num_rows},\n"
              f" avg_sites_per_row={self.avg_sites_per_row},\n"
              f" bins_count={self.bins_count},\n"
              f" bin_size_x={self.bin_size_x},\n"
              f" bin_size_y={self.bin_size_y}")
        self._create_matrix()
        self._update_matrix()
        self._update_cell_bin_mapping()

    def _create_matrix(self):
        """Create the layout matrix with binning support."""
        num_bins_x = self.avg_sites_per_row
        num_bins_y = self.num_rows
        self.matrix = {
            'cellMatrix': [[[None for _ in range(self.bin_size_x)] for _ in range(self.bin_size_y)]
                           for _ in range(num_bins_x * num_bins_y)],
            'occMatrix': [[[0 for _ in range(self.bin_size_x)] for _ in range(self.bin_size_y)]
                          for _ in range(num_bins_x * num_bins_y)],
        }
        self.cell_bin_mapping = {}
        self.netlist.create_graph()

    def _update_matrix(self):
        """Update the matrix based on the current netlist."""
        for node_name, node_obj in self.netlist.nodes.items():
            if not node_obj.movable: continue
            grid_location = self._convert_to_grid_location(node_obj)
            bin_idx = self._calculate_bin_index(grid_location['ygrid'], grid_location['xgrid'])
            local_row = grid_location['ygrid'] % self.bin_size_y
            local_col = grid_location['xgrid'] % self.bin_size_x
            #self.logger.info(f"bin_idx={bin_idx}, \n \
            #    local_row={local_row},\n \
            #    local_col={local_col}, \n \
            #")
            self.matrix['cellMatrix'][bin_idx][local_row][local_col] = node_name
            self.matrix['occMatrix'][bin_idx][local_row][local_col] = 1

    def _update_cell_bin_mapping(self):
        """Update a flattened list of cells in the bin."""
        for bin_idx in range(len(self.matrix['cellMatrix'])):
            self.cell_bin_mapping[bin_idx] = []
            for row in range(len(self.matrix['cellMatrix'][bin_idx])):
                for col in range(len(self.matrix['cellMatrix'][bin_idx][row])):
                    node_name = self.matrix['cellMatrix'][bin_idx][row][col]
                    self.cell_bin_mapping[bin_idx].append(node_name)

    def get_cells_in_bin(self, bin_number):
        """Retrieve cells in a specified bin."""
        return self.cell_bin_mapping.get(bin_number, [])

    def _convert_to_grid_location(self, node):
        """Convert a node's centroid to a grid-based location."""
        x_center = int(node.point_lb.x + (node.width / 2.0))
        y_center = node.point_lb.y

        row_object = self.grid_spec.rows[0]
        #xgrid = int((x_center - row_object.subRowOrigin) / node.width)
        xgrid = int((x_center - row_object.subRowOrigin))
        ygrid = int((y_center - row_object.coordinate) / self.single_cell_height)

        #self.logger.info(f"Node={node.name}, Centroid=({x_center}, {y_center}), xgrid={xgrid}, ygrid={ygrid}")

        return {
            'xgrid': xgrid,
            'ygrid': ygrid
        }

    def _calculate_bin_index(self, row, col):
        """Calculate the bin index for a given row and column."""
        bin_x = col // self.bin_size_x
        bin_y = row // self.bin_size_y
        return bin_y * self.avg_sites_per_row + bin_x

    def find_empty_location(self):
        """Find the first empty location in the matrix."""
        for bin_idx, bin_matrix in enumerate(self.matrix['occMatrix']):
            for row_index, row in enumerate(bin_matrix):
                for col_index, cell in enumerate(row):
                    if cell == 0:
                        return bin_idx, row_index, col_index
        return None

    def __str__(self):
        """Provide a string representation of the layout."""
        return (f"GridSpec: {self.grid_spec},\n"
                f"Number of Nodes: {self.num_nodes},\n"
                f"Matrix: {self.matrix}")
