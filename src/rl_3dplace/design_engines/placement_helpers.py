import sys

from designgines.PLGridSpec import BinnedGrid

def change_divide_ratio(layout_data, new_bin_size_x, new_bin_size_y):
    layout_data.constData.bin_size_x = new_bin_size_x
    layout_data.constData.bin_size_y = new_bin_size_y
    layout_data.binned_grid_definition = BinnedGrid.from_grid(
        layout_data.grid_definition,
        new_bin_size_x,
        new_bin_size_y
    )
