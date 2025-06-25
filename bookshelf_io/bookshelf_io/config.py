from dataclasses import dataclass
from typing import Any, List

@dataclass
class BookshelfConfig:
    grid_definition: Any = None
    binned_grid_definition: Any = None
    threeD_binned_grid_definition: Any = None
    folded_bins_map: Any = None
    divide_factor: int = 1
    single_cell_height: float = 1.0
    layer_values: List[int] | None = None
    import_num_rows: Any = 'x'
    import_num_sites: Any = 'x'
    designName: str = ''

# default global configuration
config = BookshelfConfig()

def set_config(new_config: BookshelfConfig) -> None:
    global config
    config = new_config
