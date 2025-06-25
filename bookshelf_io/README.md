# bookshelf_io

Utilities for loading and manipulating UCLA Bookshelf format netlists and grid specifications.  This package is extracted from the RL_3DPlace project and requires Python 3.10 or newer.

## Usage

```python
from bookshelf_io import importUcla, BookshelfConfig, set_config
config = BookshelfConfig()
set_config(config)
reader = importUcla(name="design", path=Path("/path/to/files"))
```
