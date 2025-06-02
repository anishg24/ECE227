# ECE 227 Final Project

## Installation

Install [uv](https://github.com/astral-sh/uv)

You can then do `uv run main.py`

## C++ Implementation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

## Generate Data

To grab the data:
```bash
cd data
make all
```

This will generate a `social.txt`, `comm.txt`, and `collab.txt` in the `data/` directory.