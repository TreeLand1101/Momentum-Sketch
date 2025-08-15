# CPU Code
This CPU code measures the performance of the sketches using the following metrics: `AAE`, `ARE`, `F1 score`, `recall`, `precision`, `insert throughput`, and `query throughput`.

## Repository structure
*  `Common/`: the hash and mmap functions
*  `Struct/`: the data structures, such as heap and hash table
*  `Src/`: sketch algorithms
*  `Benchmark.h`: the benchmarks about ARE, recall rate, and precision rate

## Basic Requirements
### C++
- `cmake`
- `g++`

### Python (only used for plotting)
- `matplotlib`
- `numpy`

## SIMD Requirements
To run MomentumSketchSIMD, the CPU must support the AVX‑512 instruction set.

## How to run
- Put the dataset (in `.dat` format) under the `CPU/` folder.
- To modify `dataset`, `memory` and `heavy hitter threshold`, please refer to `run.sh`
- To run on a different sketch, please refer to `main.cpp`

```bash
$ cmake .
$ sh run.sh
```
Experimental `.dat`: https://drive.google.com/drive/folders/1uSnbV46bnn8ByMxms7jXqk_c26W-jWyt?usp=drive_link