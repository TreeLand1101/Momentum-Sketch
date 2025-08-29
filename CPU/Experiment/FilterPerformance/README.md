# Filter Performance
We measured error and throughput on the CAIDA 2016 dataset for four sketches: two‐row CM Sketch, four‐row CM Sketch, two‐row CU Sketch, and four‐row CU Sketch.

According to the results, we selected the two-row CU Sketch as our filter sketch.

## Requirements
### C++
- `cmake`
- `g++`

### Python (only used for plotting)
- `matplotlib`
- `numpy`
- `seaborn`

## How to run
```bash
$ cmake .
$ sh run.sh
```