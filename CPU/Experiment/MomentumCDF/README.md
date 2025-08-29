# MomentumCDF
To verify that momentum metric can distinguish heavy hitters from non-heavy flows, we use MV-Sketch to compute the momentum of each flow.

After processing all packets, we classify flows into true non‐heavy and true heavy hitters and plot the cumulative distribution function (CDF) of momentum for each group.

## Requirements
### C++
- `cmake`
- `g++`

### Python (only used for plotting)
- `matplotlib`
- `numpy`

## How to run
```bash
$ cmake .
$ sh run.sh
```