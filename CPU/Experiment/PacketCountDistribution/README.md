# Packet Count Distribution
We analyze the raw packet-count distributions of network traces before and after filtering.

To simulate the filtering scheme for Stage 1, we employed a 128 KB CM Sketch and configured the heavy-hitter threshold at 0.0001 of the total packet count. We update the sketch on each packet; once a flow’s estimate meets the threshold, we mark it as a retained flow and count all its subsequent packets in the retained packet count.

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