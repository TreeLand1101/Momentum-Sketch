#include "FilterBenchMark.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << "./FilterBenchMark <memory> <dataset1> <dataset2> ...\n";
        return 1;
    }

    uint32_t memory = std::stoi(argv[1]);

    for(uint32_t i = 2; i < argc; ++i) {
        FilterBenchMark dataset(argv[i]);
        dataset.Bench<CMSketch<TUPLES, uint16_t>>(memory);
        dataset.Bench<CMSketch<TUPLES, uint16_t>>(memory, 2);
        dataset.Bench<CUSketch<TUPLES, uint16_t>>(memory);
        dataset.Bench<CUSketch<TUPLES, uint16_t>>(memory, 2);
    }
    return 0;
}