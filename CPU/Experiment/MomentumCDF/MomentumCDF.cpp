#include <arpa/inet.h>
#include <netinet/in.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include "MMap.h"
#include "Abstract.h"
#include "MVSketch.h"

template<typename DATA_TYPE>
class MVSketchMomentum : public MVSketch<DATA_TYPE> {
public:
    using MVSketch<DATA_TYPE>::HASH_NUM;
    using MVSketch<DATA_TYPE>::LENGTH;
    using MVSketch<DATA_TYPE>::sketch;

    MVSketchMomentum(uint32_t _MEMORY, std::string _name = "MVSketchMomentum")
        : MVSketch<DATA_TYPE>(_MEMORY, _name) {}

    bool Search(const DATA_TYPE& item) {
        for (uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t pos = hash(item, i) % LENGTH;
            if (sketch[i][pos].ID == item) {
                return true;
            }
        }
        return false;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <memory_in_bytes> <alpha> <dataset_path>" << std::endl;
        return EXIT_FAILURE;
    }

    uint32_t MEMORY = static_cast<uint32_t>(std::atoi(argv[1]));
    if (MEMORY == 0) {
        std::cerr << "Error: invalid memory value \"" << argv[1] << "\"" << std::endl;
        return EXIT_FAILURE;
    }

    double alpha = std::atof(argv[2]);
    if (!(alpha > 0.0 && alpha < 1.0)) {
        std::cerr << "Error: invalid alpha value \"" << argv[2]
                  << "\" (should be between 0 and 1)" << std::endl;
        return EXIT_FAILURE;
    }

    std::string input_path = argv[3];
    size_t last_slash = input_path.find_last_of("/\\");
    std::string dir_name = (last_slash == std::string::npos) ? input_path : input_path.substr(last_slash + 1);

    if (mkdir(dir_name.c_str(), 0755) != 0 && errno != EEXIST) {
        std::cerr << "Error: failed to create directory \"" << dir_name << "\"" << std::endl;
        return EXIT_FAILURE;
    }

    LoadResult result = Load(input_path.c_str());
    if (result.start == nullptr || result.length == 0) {
        std::cerr << "Error: failed to load dataset \"" << input_path << "\"" << std::endl;
        return EXIT_FAILURE;
    }
    TUPLES* dataset = reinterpret_cast<TUPLES*>(result.start);
    uint64_t length = result.length / sizeof(TUPLES);

    std::unordered_map<TUPLES, COUNT_TYPE> tuplesMp;
    std::unordered_map<TUPLES, COUNT_TYPE> tuplesMomentum;
    MVSketchMomentum<TUPLES>* sketch = new MVSketchMomentum<TUPLES>(MEMORY * 1024);

    for (uint64_t i = 0; i < length; ++i) {
        TUPLES flow = dataset[i];
        tuplesMp[flow] += 1;
        if (tuplesMomentum.find(flow) == tuplesMomentum.end()) {
            tuplesMomentum[flow] = 0;
        }

        if (sketch->Search(flow)) {
            if (sketch->Query(flow) > std::numeric_limits<COUNT_TYPE>::max() - tuplesMomentum[flow]) {
                tuplesMomentum[flow] = std::numeric_limits<COUNT_TYPE>::max();
            } else {
                tuplesMomentum[flow] = tuplesMomentum[flow] + sketch->Query(flow);
            }
        } else {
            tuplesMomentum[flow] = tuplesMomentum[flow] / 2;
        }

        sketch->Insert(flow);
    }

    COUNT_TYPE threshold = static_cast<COUNT_TYPE>(alpha * length);

    std::vector<COUNT_TYPE> nonHeavyMomentum;
    std::vector<COUNT_TYPE> heavyHitterMomentum;
    for (const auto& pair : tuplesMp) {
        const TUPLES& flow = pair.first;
        COUNT_TYPE count = pair.second;
        COUNT_TYPE momentum = tuplesMomentum[flow];
        if (count > threshold) {
            heavyHitterMomentum.push_back(momentum);
        } else {
            nonHeavyMomentum.push_back(momentum);
        }
    }
    std::sort(nonHeavyMomentum.begin(), nonHeavyMomentum.end());
    std::sort(heavyHitterMomentum.begin(), heavyHitterMomentum.end());

    std::string nonHeavyCSV = dir_name + "/nonheavy_momentum_cdf.csv";
    std::string heavyHitterCSV = dir_name + "/heavyhitter_momentum_cdf.csv";

    std::ofstream nonHeavyOut(nonHeavyCSV);
    std::ofstream heavyHitterOut(heavyHitterCSV);
    if (!nonHeavyOut.is_open() || !heavyHitterOut.is_open()) {
        std::cerr << "Error: cannot open output CSV in folder \"" << dir_name << "\"" << std::endl;
        delete sketch;
        UnLoad(result);
        return EXIT_FAILURE;
    }

    nonHeavyOut << "momentum,cdf" << std::endl;
    heavyHitterOut << "momentum,cdf" << std::endl;
    nonHeavyOut << std::fixed << std::setprecision(6);
    heavyHitterOut << std::fixed << std::setprecision(6);

    if (!nonHeavyMomentum.empty()) {
        for (size_t i = 0; i < nonHeavyMomentum.size(); ++i) {
            double cdf = static_cast<double>(i + 1) / nonHeavyMomentum.size();
            nonHeavyOut << nonHeavyMomentum[i] << "," << cdf << std::endl;
        }
    } else {
        nonHeavyOut << "0,0" << std::endl;
    }

    if (!heavyHitterMomentum.empty()) {
        for (size_t i = 0; i < heavyHitterMomentum.size(); ++i) {
            double cdf = static_cast<double>(i + 1) / heavyHitterMomentum.size();
            heavyHitterOut << heavyHitterMomentum[i] << "," << cdf << std::endl;
        }
    } else {
        heavyHitterOut << "0,0" << std::endl;
    }

    nonHeavyOut.close();
    heavyHitterOut.close();

    delete sketch;
    UnLoad(result);
    return EXIT_SUCCESS;
}