#include <filesystem>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <cstdlib> 

#include "Util.h"
#include "MMap.h"
#include "CMSketch.h"

namespace fs = std::filesystem;

template<class T>
void ComputeFrequencyDistribution(const std::unordered_map<T, COUNT_TYPE>& record,
                                  const fs::path& outputFile) {
    std::vector<std::pair<T, COUNT_TYPE>> freqVector(record.begin(), record.end());
    std::sort(freqVector.begin(), freqVector.end(),
              [](auto const& a, auto const& b){ return a.second > b.second; });

    std::ofstream file(outputFile);
    if (!file) {
        std::cerr << "Error opening file" << "\n";
        return;
    }
    // Write CSV header
    file << "frequency" << "\n";
    for (auto const& p : freqVector)
        file << p.second << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <MEMORY> <ALPHA> <dataset1> [dataset2 ...]\n";
        return 1;
    }

    uint32_t MEMORY = static_cast<uint32_t>(std::atoll(argv[1]));
    double alpha = std::atof(argv[2]);

    char buf[64];
    std::snprintf(buf, sizeof(buf), "memory_%uKB_threshold_%.6f", MEMORY, alpha);
    std::string prefix(buf);

    for (int idx = 3; idx < argc; ++idx) {
        fs::path inFile{ argv[idx] };
        fs::path outDir = fs::current_path() / inFile.filename(); // Use full filename

        if (!fs::exists(outDir)) {
            if (!fs::create_directory(outDir)) {
                std::cerr << "Failed to create directory: " << outDir << "\n";
                continue;
            }
        }

        LoadResult result = Load(inFile.string().c_str());
        if (!result.start) {
            std::cerr << "Failed to load file: " << inFile << "\n";
            continue;
        }
        TUPLES* dataset = reinterpret_cast<TUPLES*>(result.start);
        uint64_t length = result.length / sizeof(TUPLES);

        std::unordered_map<TUPLES, COUNT_TYPE> flowFrequency, retainedFlowFrequency;
        CMSketch<TUPLES, COUNT_TYPE> sketch(MEMORY * 1024);
        COUNT_TYPE threshold = static_cast<COUNT_TYPE>(alpha * length);
        uint64_t retainedCount = 0;

        for (uint64_t i = 0; i < length; ++i) {
            auto& key = dataset[i];
            flowFrequency[key]++;
            if (sketch.Query(key) >= threshold) {
                retainedFlowFrequency[key]++;
                retainedCount++;
            }
            else {
                sketch.Insert(key);                
            }
        }

        std::cout << "\n=== File: " << inFile << " ===\n"
                  << "Threshold: " << threshold << "\n"
                  << "Total Packets: " << length << "\n"
                  << "Total Flows: " << flowFrequency.size() << "\n"
                  << "Retained Packets: " << retainedCount << "\n"
                  << "Retained Flows: " << retainedFlowFrequency.size() << "\n";

        ComputeFrequencyDistribution(
            flowFrequency,
            outDir / (prefix + "_total.csv")
        );
        ComputeFrequencyDistribution(
            retainedFlowFrequency,
            outDir / (prefix + "_retained.csv")
        );

        UnLoad(result);
    }

    return 0;
}