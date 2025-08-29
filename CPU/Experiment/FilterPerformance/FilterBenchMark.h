#include <arpa/inet.h>
#include <netinet/in.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <utility>
#include <chrono>
#include <unordered_map>

#include "MMap.h"
#include "CUSketch.h"
#include "CMSketch.h"

class FilterBenchMark {
public:
    FilterBenchMark(std::string PATH) {
        result = Load(PATH.c_str());
        dataset = (TUPLES*)result.start;
        length = result.length / sizeof(TUPLES);

        for (uint64_t i = 0; i < length; ++i) {
            tuplesMp[dataset[i]] += 1;
        }
    }

    ~FilterBenchMark() {
        if (outFile.is_open()) {
            outFile.close();
        }
        UnLoad(result);
    }

    template<typename Filter, typename... Args>
    void Bench(uint32_t MEMORY, Args&&... args) {
        if (!outFile.is_open()) {
            std::ostringstream fname;
            fname << "memory_" << MEMORY << "KB" << ".csv";
            outFile.open(fname.str(), std::ios::out | std::ios::trunc);
            if (!outFile.is_open()) {
                std::cerr << "Unable to open file: " << fname.str() << std::endl;
                return;
            }
            outFile << "Filter Name,Insert Throughput (Mops),Query Throughput (Mops),AAE,ARE" << std::endl;
        }

        Filter* tupleFilter = new Filter(MEMORY * 1024, std::forward<Args>(args)...);
        auto metrics = ComputeMetrics(tupleFilter);

        outFile << tupleFilter->name << ","
                << metrics.insert_throughput << ","
                << metrics.query_throughput << ","
                << metrics.aae << ","
                << metrics.are << std::endl;

        delete tupleFilter;
    }

private:
    LoadResult result;
    TUPLES* dataset;
    uint64_t length;
    std::unordered_map<TUPLES, COUNT_TYPE> tuplesMp;
    std::ofstream outFile;

    struct Metrics {
        double insert_throughput;
        double query_throughput;
        double aae;
        double are;
    };

    template<class Filter>
    Metrics ComputeMetrics(Filter* tupleFilter) {
        TP start, end;

        start = std::chrono::high_resolution_clock::now();
        for (uint32_t j = 0; j < length; ++j) {
            tupleFilter->Insert(dataset[j]);
        }
        end = std::chrono::high_resolution_clock::now();
        double duration_insert = std::chrono::duration<double>(end - start).count();
        double throughput_insert = (static_cast<double>(length) / duration_insert) / 1e6;

        start = std::chrono::high_resolution_clock::now();
        for (uint32_t j = 0; j < length; ++j) {
            tupleFilter->Query(dataset[j]);
        }
        end = std::chrono::high_resolution_clock::now();
        double duration_query = std::chrono::duration<double>(end - start).count();
        double throughput_query = (static_cast<double>(length) / duration_query) / 1e6;

        double aae = 0, are = 0;
        for (auto it = tuplesMp.begin(); it != tuplesMp.end(); ++it) {
            double realF = it->second;
            double estF = tupleFilter->Query(it->first);
            aae += std::abs(realF - estF);
            are += std::abs(realF - estF) / realF;
        }
        size_t num_flows = tuplesMp.size();
        aae /= num_flows;
        are /= num_flows;

        return {throughput_insert, throughput_query, aae, are};
    }
};