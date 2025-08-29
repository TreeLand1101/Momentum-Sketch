#include "MMap.h"
#include "TwoStage.h"

#include <unordered_map>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

struct Metrics {
    double insert_throughput;
    double query_throughput;
    double recall;
    double precision;
    double f1score;
    double aae;
    double are;
};


class BenchMark {
public:
    BenchMark(const std::string &PATH, const std::string &name) {
        fileName = name;
        result = Load(PATH.c_str());
        dataset = (TUPLES*)result.start;
        length = result.length / sizeof(TUPLES);
        for (uint64_t i = 0; i < length; ++i) {
            tuplesMp[dataset[i]] += 1;
        }
    }

    ~BenchMark() {
        UnLoad(result);
    }

    Metrics HHBenchWithMetrics(uint32_t MEMORY, double alpha, double FILTER_RATIO, double SKETCH_RATIO, double ADMISSION_RATIO) {
        COUNT_TYPE threshold = alpha * length;
        TwoStage<TUPLES>* tupleSketch = new TwoStage<TUPLES>(MEMORY, threshold, FILTER_RATIO, SKETCH_RATIO, ADMISSION_RATIO);

        Metrics metrics = ComputeMetrics(tupleSketch, threshold);

        delete tupleSketch;
        return metrics;
    }

private:
    std::string fileName;
    LoadResult result;
    TUPLES* dataset;
    uint64_t length;
    std::unordered_map<TUPLES, COUNT_TYPE> tuplesMp;

    Metrics ComputeMetrics(TwoStage<TUPLES>* tupleSketch, COUNT_TYPE threshold) {
        using namespace std::chrono;
        // insert throughput
        auto t1 = high_resolution_clock::now();
        for (uint64_t i = 0; i < length; ++i) {
            tupleSketch->Insert(dataset[i]);
        }
        auto t2 = high_resolution_clock::now();
        double duration_insert = duration<double>(t2 - t1).count();
        double insert_throughput = (static_cast<double>(length) / duration_insert) / 1e6;

        // query throughput
        t1 = high_resolution_clock::now();
        for (uint64_t i = 0; i < length; ++i) {
            tupleSketch->Query(dataset[i]);
        }
        t2 = high_resolution_clock::now();
        double duration_query = duration<double>(t2 - t1).count();
        double query_throughput = (static_cast<double>(length) / duration_query) / 1e6;

        // heavy hitter metrics
        double realHH = 0, estHH = 0, bothHH = 0, aae = 0, are = 0;
        std::unordered_map<TUPLES, COUNT_TYPE> estTuple = tupleSketch->AllQuery();

        for(auto it = tuplesMp.begin(); it != tuplesMp.end(); ++it){
            bool real, est;
            double realF = it->second, estF = estTuple[it->first];
            
            real = (realF > threshold);
            est = (estF > threshold);

            realHH += real;
            estHH += est;

            if(real && est){
                bothHH += 1;
                aae += abs(realF - estF);
                are += abs(realF - estF) / realF;
            }
        }

        double recall = bothHH / realHH;
        double precision = bothHH / estHH;
        double f1score = 2 * (precision * recall) / (precision + recall);
        aae /= bothHH;
        are /= bothHH;

        return {insert_throughput, query_throughput, recall, precision, f1score, aae, are};
    }
};
