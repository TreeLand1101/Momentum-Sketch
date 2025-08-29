#ifndef HHBENCH_H
#define HHBENCH_H

#include <arpa/inet.h>
#include <netinet/in.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <iostream>

#include "MMap.h"
#include "CocoSketch.h"
#include "Elastic.h"
#include "CMHeap.h"
#include "SpaceSaving.h"
#include "MVSketch.h"
#include "StableSketch.h"
#include "TwoStage.h"
#include "TightSketch.h"
#include "MomentumSketch.h"
#include "MomentumSketchSIMD.h"
#include "MomentumSketchVariant.h"
#include <iomanip>

class BenchMark {
public:
    BenchMark(const std::string& PATH) {
        filePath = PATH;
        result = Load(PATH.c_str());
        dataset = (TUPLES*)result.start;
        length = result.length / sizeof(TUPLES);

        for (uint64_t i = 0; i < length; ++i) {
            tuplesMp[dataset[i]] += 1;
        }

        std::cout << "Dataset: " << filePath << std::endl;
        std::cout << "Number of packets: " << length << std::endl;
        std::cout << "Number of flows: " << tuplesMp.size() << std::endl << std::endl;
    }

    ~BenchMark() {
        if (outFile.is_open()) outFile.close();
        UnLoad(result);
    }

    template<typename Sketch>
    void HHBench(uint32_t MEMORY, double alpha) {
        Abstract<TUPLES>* tupleSketch;
        COUNT_TYPE threshold = alpha * length;

        if (!outFile.is_open()) {
            std::ostringstream fname;
            fname << "Performance/" << filePath << "/memory_" << MEMORY << "KB"<< "_alpha_" << alpha
                  << ".csv";
            outFile.open(fname.str(), std::ios::out | std::ios::trunc);
            if (!outFile.is_open()) {
                std::cerr << "Unable to open file: " << fname.str() << std::endl;
                return;
            }
            outFile << "Sketch Name,Insert Throughput (Mops),Query Throughput (Mops),"
                    << "Recall,Precision,F1 Score,AAE,ARE" << std::endl;
        }

        if constexpr (std::is_same_v<Sketch, TwoStage<TUPLES>>) {
            tupleSketch = new Sketch(MEMORY * 1024, threshold);
        } else {
            tupleSketch = new Sketch(MEMORY * 1024);
        }

        auto metrics = ComputeMetrics(tupleSketch, threshold);

        outFile << tupleSketch->name << ","
                << metrics.insert_throughput << ","
                << metrics.query_throughput << ","
                << metrics.recall << ","
                << metrics.precision << ","
                << metrics.f1score << ","
                << metrics.aae << ","
                << metrics.are << std::endl;

        std::cout << "=== " << tupleSketch->name << " ===" << std::endl;
        std::cout << "Heavy Hitter Threshold  : " << threshold << std::endl;
        std::cout << "Insert Throughput (Mops): " << metrics.insert_throughput << std::endl;
        std::cout << "Query Throughput (Mops) : " << metrics.query_throughput << std::endl;
        std::cout << "Recall                  : " << metrics.recall << std::endl;
        std::cout << "Precision               : " << metrics.precision << std::endl;
        std::cout << "F1 Score                : " << metrics.f1score << std::endl;
        std::cout << "AAE                     : " << metrics.aae << std::endl;
        std::cout << "ARE                     : " << metrics.are << std::endl;
        std::cout << std::endl;

        delete tupleSketch;
    }

private:
    std::string filePath;
    LoadResult result;
    TUPLES* dataset;
    uint64_t length;
    std::unordered_map<TUPLES, COUNT_TYPE> tuplesMp;
    std::ofstream outFile;

    struct Metrics {
        double insert_throughput;
        double query_throughput;
        double recall;
        double precision;
        double f1score;
        double aae;
        double are;
    };

    template<class Sketch>
    Metrics ComputeMetrics(Sketch* tupleSketch, COUNT_TYPE threshold) {
        using namespace std::chrono;
        // insert throughput
        auto t1 = high_resolution_clock::now();
        for (uint64_t i = 0; i < length; ++i) {
            tupleSketch->Insert(dataset[i]);
        }
        auto t2 = high_resolution_clock::now();
        double insert_duration = duration<double>(t2 - t1).count();
        double insert_throughput = (static_cast<double>(length) / insert_duration) / 1e6;

        // query throughput
        t1 = high_resolution_clock::now();
        for (uint64_t i = 0; i < length; ++i) {
            tupleSketch->Query(dataset[i]);
        }
        t2 = high_resolution_clock::now();
        double query_duration = duration<double>(t2 - t1).count();
        double query_throughput = (static_cast<double>(length) / query_duration) / 1e6;

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

#endif // HHBENCH_H
