#include "BenchMark.h"
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <utility>
#include <cstdlib>
#include <filesystem>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <memory> <alpha> <dataset>" << std::endl;
        return 1;
    }
    
    uint32_t memory = std::stoi(argv[1]);
    double alpha = std::stod(argv[2]);
    std::string dataset_path = argv[3];
    
    std::filesystem::path p(dataset_path);
    std::string dataset_name = p.filename().string();
    std::string dir_name = dataset_name;
    std::filesystem::create_directory(dir_name);
    
    std::string csv_filename = dir_name + "/performance.csv";
    FILE* fp = fopen(csv_filename.c_str(), "w");
    if (fp == nullptr) {
        std::cerr << "Cannot open file " << csv_filename << std::endl;
        return 1;
    }
    
    fprintf(fp, "Filter Ratio,Sketch Ratio,Admission Ratio,F1 Score,Precision,Recall,Insert Throughput (Mops),Query Throughput (Mops)\n");
    
    std::vector<double> admission_ratios = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    std::vector<std::pair<double, double>> memory_ratios = {
        {0.2, 0.8}, {0.3, 0.7}, {0.4, 0.6}, {0.5, 0.5}, {0.6, 0.4}, {0.7, 0.3}, {0.8, 0.2}
    };
    
    BenchMark dataset(dataset_path, "Dataset");
    for (auto& ratio_pair : memory_ratios) {
        double filter_ratio = ratio_pair.first;
        double sketch_ratio = ratio_pair.second;
        for (double admission_ratio : admission_ratios) {
            Metrics metrics = dataset.HHBenchWithMetrics(memory * 1024, alpha, filter_ratio, sketch_ratio, admission_ratio);
            fprintf(fp, "%.2f,%.2f,%.2f,%.6f,%.6f,%.6f,%.6f,%.6f\n", 
                    filter_ratio, sketch_ratio, admission_ratio, 
                    metrics.f1score, metrics.precision, metrics.recall, 
                    metrics.insert_throughput, metrics.query_throughput);
        }
    }
    
    fclose(fp);
    std::cout << "CSV output saved to " << csv_filename << std::endl;
    return 0;
}