#ifndef UTIL_H
#define UTIL_H

#include <x86intrin.h>

#include <vector>
#include <chrono>
#include <algorithm>
#include <functional>
#include <cstring>

#include "hash.h"

#pragma pack(1)

#define TUPLES_LEN 13

struct TUPLES{
    uint8_t data[TUPLES_LEN];

    inline uint32_t srcIP() const{
        return *((uint32_t*)(data));
    }

    inline uint32_t dstIP() const{
        return *((uint32_t*)(&data[4]));
    }

    inline uint16_t srcPort() const{
        return *((uint16_t*)(&data[8]));
    }

    inline uint16_t dstPort() const{
        return *((uint16_t*)(&data[10]));
    }

    inline uint8_t proto() const{
        return *((uint8_t*)(&data[12]));
    }

    uint8_t& operator[](size_t index) {
        return data[index];
    }
    
    inline uint64_t high64() const {
        return *((uint64_t*)(data));
    }

    inline uint64_t low40() const {
        uint64_t v = 0;
        v |= uint64_t(srcPort()) << 48;
        v |= uint64_t(dstPort()) << 32;
        v |= uint64_t(proto())   << 24;
        return v;
    }
};

bool operator == (const TUPLES& a, const TUPLES& b){
    return memcmp(a.data, b.data, sizeof(TUPLES)) == 0;
}

namespace std {
    template<>
    struct hash<TUPLES> {
        size_t operator()(const TUPLES& item) const noexcept {
            uint32_t out32;
            MurmurHash3_x86_32((uint8_t*)&item, sizeof(TUPLES), 0, &out32);
            return out32;
        }
    };
}

typedef uint32_t COUNT_TYPE;   

typedef std::chrono::high_resolution_clock::time_point TP;

inline TP now(){
    return std::chrono::high_resolution_clock::now();
}

template<typename T>
T Median(std::vector<T> vec, uint32_t len){
    std::sort(vec.begin(), vec.end());
    return (len & 1) ? vec[len >> 1] : (vec[len >> 1] + vec[(len >> 1) - 1]) / 2.0;
}

#endif
