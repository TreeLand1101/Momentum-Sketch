#ifndef HASH_H
#define HASH_H

#include <cstdint>
#include <array>
#include "murmur3.h"

#include <random>
#include <iostream>
#include <limits.h>
#include <stdint.h>

template<typename T>
inline uint32_t hash(const T& data, uint32_t seed = 0);

template<typename T>
inline void hash128(const T& data, uint32_t out32[4], uint32_t seed = 0);

inline uint64_t randomGenerator();

static std::random_device rd;
static std::mt19937_64 rng(rd());
static std::uniform_real_distribution<double> dis(0, 1);

inline uint64_t randomGenerator(){
    return rng();
}

class Hash {
public:
    template<typename T>
    static uint32_t MurmurHash32(const T& data, uint32_t seed) {
        uint32_t out32;
        MurmurHash3_x86_32(&data, sizeof(T), seed, &out32);
        return out32;
    }

    template<typename T>
    static void MurmurHash128(const T& data, uint32_t out32[4], uint32_t seed) {
        MurmurHash3_x86_128(&data, sizeof(T), seed, out32);
    }
};

template<typename T>
inline uint32_t hash(const T& data, uint32_t seed){
    return Hash::MurmurHash32(data, seed);
}

template<typename T>
inline void hash128(const T& data, uint32_t out32[4], uint32_t seed){
    return Hash::MurmurHash128(data, out32, seed);
}

#endif
