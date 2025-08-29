#ifndef MOMENTUMSKETCHSIMD_H
#define MOMENTUMSKETCHSIMD_H

#include "Abstract.h"
#include <bit>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <omp.h>
#include <immintrin.h>

template<typename DATA_TYPE>
class MomentumSketchSIMD : public Abstract<DATA_TYPE> {
public:
    typedef std::unordered_map<DATA_TYPE, COUNT_TYPE> HashMap;

    struct Bucket {
        COUNT_TYPE momentum;
        DATA_TYPE ID;
        COUNT_TYPE counter;
    };

    MomentumSketchSIMD(uint32_t _MEMORY, std::string _name = "Momentum-Sketch SIMD") {
        this->name = _name;

        LENGTH = _MEMORY / sizeof(Bucket) / HASH_NUM;
        sketch = new Bucket* [HASH_NUM];
        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            sketch[i] = new Bucket[LENGTH];
            memset(sketch[i], 0, sizeof(Bucket) * LENGTH);
        }
    }

    ~MomentumSketchSIMD() {
        for(uint32_t i = 0; i < HASH_NUM; ++i)
            delete [] sketch[i];
        delete [] sketch;
    }

    void Insert(const DATA_TYPE& item) {
        COUNT_TYPE min = std::numeric_limits<COUNT_TYPE>::max();
        int R = -1, P = -1;

        uint32_t onehash[4];
        hash128(item, onehash);

        __m256i vhash = _mm256_set_epi32(0, onehash[0], 0, onehash[1], 0, onehash[2], 0, onehash[3]);
        __m256i vwidth = _mm256_set1_epi32(LENGTH);
        union {__m256i vindex; uint64_t index[4];};
        vindex = _mm256_srli_epi64(_mm256_mul_epu32(vhash, vwidth), 32);

        __m512i vkey = _mm512_set_epi64(
            sketch[3][index[3]].ID.high64(), sketch[3][index[3]].ID.low40(),
            sketch[2][index[2]].ID.high64(), sketch[2][index[2]].ID.low40(),
            sketch[1][index[1]].ID.high64(), sketch[1][index[1]].ID.low40(),
            sketch[0][index[0]].ID.high64(), sketch[0][index[0]].ID.low40()
        );

        __m512i cmpkey = _mm512_broadcast_i64x2(_mm_set_epi64x(item.high64(), item.low40()));
        __mmask8 cmpmask = _mm512_cmpeq_epi64_mask(vkey, cmpkey);

        uint8_t lanemask = 0b11;

        for(int i = 0; i < HASH_NUM; ++i) {
            if ((cmpmask & lanemask) == lanemask) {
                if (std::numeric_limits<COUNT_TYPE>::max() - sketch[i][index[i]].momentum < sketch[i][index[i]].counter) {
                    sketch[i][index[i]].momentum = std::numeric_limits<COUNT_TYPE>::max();
                } 
                else {
                    sketch[i][index[i]].momentum += sketch[i][index[i]].counter;
                }
                sketch[i][index[i]].counter++;
                return;                
            }
            else {
                if (sketch[i][index[i]].ID[0] == '\0' && sketch[i][index[i]].counter == 0) {
                    sketch[i][index[i]].ID = item;
                    sketch[i][index[i]].counter = 1;
                    sketch[i][index[i]].momentum = 1;
                    return;
                }                
                if (sketch[i][index[i]].counter < min) {
                    min = sketch[i][index[i]].counter;
                    R = i;
                    P = index[i];
                }                
            }
            lanemask <<= 2;
        }

        sketch[R][P].momentum /= DECAY_FACTOR;

        if (randomGenerator() % ((uint64_t)sketch[R][P].counter * sketch[R][P].momentum + 1) == 0) {
            if (--sketch[R][P].counter == 0) {
                sketch[R][P].ID = item;
                sketch[R][P].counter = 1;
                sketch[R][P].momentum = 1;
            }
        }
    }

    COUNT_TYPE Query(const DATA_TYPE& item){
        uint32_t onehash[4];
        hash128(item, onehash);

        __m256i vhash = _mm256_set_epi32(0, onehash[0], 0, onehash[1], 0, onehash[2], 0, onehash[3]);
        __m256i vwidth = _mm256_set1_epi32(LENGTH);
        union {__m256i vindex; uint64_t index[4];};
        vindex = _mm256_srli_epi64(_mm256_mul_epu32(vhash, vwidth), 32);

        __m512i vkey = _mm512_set_epi64(
            sketch[3][index[3]].ID.high64(), sketch[3][index[3]].ID.low40(),
            sketch[2][index[2]].ID.high64(), sketch[2][index[2]].ID.low40(),
            sketch[1][index[1]].ID.high64(), sketch[1][index[1]].ID.low40(),
            sketch[0][index[0]].ID.high64(), sketch[0][index[0]].ID.low40()
        );

        __m512i cmpkey = _mm512_broadcast_i64x2(_mm_set_epi64x(item.high64(), item.low40()));
        __mmask8 cmpmask = _mm512_cmpeq_epi64_mask(vkey, cmpkey);

        uint8_t lanemask = 0b11;

        for(int i = 0; i < HASH_NUM; ++i) {
            if ((cmpmask & lanemask) == lanemask) {
                return sketch[i][index[i]].counter;
            }
            lanemask <<= 2;
        }
        return 0;
    }

    HashMap AllQuery() {
        HashMap ret;

        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            for(uint32_t j = 0; j < LENGTH; ++j) {
                if (sketch[i][j].ID[0] != '\0' && sketch[i][j].counter != 0 && ret.find(sketch[i][j].ID) == ret.end()) {
                    ret[sketch[i][j].ID] = Query(sketch[i][j].ID);
                }
            }
        }

        return ret;
    }

    uint32_t LENGTH;
    const uint32_t HASH_NUM = 4;
    const double DECAY_FACTOR = 1.1;
    Bucket** sketch;
};

#endif