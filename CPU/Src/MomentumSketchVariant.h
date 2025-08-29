#ifndef MOMENTUMSKETCHVARIANT_H
#define MOMENTUMSKETCHVARIANT_H

#include "Abstract.h"
#include "MomentumSketch.h"
#include <bit>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <limits>

template<typename DATA_TYPE>
class Default : public MomentumSketch<DATA_TYPE> {
public:
    using MomentumSketch<DATA_TYPE>::LENGTH;
    using MomentumSketch<DATA_TYPE>::HASH_NUM;
    using MomentumSketch<DATA_TYPE>::sketch;
    using MomentumSketch<DATA_TYPE>::DECAY_FACTOR;

    Default(uint32_t _MEMORY, std::string _name = "Default") : MomentumSketch<DATA_TYPE>(_MEMORY, _name) {}
};

template<typename DATA_TYPE>
class Additive : public MomentumSketch<DATA_TYPE> {
public:
    using MomentumSketch<DATA_TYPE>::LENGTH;
    using MomentumSketch<DATA_TYPE>::HASH_NUM;
    using MomentumSketch<DATA_TYPE>::sketch;
    using MomentumSketch<DATA_TYPE>::DECAY_FACTOR;

    Additive(uint32_t _MEMORY, std::string _name = "Additive") : MomentumSketch<DATA_TYPE>(_MEMORY, _name) {}

    void Insert(const DATA_TYPE& item) {
        COUNT_TYPE min = std::numeric_limits<COUNT_TYPE>::max();
        int R = -1;
        int P = -1;

        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t pos = hash(item, i) % LENGTH;   
            if (sketch[i][pos].ID[0] == '\0' && sketch[i][pos].counter == 0) {
                sketch[i][pos].ID = item;
                sketch[i][pos].counter = 1;
                sketch[i][pos].momentum = 1;
                return;
            }
            if (item == sketch[i][pos].ID) {
                if (std::numeric_limits<COUNT_TYPE>::max() - sketch[i][pos].momentum < sketch[i][pos].counter) {
                    sketch[i][pos].momentum = std::numeric_limits<COUNT_TYPE>::max();
                } 
                else {
                    sketch[i][pos].momentum += sketch[i][pos].counter;
                }
                sketch[i][pos].counter++;
                return;
            }
            if (sketch[i][pos].counter < min) {
                min = sketch[i][pos].counter;
                R = i;
                P = pos;
            }
        }

        sketch[R][P].momentum /= DECAY_FACTOR;

        if (randomGenerator() % ((uint64_t)sketch[R][P].counter + sketch[R][P].momentum + 1) == 0) {
            if (--sketch[R][P].counter == 0) {
                sketch[R][P].ID = item;
                sketch[R][P].counter = 1;
                sketch[R][P].momentum = 1;
            }
        }
    }

};


template<typename DATA_TYPE>
class MomentumOnly : public MomentumSketch<DATA_TYPE> {
public:
    using MomentumSketch<DATA_TYPE>::LENGTH;
    using MomentumSketch<DATA_TYPE>::HASH_NUM;
    using MomentumSketch<DATA_TYPE>::sketch;
    using MomentumSketch<DATA_TYPE>::DECAY_FACTOR;

    MomentumOnly(uint32_t _MEMORY, std::string _name = "Momentum-Only") : MomentumSketch<DATA_TYPE>(_MEMORY, _name) {}

    void Insert(const DATA_TYPE& item) {
        COUNT_TYPE min = std::numeric_limits<COUNT_TYPE>::max();
        int R = -1;
        int P = -1;

        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t pos = hash(item, i) % LENGTH;   
            if (sketch[i][pos].ID[0] == '\0' && sketch[i][pos].counter == 0) {
                sketch[i][pos].ID = item;
                sketch[i][pos].counter = 1;
                sketch[i][pos].momentum = 1;
                return;
            }
            if (item == sketch[i][pos].ID) {
                if (std::numeric_limits<COUNT_TYPE>::max() - sketch[i][pos].momentum < sketch[i][pos].counter) {
                    sketch[i][pos].momentum = std::numeric_limits<COUNT_TYPE>::max();
                } 
                else {
                    sketch[i][pos].momentum += sketch[i][pos].counter;
                }
                sketch[i][pos].counter++;
                return;
            }
            if (sketch[i][pos].counter < min) {
                min = sketch[i][pos].counter;
                R = i;
                P = pos;
            }
        }

        sketch[R][P].momentum /= DECAY_FACTOR;

        if (randomGenerator() % ((uint64_t)sketch[R][P].momentum + 1) == 0) {
            if (--sketch[R][P].counter == 0) {
                sketch[R][P].ID = item;
                sketch[R][P].counter = 1;
                sketch[R][P].momentum = 1;
            }
        }
    }

};


template<typename DATA_TYPE>
class CounterOnly : public MomentumSketch<DATA_TYPE> {
public:
    using MomentumSketch<DATA_TYPE>::LENGTH;
    using MomentumSketch<DATA_TYPE>::HASH_NUM;
    using MomentumSketch<DATA_TYPE>::sketch;
    using MomentumSketch<DATA_TYPE>::DECAY_FACTOR;

    CounterOnly(uint32_t _MEMORY, std::string _name = "Counter-Only") : MomentumSketch<DATA_TYPE>(_MEMORY, _name) {}

    void Insert(const DATA_TYPE& item) {
        COUNT_TYPE min = std::numeric_limits<COUNT_TYPE>::max();
        int R = -1;
        int P = -1;

        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t pos = hash(item, i) % LENGTH;   
            if (sketch[i][pos].ID[0] == '\0' && sketch[i][pos].counter == 0) {
                sketch[i][pos].ID = item;
                sketch[i][pos].counter = 1;
                sketch[i][pos].momentum = 1;
                return;
            }
            if (item == sketch[i][pos].ID) {
                if (std::numeric_limits<COUNT_TYPE>::max() - sketch[i][pos].momentum < sketch[i][pos].counter) {
                    sketch[i][pos].momentum = std::numeric_limits<COUNT_TYPE>::max();
                } 
                else {
                    sketch[i][pos].momentum += sketch[i][pos].counter;
                }
                sketch[i][pos].counter++;
                return;
            }
            if (sketch[i][pos].counter < min) {
                min = sketch[i][pos].counter;
                R = i;
                P = pos;
            }
        }

        sketch[R][P].momentum /= DECAY_FACTOR;

        if (randomGenerator() % ((uint64_t)sketch[R][P].counter + 1) == 0) {
            if (--sketch[R][P].counter == 0) {
                sketch[R][P].ID = item;
                sketch[R][P].counter = 1;
                sketch[R][P].momentum = 1;
            }
        }
    }

};

template<typename DATA_TYPE>
class MomentumSketchDecayFactor : public MomentumSketch<DATA_TYPE> {
public:
    using MomentumSketch<DATA_TYPE>::LENGTH;
    using MomentumSketch<DATA_TYPE>::HASH_NUM;
    using MomentumSketch<DATA_TYPE>::sketch;

    MomentumSketchDecayFactor(uint32_t _MEMORY,
                      std::string _name = /* std::string("df = ") + */ std::to_string(decay_factor_))
        : MomentumSketch<DATA_TYPE>(_MEMORY, _name) {}

    static void SetDecayFactor(double df) { decay_factor_ = df; }

    void Insert(const DATA_TYPE& item) {
        COUNT_TYPE min = std::numeric_limits<COUNT_TYPE>::max();
        int R = -1;
        int P = -1;

        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t pos = hash(item, i) % LENGTH;   
            if (sketch[i][pos].ID[0] == '\0' && sketch[i][pos].counter == 0) {
                sketch[i][pos].ID = item;
                sketch[i][pos].counter = 1;
                sketch[i][pos].momentum = 1;
                return;
            }
            if (item == sketch[i][pos].ID) {
                if (std::numeric_limits<COUNT_TYPE>::max() - sketch[i][pos].momentum < sketch[i][pos].counter) {
                    sketch[i][pos].momentum = std::numeric_limits<COUNT_TYPE>::max();
                } 
                else {
                    sketch[i][pos].momentum += sketch[i][pos].counter;
                }
                sketch[i][pos].counter++;
                return;
            }
            if (sketch[i][pos].counter < min) {
                min = sketch[i][pos].counter;
                R = i;
                P = pos;
            }
        }

        sketch[R][P].momentum /= decay_factor_;

        if (randomGenerator() % ((uint64_t)sketch[R][P].counter * sketch[R][P].momentum + 1) == 0) {
            if (--sketch[R][P].counter == 0) {
                sketch[R][P].ID = item;
                sketch[R][P].counter = 1;
                sketch[R][P].momentum = 1;
            }
        }
    }

    static double decay_factor_;
};

template<typename DATA_TYPE>
double MomentumSketchDecayFactor<DATA_TYPE>::decay_factor_ = 2;

#endif