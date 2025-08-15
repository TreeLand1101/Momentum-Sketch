#ifndef STABLESKETCH_H
#define STABLESKETCH_H

#include "Abstract.h"
#include <limits>

template<typename DATA_TYPE>
class StableSketch : public Abstract<DATA_TYPE> {
public:

    typedef std::unordered_map<DATA_TYPE, COUNT_TYPE> HashMap;

    struct Bucket{
        COUNT_TYPE stability;
        DATA_TYPE ID;
        COUNT_TYPE counter;
    };

    StableSketch(uint32_t _MEMORY, std::string _name = "StableSketch"){
        this->name = _name;

        LENGTH = _MEMORY / sizeof(Bucket) / HASH_NUM;
        sketch = new Bucket* [HASH_NUM];
        for(uint32_t i = 0; i < HASH_NUM; ++i){
            sketch[i] = new Bucket[LENGTH];
            memset(sketch[i], 0, sizeof(Bucket) * LENGTH);
        }
    }

    ~StableSketch(){
        for(uint32_t i = 0; i < HASH_NUM; ++i)
            delete [] sketch[i];
        delete [] sketch;
    }

    void Insert(const DATA_TYPE& item) {
        COUNT_TYPE min = std::numeric_limits<COUNT_TYPE>::max();
        int R = -1;
        int M = -1;

        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t pos = hash(item, i) % LENGTH;   
            if (sketch[i][pos].ID[0] == '\0' && sketch[i][pos].counter == 0) {
                sketch[i][pos].ID = item;
                sketch[i][pos].stability = 1;
                sketch[i][pos].counter = 1;
                return;
            }
            if (item == sketch[i][pos].ID) {
                sketch[i][pos].stability++;
                sketch[i][pos].counter++;
                return;
            }
            if (sketch[i][pos].counter < min) {
                min = sketch[i][pos].counter;
                R = i;
                M = pos;
            }
        }
        
        if (randomGenerator() % ((uint64_t)sketch[R][M].counter * sketch[R][M].stability + 1) == 0) {
            if (--sketch[R][M].counter == 0) {
                sketch[R][M].ID = item;
                sketch[R][M].counter = 1;
                if (sketch[R][M].stability > 0) {
                    sketch[R][M].stability--;
                }
            }
        }
    }

    COUNT_TYPE Query(const DATA_TYPE& item){
        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t pos = hash(item, i) % LENGTH;
            if (sketch[i][pos].ID == item) {
                return sketch[i][pos].counter;
            }
        }
        return 0;
    }

    HashMap AllQuery(){
        HashMap ret;

        for(uint32_t i = 0; i < HASH_NUM; ++i){
            for (uint32_t j = 0; j < LENGTH; ++j) {
                if (sketch[i][j].ID[0] != '\0' && sketch[i][j].counter != 0 && ret.find(sketch[i][j].ID) == ret.end()) {
                    ret[sketch[i][j].ID] = Query(sketch[i][j].ID);
                }
            }
        }

        return ret;
    }

    uint32_t LENGTH;
    const uint32_t HASH_NUM = 4;

    Bucket** sketch;
};

#endif