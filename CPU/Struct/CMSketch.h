#ifndef CMSKETCH_H
#define CMSKETCH_H

#include "Util.h"

template<typename DATA_TYPE,typename COUNT_TYPE>
class CMSketch{
public:
    std::string name = "CM Sketch";

    CMSketch(uint32_t _MEMORY, uint32_t _HASH_NUM = 4) {
        HASH_NUM = _HASH_NUM;
        name += " (d = " + std::to_string(_HASH_NUM) + ")";
        LENGTH = _MEMORY / sizeof(COUNT_TYPE) / HASH_NUM;
        index = new uint32_t [HASH_NUM];
        sketch = new COUNT_TYPE* [HASH_NUM];
        for(uint32_t i = 0;i < HASH_NUM; ++i){
            sketch[i] = new COUNT_TYPE[LENGTH];
            memset(sketch[i], 0, sizeof(COUNT_TYPE) * LENGTH);
        }
    }

    ~CMSketch(){
        for(uint32_t i = 0;i < HASH_NUM;++i)
            delete [] sketch[i];
        delete [] sketch;
        delete [] index;
    }

    COUNT_TYPE Insert(const DATA_TYPE& item) {
        COUNT_TYPE ret = std::numeric_limits<COUNT_TYPE>::max();

        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t position = hash(item, i) % LENGTH;
            sketch[i][position] += 1;
            ret = std::min(ret, sketch[i][position]);
        }

        return ret;
    }

    bool InsertWithThreshold(const DATA_TYPE& item, COUNT_TYPE ADMISSION_TRESHOLD) {
        COUNT_TYPE minVal = std::numeric_limits<COUNT_TYPE>::max();

        for (uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t position = hash(item, i) % LENGTH;
            index[i] = position;
            minVal = std::min(minVal, sketch[i][position]);
        }

        if (minVal >= ADMISSION_TRESHOLD) {
            return true;
        }

        for (uint32_t i = 0; i < HASH_NUM; ++i) {
            sketch[i][index[i]]++;
        }

        return false;
    }

    COUNT_TYPE Query(const DATA_TYPE& item){
        COUNT_TYPE ret = std::numeric_limits<COUNT_TYPE>::max();

        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t position = hash(item, i) % LENGTH;
            ret = std::min(ret, sketch[i][position]);
        }

        return ret;
    }

private:
    uint32_t LENGTH;
    uint32_t HASH_NUM;
    COUNT_TYPE** sketch;
    uint32_t* index;
};

#endif
