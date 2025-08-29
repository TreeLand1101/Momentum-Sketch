#ifndef CUSKETCH_H
#define CUSKETCH_H

#include "Util.h"

template<typename DATA_TYPE,typename COUNT_TYPE>
class CUSketch{
public:
    std::string name = "CU Sketch";   

    CUSketch(uint32_t _MEMORY, uint32_t _HASH_NUM = 4) {
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

    ~CUSketch(){
        for(uint32_t i = 0;i < HASH_NUM;++i)
            delete [] sketch[i];
        delete [] sketch;
        delete [] index;
    }

    COUNT_TYPE Insert(const DATA_TYPE& item) {
        COUNT_TYPE minVal = std::numeric_limits<COUNT_TYPE>::max();

        for (uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t position = hash(item, i) % LENGTH;
            index[i] = position;
            minVal = std::min(minVal, sketch[i][position]);
        }

        COUNT_TYPE ret = minVal + 1;

        for (uint32_t i = 0; i < HASH_NUM; ++i) {
            sketch[i][index[i]] = std::max(sketch[i][index[i]], ret);
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

        COUNT_TYPE updateVal = minVal + 1;

        for (uint32_t i = 0; i < HASH_NUM; ++i) {
            sketch[i][index[i]] = std::max(sketch[i][index[i]], updateVal);
        }

        return false;
    }

    COUNT_TYPE Query(const DATA_TYPE& item){
        COUNT_TYPE ret = std::numeric_limits<COUNT_TYPE>::max();

        for (uint32_t i = 0; i < HASH_NUM; ++i) {
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
