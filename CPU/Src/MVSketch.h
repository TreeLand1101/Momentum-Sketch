#ifndef MVSKETCH_H
#define MVSKETCH_H

#include "Abstract.h"
#include <limits>

template<typename DATA_TYPE>
class MVSketch : public Abstract<DATA_TYPE> {
public:

    typedef std::unordered_map<DATA_TYPE, COUNT_TYPE> HashMap;

    struct Bucket{
        COUNT_TYPE total_sum;
        DATA_TYPE ID;
        COUNT_TYPE counter;
    };

    MVSketch(uint32_t _MEMORY, std::string _name = "MVSketch"){
        this->name = _name;

        LENGTH = _MEMORY / sizeof(Bucket) / HASH_NUM;
        sketch = new Bucket* [HASH_NUM];
        for(uint32_t i = 0; i < HASH_NUM; ++i){
            sketch[i] = new Bucket[LENGTH];
            memset(sketch[i], 0, sizeof(Bucket) * LENGTH);
        }
    }

    ~MVSketch(){
        for(uint32_t i = 0; i < HASH_NUM; ++i)
            delete [] sketch[i];
        delete [] sketch;
    }

    void Insert(const DATA_TYPE& item) {
        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t pos = hash(item, i) % LENGTH;    
            sketch[i][pos].total_sum++;
            if (item == sketch[i][pos].ID) {
                sketch[i][pos].counter++;
            }
            else {
                if (sketch[i][pos].counter == 0) {
                    sketch[i][pos].ID = item;
                    sketch[i][pos].counter = 1;                    
                }
                else {
                    --sketch[i][pos].counter;
                }
            }
        }
    }

    COUNT_TYPE Query(const DATA_TYPE& item){
        COUNT_TYPE ret = std::numeric_limits<COUNT_TYPE>::max();

        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t pos = hash(item, i) % LENGTH;    
            if (sketch[i][pos].ID == item) {
                ret = std::min(ret, (sketch[i][pos].total_sum + sketch[i][pos].counter) / 2);
            }
            else {
                ret = std::min(ret, (sketch[i][pos].total_sum - sketch[i][pos].counter) / 2);
            }
        }

        return ret;
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