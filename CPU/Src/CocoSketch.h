#ifndef COCOSKETCH_H
#define COCOSKETCH_H

#include "Abstract.h"
#include <limits>

template<typename DATA_TYPE>
class CocoSketch : public Abstract<DATA_TYPE>{
public:

    struct Counter{
        DATA_TYPE ID;
        COUNT_TYPE count;
    };

    typedef std::unordered_map<DATA_TYPE, COUNT_TYPE> HashMap;

    CocoSketch(uint32_t _MEMORY, uint32_t _HASH_NUM = 2, std::string _name = "CocoSketch"){
        this->name = _name;

        HASH_NUM = _HASH_NUM;
        LENGTH = _MEMORY / _HASH_NUM / sizeof(Counter);

        counter = new Counter* [HASH_NUM];
        for(uint32_t i = 0;i < HASH_NUM;++i){
            counter[i] = new Counter [LENGTH];
            memset(counter[i], 0, sizeof(Counter) * LENGTH);
        }
    }

    ~CocoSketch(){
        for(uint32_t i = 0;i < HASH_NUM;++i){
            delete [] counter[i];
        }
        delete [] counter;
    }

    void Insert(const DATA_TYPE& item){
        COUNT_TYPE minimum = std::numeric_limits<COUNT_TYPE>::max();
        uint32_t minPos, minHash;

        for(uint32_t i = 0;i < HASH_NUM;++i){
            uint32_t position = hash(item, i) % LENGTH;
            if(counter[i][position].ID == item){
                counter[i][position].count += 1;
                return;
            }
            if(counter[i][position].count < minimum){
                minPos = position;
                minHash = i;
                minimum = counter[i][position].count;
            }
        }

        counter[minHash][minPos].count += 1;
        if(randomGenerator() % counter[minHash][minPos].count == 0){
            counter[minHash][minPos].ID = item;
        }
    }

    COUNT_TYPE Query(const DATA_TYPE& item){
        for(uint32_t i = 0; i < HASH_NUM; ++i) {
            uint32_t position = hash(item, i) % LENGTH;
            if(counter[i][position].ID == item){
                return counter[i][position].count;
            }
        }
        return 0;
    }

    HashMap AllQuery(){
        HashMap ret;

        for(uint32_t i = 0;i < HASH_NUM;++i){
            for(uint32_t j = 0;j < LENGTH;++j){
                ret[counter[i][j].ID] = counter[i][j].count;
            }
        }

        return ret;
    }

    uint32_t LENGTH;
    uint32_t HASH_NUM;

    Counter** counter;
};

#endif