#ifndef TWOSTAGE_H
#define TWOSTAGE_H

#include "Abstract.h"
#include "Util.h"

#include "CUSketch.h"
#include "MVSketch.h"
#include "CocoSketch.h"
#include "Elastic.h"
#include "CMHeap.h"
#include "SpaceSaving.h"
#include "StableSketch.h"
#include "MomentumSketch.h"
#include "MomentumSketchSIMD.h"
#include "TightSketch.h"

#define Stage1FilterType CUSketch
#define Stage2SketchType MomentumSketch

template<typename DATA_TYPE>
class TwoStage : public Abstract<DATA_TYPE>{
public:
    typedef std::unordered_map<DATA_TYPE, COUNT_TYPE> HashMap;
    
    TwoStage(uint32_t _MEMORY, uint32_t _THRESHOLD, double _FILTER_RATIO = 0.7, double _SKETCH_RATIO = 0.3, double _ADMISSION_RATIO = 0.9, std::string _name = "TwoStage"){
        this->name = _name;
        uint32_t FILTER_MEMORY = _MEMORY * _FILTER_RATIO;
        uint32_t SKETCH_MEMORY = _MEMORY * _SKETCH_RATIO;
        ADMISSION_TRESHOLD = _THRESHOLD * _ADMISSION_RATIO;

        filter = new Stage1FilterType<DATA_TYPE, uint16_t>(FILTER_MEMORY, 2);
        sketch = new Stage2SketchType<DATA_TYPE>(SKETCH_MEMORY);

        // this->name += ("(" + filter->name + " + " + sketch->name + " )");
    }

    ~TwoStage(){
        delete filter;
        delete sketch;
    }

    void Insert(const DATA_TYPE& item){
        bool admitted = filter->InsertWithThreshold(item, ADMISSION_TRESHOLD);
        if (admitted) {
            sketch->Insert(item);
        }
    }

    COUNT_TYPE Query(const DATA_TYPE& item){
        COUNT_TYPE temp = filter->Query(item);
        if (temp >= ADMISSION_TRESHOLD) {
            return ADMISSION_TRESHOLD + sketch->Query(item);
        }
        return temp;
    }

    HashMap AllQuery(){
        HashMap res = sketch->AllQuery();
        for (auto& it:res) {
            it.second += ADMISSION_TRESHOLD;
        }
        return res;
    }
    
    COUNT_TYPE ADMISSION_TRESHOLD;

    Stage1FilterType<DATA_TYPE, uint16_t>* filter;
    Stage2SketchType<DATA_TYPE>* sketch;
};

#endif