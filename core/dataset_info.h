#ifndef VSB_SEMESTRAL_PROJECT_DATASETINFO_H
#define VSB_SEMESTRAL_PROJECT_DATASETINFO_H

#include <opencv2/core/types.hpp>
#include "template.h"

struct DatasetInfo {
public:
    int minEdgels;
    cv::Size smallestTemplateSize;
    cv::Size largestTemplateSize;

    // Constructors
    DatasetInfo();
    
    // Methods
    void reset();
};

#endif //VSB_SEMESTRAL_PROJECT_DATASETINFO_H
