#ifndef VSB_SEMESTRAL_PROJECT_DATASETINFO_H
#define VSB_SEMESTRAL_PROJECT_DATASETINFO_H

#include <opencv2/core/types.hpp>
#include "template.h"

/**
 * struct DataSetInfo
 *
 * Holds information about loaded data set for further computation.
 */
struct DataSetInfo {
public:
    int minEdgels;
    cv::Size smallestTemplate;
    cv::Size maxTemplate;

    // Constructors
    DataSetInfo();
    
    // Methods
    void reset();
};

#endif //VSB_SEMESTRAL_PROJECT_DATASETINFO_H
