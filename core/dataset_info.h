#ifndef VSB_SEMESTRAL_PROJECT_DATASETINFO_H
#define VSB_SEMESTRAL_PROJECT_DATASETINFO_H

#include <opencv2/core/types.hpp>
#include <ostream>
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

    friend std::ostream &operator<<(std::ostream &os, const DataSetInfo &info);
};

#endif //VSB_SEMESTRAL_PROJECT_DATASETINFO_H
