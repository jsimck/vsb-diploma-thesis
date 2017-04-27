#include "dataset_info.h"

DataSetInfo::DataSetInfo() {
    reset();
}

void DataSetInfo::reset() {
    this->smallestTemplate = cv::Size(500, 500); // There are no templates larger than 400x400
    this->maxTemplate = cv::Size(0, 0);
    this->minEdgels = INT_MAX;
}
