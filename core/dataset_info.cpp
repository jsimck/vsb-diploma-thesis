#include "dataset_info.h"

DatasetInfo::DatasetInfo() {
    reset();
}

void DatasetInfo::reset() {
    this->smallestTemplateSize = cv::Size(500, 500); // There are no templates larger than 400x400
    this->maxTemplateSize = cv::Size(0, 0);
    this->minEdgels = INT_MAX;
}
