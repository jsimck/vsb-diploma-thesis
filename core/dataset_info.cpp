#include "dataset_info.h"

DataSetInfo::DataSetInfo() {
    reset();
}

void DataSetInfo::reset() {
    this->smallestTemplate = cv::Size(500, 500); // There are no templates larger than 400x400
    this->maxTemplate = cv::Size(0, 0);
    this->minEdgels = INT_MAX;
}

std::ostream &operator<<(std::ostream &os, const DataSetInfo &info) {
    os << "minEdgels: " << info.minEdgels << " smallestTemplate: " << info.smallestTemplate << " maxTemplate: "
       << info.maxTemplate;
    return os;
}
