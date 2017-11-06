#include "dataset_info.h"

DataSetInfo::DataSetInfo() {
    reset();
}

void DataSetInfo::reset() {
    smallestTemplate = cv::Size(500, 500); // There are no templates larger than 400x400
    maxTemplate = cv::Size(0, 0);
    minEdgels = INT_MAX;
}

DataSetInfo DataSetInfo::load(cv::FileStorage fsr) {
    DataSetInfo dataSet;

    fsr["smallestTemplate"] >> dataSet.smallestTemplate;
    fsr["maxTemplate"] >> dataSet.maxTemplate;
    fsr["minEdgels"] >> dataSet.minEdgels;

    return dataSet;
}

void DataSetInfo::save(cv::FileStorage &fsw) {
    fsw << "smallestTemplate" << smallestTemplate;
    fsw << "maxTemplate" << maxTemplate;
    fsw << "minEdgels" << minEdgels;
}

std::ostream &operator<<(std::ostream &os, const DataSetInfo &info) {
    os << "minEdgels: " << info.minEdgels << " smallestTemplate: " << info.smallestTemplate << " maxTemplate: "
       << info.maxTemplate;
    return os;
}