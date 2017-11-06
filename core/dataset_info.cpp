#include "dataset_info.h"

DataSetInfo::DataSetInfo() {
    reset();
}

void DataSetInfo::reset() {
    smallestTemplate = cv::Size(500, 500); // There are no templates larger than 400x400
    maxTemplate = cv::Size(0, 0);
    minEdgels = INT_MAX;
}

DataSetInfo DataSetInfo::load(cv::FileStorage node) {
    DataSetInfo dataSet;

    node["smallest_template"] >> dataSet.smallestTemplate;
    node["max_template"] >> dataSet.maxTemplate;
    node["min_edgels"] >> dataSet.minEdgels;

    return dataSet;
}

void DataSetInfo::save(cv::FileStorage &fs) {
    fs << "smallest_template" << smallestTemplate;
    fs << "max_template" << maxTemplate;
    fs << "min_edgels" << minEdgels;
}

std::ostream &operator<<(std::ostream &os, const DataSetInfo &info) {
    os << "minEdgels: " << info.minEdgels << " smallestTemplate: " << info.smallestTemplate << " maxTemplate: "
       << info.maxTemplate;
    return os;
}