#include <iostream>
#include "classifier_criteria.h"

namespace tless {
    std::ostream &operator<<(std::ostream &os, const ClassifierCriteria &crit) {
        os << "Params: " << std::endl;
        os << "  |_ tripletGrid: " << crit.tripletGrid.width << "x" << crit.tripletGrid.height << std::endl;
        os << "  |_ maxTripletDist: " << crit.maxTripletDist << std::endl;
        os << "  |_ tablesCount: " << crit.tablesCount << std::endl;
        os << "  |_ featurePointsCount: " << crit.featurePointsCount << std::endl;
        os << "  |_ minMagnitude: " << crit.minMagnitude << std::endl;
        os << "  |_ maxDepthDiff: " << crit.maxDepthDiff << std::endl;
        os << "  |_ depthDeviationFun (size): " << crit.depthDeviationFun.size() << std::endl;
        os << "  |_ minVotes: " << crit.minVotes << std::endl;
        os << "  |_ windowStep: " << crit.windowStep << std::endl;
        os << "  |_ patchOffset: " << crit.patchOffset << std::endl;
        os << "  |_ objectnessFactor: " << crit.objectnessFactor << std::endl;
        os << "  |_ matchFactor: " << crit.matchFactor << std::endl;
        os << "  |_ overlapFactor: " << crit.overlapFactor << std::endl;
        os << "  |_ depthK: " << crit.depthK << std::endl;
        os << "Info: " << std::endl;
        os << "  |_ minDepth: " << crit.info.minDepth << std::endl;
        os << "  |_ maxDepth: " << crit.info.maxDepth << std::endl;
        os << "  |_ minEdgels: " << crit.info.minEdgels << std::endl;
        os << "  |_ depthScaleFactor: " << crit.info.depthScaleFactor << std::endl;
        os << "  |_ smallestTemplate: " << crit.info.smallestTemplate.width << "x" << crit.info.smallestTemplate.height
           << std::endl;
        os << "  |_ largestArea: " << crit.info.largestArea.width << "x" << crit.info.largestArea.height
           << std::endl;

        return os;
    }

    void operator>>(const cv::FileNode &node, cv::Ptr<ClassifierCriteria> crit) {
        node["tripletGrid"] >> crit->tripletGrid;
        node["minMagnitude"] >> crit->minMagnitude;
        node["maxDepthDiff"] >> crit->maxDepthDiff;
        node["depthDeviationFun"] >> crit->depthDeviationFun;

        // Load unsigned int params
        int maxTripletDist, tablesCount, featurePointsCount;
        node["maxTripletDist"] >> maxTripletDist;
        node["tablesCount"] >> tablesCount;
        node["featurePointsCount"] >> featurePointsCount;
        crit->maxTripletDist = static_cast<uint>(maxTripletDist);
        crit->tablesCount = static_cast<uint>(tablesCount);
        crit->featurePointsCount = static_cast<uint>(featurePointsCount);

        cv::FileNode info = node["info"];
        info["depthScaleFactor"] >> crit->info.depthScaleFactor;
        info["smallestTemplate"] >> crit->info.smallestTemplate;
        info["largestArea"] >> crit->info.largestArea;
        info["minEdgels"] >> crit->info.minEdgels;
        info["minDepth"] >> crit->info.minDepth;
        info["maxDepth"] >> crit->info.maxDepth;
    }

    cv::FileStorage &operator<<(cv::FileStorage &fs, const ClassifierCriteria &crit) {
        fs << "{";
        fs << "tripletGrid" << crit.tripletGrid;
        fs << "maxTripletDist" << static_cast<int>(crit.maxTripletDist);
        fs << "tablesCount" << static_cast<int>(crit.tablesCount);
        fs << "featurePointsCount" << static_cast<int>(crit.featurePointsCount);
        fs << "minMagnitude" << crit.minMagnitude;
        fs << "maxDepthDiff" << crit.maxDepthDiff;
        fs << "depthDeviationFun" << crit.depthDeviationFun;
        fs << "info" << "{";
        fs << "depthScaleFactor" << crit.info.depthScaleFactor;
        fs << "smallestTemplate" << crit.info.smallestTemplate;
        fs << "largestArea" << crit.info.largestArea;
        fs << "minEdgels" << crit.info.minEdgels;
        fs << "minDepth" << crit.info.minDepth;
        fs << "maxDepth" << crit.info.maxDepth;
        fs << "}";
        fs << "}";

        return fs;
    }
}