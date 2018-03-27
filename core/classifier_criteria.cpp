#include <iostream>
#include "classifier_criteria.h"

namespace tless {
    std::ostream &operator<<(std::ostream &os, const ClassifierCriteria &crit) {
        os << "Params: " << std::endl;
        os << "  |_ tripletGrid: " << crit.tripletGrid.width << "x" << crit.tripletGrid.height << std::endl;
        os << "  |_ tablesCount: " << crit.tablesCount << std::endl;
        os << "  |_ depthBinCount: " << crit.depthBinCount << std::endl;
        os << "  |_ tablesTrainingMultiplier: " << crit.tablesTrainingMultiplier << std::endl;
        os << "  |_ featurePointsCount: " << crit.featurePointsCount << std::endl;
        os << "  |_ minMagnitude: " << crit.minMagnitude << std::endl;
        os << "  |_ maxDepthDiff: " << crit.maxDepthDiff << std::endl;
        os << "  |_ depthDeviation: " << crit.depthDeviation << std::endl;
        os << "  |_ minVotes: " << crit.minVotes << std::endl;
        os << "  |_ windowStep: " << crit.windowStep << std::endl;
        os << "  |_ patchOffset: " << crit.patchOffset << std::endl;
        os << "  |_ objectnessFactor: " << crit.objectnessFactor << std::endl;
        os << "  |_ matchFactor: " << crit.matchFactor << std::endl;
        os << "  |_ overlapFactor: " << crit.overlapFactor << std::endl;
        os << "  |_ depthK: " << crit.depthK << std::endl;
        os << "  |_ objectnessDiameterThreshold: " << crit.objectnessDiameterThreshold << std::endl;
        os << "Info: " << std::endl;
        os << "  |_ minDepth: " << crit.info.minDepth << std::endl;
        os << "  |_ maxDepth: " << crit.info.maxDepth << std::endl;
        os << "  |_ smallestDiameter: " << crit.info.smallestDiameter << std::endl;
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
        node["depthDeviation"] >> crit->depthDeviation;
        node["objectnessDiameterThreshold"] >> crit->objectnessDiameterThreshold;

        // Load unsigned int params
        int tablesCount, featurePointsCount, depthBinCount, tablesTrainingMultiplier;
        node["tablesCount"] >> tablesCount;
        node["depthBinCount"] >> depthBinCount;
        node["tablesTrainingMultiplier"] >> tablesTrainingMultiplier;
        node["featurePointsCount"] >> featurePointsCount;
        crit->tablesCount = static_cast<uint>(tablesCount);
        crit->depthBinCount = static_cast<uint>(depthBinCount);
        crit->tablesTrainingMultiplier = static_cast<uint>(tablesTrainingMultiplier);
        crit->featurePointsCount = static_cast<uint>(featurePointsCount);

        cv::FileNode info = node["info"];
        info["depthScaleFactor"] >> crit->info.depthScaleFactor;
        info["smallestTemplate"] >> crit->info.smallestTemplate;
        info["largestArea"] >> crit->info.largestArea;
        info["minEdgels"] >> crit->info.minEdgels;
        info["minDepth"] >> crit->info.minDepth;
        info["maxDepth"] >> crit->info.maxDepth;
        info["smallestDiameter"] >> crit->info.smallestDiameter;
    }

    cv::FileStorage &operator<<(cv::FileStorage &fs, const ClassifierCriteria &crit) {
        fs << "{";
        fs << "tripletGrid" << crit.tripletGrid;
        fs << "tablesCount" << static_cast<int>(crit.tablesCount);
        fs << "depthBinCount" << static_cast<int>(crit.depthBinCount);
        fs << "tablesTrainingMultiplier" << static_cast<int>(crit.tablesTrainingMultiplier);
        fs << "featurePointsCount" << static_cast<int>(crit.featurePointsCount);
        fs << "minMagnitude" << crit.minMagnitude;
        fs << "maxDepthDiff" << crit.maxDepthDiff;
        fs << "depthDeviation" << crit.depthDeviation;
        fs << "objectnessDiameterThreshold" << crit.objectnessDiameterThreshold;
        fs << "info" << "{";
        fs << "depthScaleFactor" << crit.info.depthScaleFactor;
        fs << "smallestTemplate" << crit.info.smallestTemplate;
        fs << "largestArea" << crit.info.largestArea;
        fs << "minEdgels" << crit.info.minEdgels;
        fs << "minDepth" << crit.info.minDepth;
        fs << "maxDepth" << crit.info.maxDepth;
        fs << "smallestDiameter" << crit.info.smallestDiameter;
        fs << "}";
        fs << "}";

        return fs;
    }
}