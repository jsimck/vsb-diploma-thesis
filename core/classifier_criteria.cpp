#include "classifier_criteria.h"

ClassifierCriteria::ClassifierCriteria() {
    // ------- Train params (DEFAULTS) -------
    // Objectness
    train.objectness.tEdgesMin = 0.01f;
    train.objectness.tEdgesMax = 0.1f;

    // Hasher
    train.hasher.grid = cv::Size(12, 12);
    train.hasher.tablesCount = 100;
    train.hasher.binCount = 5;
    train.hasher.maxDistance = 3;

    // Matcher
    train.matcher.pointsCount = 100;

    // ------- Detect params (DEFAULTS) -------
    // Objectness
    detect.objectness.step = 5;
    detect.objectness.tMatch = 0.3f;

    // Hasher
    detect.hasher.minVotes = 3;

    // Matcher
    detect.matcher.tMatch = 0.6f;
    detect.matcher.tOverlap = 0.1f;
    detect.matcher.neighbourhood = cv::Range(-2, 2); // 5x5 -> [-2, -1, 0, 1, 2]
    detect.matcher.tColorTest = 3;
    detect.matcher.depthDeviationFunction = {{10000, 0.14f}, {15000, 0.12f}, {20000, 0.1f}, {70000, 0.08f}};
    detect.matcher.depthK = 0.05f;
    detect.matcher.tMinGradMag = 0.1f;
    detect.matcher.maxDifference = 100;

    // Initialize info
    info.depthScaleFactor = 10.0f;
    info.smallestTemplate = cv::Size(500, 500); // There are no templates larger than 400x400
    info.largestTemplate = cv::Size(0, 0);
    info.minEdgels = INT_MAX;
    info.maxDepth = 0;
}

void ClassifierCriteria::load(cv::FileStorage fsr, std::shared_ptr<ClassifierCriteria> criteria) {
    cv::FileNode criteriaNode = fsr["criteria"];
    cv::FileNode paramsNode = criteriaNode["train"], infoNode = criteriaNode["info"];
    cv::FileNode hasherNode = paramsNode["hasher"], matcherNode = paramsNode["matcher"], objectnessNode = paramsNode["objectness"];

    hasherNode["grid" ] >> criteria->train.hasher.grid;
    hasherNode["tablesCount"] >> criteria->train.hasher.tablesCount;
    hasherNode["maxDistance"] >> criteria->train.hasher.maxDistance;
    hasherNode["binCount"] >> criteria->train.hasher.binCount;

    matcherNode["pointsCount"] >> criteria-> train.matcher.pointsCount;

    objectnessNode["tEdgesMin"] >> criteria-> train.objectness.tEdgesMin;
    objectnessNode["tEdgesMax"] >> criteria-> train.objectness.tEdgesMax;

    infoNode["depthScaleFactor"] >> criteria->info.depthScaleFactor;
    infoNode["smallestTemplate"] >> criteria->info.smallestTemplate;
    infoNode["largestTemplate"] >> criteria->info.largestTemplate;
    infoNode["minEdgels"] >> criteria->info.minEdgels;
    infoNode["maxDepth"] >> criteria->info.maxDepth;
}

void ClassifierCriteria::save(cv::FileStorage &fsw) {
    fsw << "criteria" << "{";
        fsw << "train" << "{";
            fsw << "hasher" << "{";
                fsw << "grid" << train.hasher.grid;
                fsw << "tablesCount" << train.hasher.tablesCount;
                fsw << "maxDistance" << train.hasher.maxDistance;
                fsw << "binCount" << train.hasher.binCount;
            fsw << "}";
            fsw << "matcher" << "{";
                fsw << "pointsCount" << train.matcher.pointsCount;
            fsw << "}";
            fsw << "objectness" << "{";
                fsw << "tEdgesMin" << train.objectness.tEdgesMin;
                fsw << "tEdgesMax" << train.objectness.tEdgesMax;
            fsw << "}";
        fsw << "}";
        fsw << "info" << "{";
            fsw << "depthScaleFactor" << info.depthScaleFactor;
            fsw << "smallestTemplate" << info.smallestTemplate;
            fsw << "largestTemplate" << info.largestTemplate;
            fsw << "minEdgels" << info.minEdgels;
            fsw << "maxDepth" << info.maxDepth;
        fsw << "}";
    fsw << "}";
}

std::ostream &operator<<(std::ostream &os, const ClassifierCriteria &criteria) {
    os << "Params: " << std::endl;
    os << "  hasher: " << std::endl;
    os << "    |_ grid: " << criteria.train.hasher.grid << std::endl;
    os << "    |_ minVotes: " << criteria.detect.hasher.minVotes << std::endl;
    os << "    |_ tablesCount: " << criteria.train.hasher.tablesCount << std::endl;
    os << "    |_ maxDistance: " << criteria.train.hasher.maxDistance << std::endl;
    os << "    |_ binCount: " << criteria.train.hasher.binCount << std::endl;
    os << "  matcher: " << std::endl;
    os << "    |_ pointsCount: " << criteria.train.matcher.pointsCount << std::endl;
    os << "    |_ tMatch: " << criteria.detect.matcher.tMatch << std::endl;
    os << "    |_ tOverlap: " << criteria.detect.matcher.tOverlap << std::endl;
    os << "    |_ tColorTest: " << criteria.detect.matcher.tColorTest << std::endl;
    os << "    |_ neighbourhood: (" << criteria.detect.matcher.neighbourhood.start << ", " << criteria.detect.matcher.neighbourhood.end << ")" << std::endl;
    os << "  objectness: " << std::endl;
    os << "    |_ step: " << criteria.detect.objectness.step << std::endl;
    os << "    |_ tEdgesMin: " << criteria.train.objectness.tEdgesMin << std::endl;
    os << "    |_ tEdgesMax: " << criteria.train.objectness.tEdgesMax << std::endl;
    os << "    |_ tMatch: " << criteria.detect.objectness.tMatch << std::endl << std::endl;
    os << "Info: " << std::endl;
    os << "  |_ depthScaleFactor: " << criteria.info.depthScaleFactor << std::endl;
    os << "  |_ smallestTemplate: " << criteria.info.smallestTemplate << std::endl;
    os << "  |_ minEdgels: " << criteria.info.minEdgels << std::endl;
    os << "  |_ largestTemplate: " << criteria.info.largestTemplate << std::endl;
    os << "  |_ maxDepth: " << criteria.info.maxDepth << std::endl;

    return os;
}
