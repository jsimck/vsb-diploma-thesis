#include "classifier_criteria.h"

ClassifierCriteria::ClassifierCriteria() {
    // ------- Train params (DEFAULTS) -------
    // Objectness
    trainParams.objectness.tEdgesMin = 0.01f;
    trainParams.objectness.tEdgesMax = 0.1f;

    // Hasher
    trainParams.hasher.grid = cv::Size(12, 12);
    trainParams.hasher.tablesCount = 100;
    trainParams.hasher.binCount = 5;
    trainParams.hasher.maxDistance = 3;

    // Matcher
    trainParams.matcher.pointsCount = 100;

    // ------- Detect params (DEFAULTS) -------
    // Objectness
    detectParams.objectness.step = 5;
    detectParams.objectness.tMatch = 0.3f;

    // Hasher
    detectParams.hasher.minVotes = 3;

    // Matcher
    detectParams.matcher.tMatch = 0.6f;
    detectParams.matcher.tOverlap = 0.1f;
    detectParams.matcher.neighbourhood = cv::Range(-2, 2); // 5x5 -> [-2, -1, 0, 1, 2]
    detectParams.matcher.tColorTest = 3;

    // Initialize info
    resetInfo();
}

void ClassifierCriteria::resetInfo() {
    info.smallestTemplate = cv::Size(500, 500); // There are no templates larger than 400x400
    info.maxTemplate = cv::Size(0, 0);
    info.minEdgels = INT_MAX;
}

void ClassifierCriteria::load(cv::FileStorage fsr, std::shared_ptr<ClassifierCriteria> criteria) {
    cv::FileNode criteriaNode = fsr["criteria"];
    cv::FileNode paramsNode = criteriaNode["trainParams"], infoNode = criteriaNode["info"];
    cv::FileNode hasherNode = paramsNode["hasher"], matcherNode = paramsNode["matcher"], objectnessNode = paramsNode["objectness"];

    hasherNode["grid" ] >> criteria->trainParams.hasher.grid;
    hasherNode["tablesCount"] >> criteria->trainParams.hasher.tablesCount;
    hasherNode["maxDistance"] >> criteria->trainParams.hasher.maxDistance;
    hasherNode["binCount"] >> criteria->trainParams.hasher.binCount;

    matcherNode["pointsCount"] >> criteria-> trainParams.matcher.pointsCount;

    objectnessNode["tEdgesMin"] >> criteria-> trainParams.objectness.tEdgesMin;
    objectnessNode["tEdgesMax"] >> criteria-> trainParams.objectness.tEdgesMax;

    infoNode["smallestTemplate"] >> criteria->info.smallestTemplate;
    infoNode["maxTemplate"] >> criteria->info.maxTemplate;
    infoNode["minEdgels"] >> criteria->info.minEdgels;
}

void ClassifierCriteria::save(cv::FileStorage &fsw) {
    fsw << "criteria" << "{";
        fsw << "trainParams" << "{";
            fsw << "hasher" << "{";
                fsw << "grid" << trainParams.hasher.grid;
                fsw << "tablesCount" << trainParams.hasher.tablesCount;
                fsw << "maxDistance" << trainParams.hasher.maxDistance;
                fsw << "binCount" << trainParams.hasher.binCount;
            fsw << "}";
            fsw << "matcher" << "{";
                fsw << "pointsCount" << trainParams.matcher.pointsCount;
            fsw << "}";
            fsw << "objectness" << "{";
                fsw << "tEdgesMin" << trainParams.objectness.tEdgesMin;
                fsw << "tEdgesMax" << trainParams.objectness.tEdgesMax;
            fsw << "}";
        fsw << "}";
        fsw << "info" << "{";
            fsw << "smallestTemplate" << info.smallestTemplate;
            fsw << "maxTemplate" << info.maxTemplate;
            fsw << "minEdgels" << info.minEdgels;
        fsw << "}";
    fsw << "}";
}

std::ostream &operator<<(std::ostream &os, const ClassifierCriteria &criteria) {
    os << "Params: " << std::endl;
    os << "  hasher: " << std::endl;
    os << "    |_ grid: " << criteria.trainParams.hasher.grid << std::endl;
    os << "    |_ minVotes: " << criteria.detectParams.hasher.minVotes << std::endl;
    os << "    |_ tablesCount: " << criteria.trainParams.hasher.tablesCount << std::endl;
    os << "    |_ maxDistance: " << criteria.trainParams.hasher.maxDistance << std::endl;
    os << "    |_ binCount: " << criteria.trainParams.hasher.binCount << std::endl;
    os << "  matcher: " << std::endl;
    os << "    |_ pointsCount: " << criteria.trainParams.matcher.pointsCount << std::endl;
    os << "    |_ tMatch: " << criteria.detectParams.matcher.tMatch << std::endl;
    os << "    |_ tOverlap: " << criteria.detectParams.matcher.tOverlap << std::endl;
    os << "    |_ tColorTest: " << criteria.detectParams.matcher.tColorTest << std::endl;
    os << "    |_ neighbourhood: (" << criteria.detectParams.matcher.neighbourhood.start << ", " << criteria.detectParams.matcher.neighbourhood.end << ")" << std::endl;
    os << "  objectness: " << std::endl;
    os << "    |_ step: " << criteria.detectParams.objectness.step << std::endl;
    os << "    |_ tEdgesMin: " << criteria.trainParams.objectness.tEdgesMin << std::endl;
    os << "    |_ tEdgesMax: " << criteria.trainParams.objectness.tEdgesMax << std::endl;
    os << "    |_ tMatch: " << criteria.detectParams.objectness.tMatch << std::endl << std::endl;
    os << "Info: " << std::endl;
    os << "  |_ minEdgels: " << criteria.info.minEdgels << std::endl;
    os << "  |_ smallestTemplate: " << criteria.info.smallestTemplate << std::endl;
    os << "  |_ maxTemplate: " << criteria.info.maxTemplate << std::endl;

    return os;
}
