#include "classifier_terms.h"

ClassifierTerms::ClassifierTerms() {
    // Init objectness params
    params.objectness.step = 5;
    params.objectness.tEdgesMin = 0.01f;
    params.objectness.tEdgesMax = 0.1f;
    params.objectness.tMatch = 0.3f;

    // Init hasher params
    params.hasher.grid = cv::Size(12, 12);
    params.hasher.tablesCount = 100;
    params.hasher.binCount = 5;
    params.hasher.minVotes = 3;
    params.hasher.maxDistance = 3;

    // Init template matcher
    params.matcher.pointsCount = 100;
    params.matcher.tMatch = 0.6f;
    params.matcher.tOverlap = 0.1f;
    params.matcher.neighbourhood = cv::Range(-2, 2); // 5x5 -> [-2, -1, 0, 1, 2]
    params.matcher.tColorTest = 5;

    // Initialize info
    resetInfo();
}

void ClassifierTerms::resetInfo() {
    info.smallestTemplate = cv::Size(500, 500); // There are no templates larger than 400x400
    info.maxTemplate = cv::Size(0, 0);
    info.minEdgels = INT_MAX;
}

std::shared_ptr<ClassifierTerms> ClassifierTerms::load(cv::FileStorage fsr) {
    std::shared_ptr<ClassifierTerms> terms(new ClassifierTerms());
    cv::FileNode termsNode = fsr["terms"];
    cv::FileNode paramsNode = termsNode["params"], infoNode = termsNode["info"];
    cv::FileNode hasherNode = paramsNode["hasher"], matcherNode = paramsNode["matcher"], objectnessNode = paramsNode["objectness"];

    hasherNode["grid" ] >> terms->params.hasher.grid;
    hasherNode["minVotes"] >> terms->params.hasher.minVotes;
    hasherNode["tablesCount"] >> terms->params.hasher.tablesCount;
    hasherNode["maxDistance"] >> terms->params.hasher.maxDistance;
    hasherNode["binCount"] >> terms->params.hasher.binCount;

    matcherNode["pointsCount"] >> terms-> params.matcher.pointsCount;
    matcherNode["tMatch"] >> terms-> params.matcher.tMatch;
    matcherNode["tOverlap"] >> terms-> params.matcher.tOverlap;
    matcherNode["tColorTest"] >> terms-> params.matcher.tColorTest;
    matcherNode["neighbourhood"] >> terms-> params.matcher.neighbourhood;

    objectnessNode["step"] >> terms-> params.objectness.step;
    objectnessNode["tEdgesMin"] >> terms-> params.objectness.tEdgesMin;
    objectnessNode["tEdgesMax"] >> terms-> params.objectness.tEdgesMax;
    objectnessNode["tMatch"] >> terms-> params.objectness.tMatch;

    infoNode["smallestTemplate"] >> terms->info.smallestTemplate;
    infoNode["maxTemplate"] >> terms->info.maxTemplate;
    infoNode["minEdgels"] >> terms->info.minEdgels;

    return terms;
}

void ClassifierTerms::save(cv::FileStorage &fsw) {
    fsw << "terms" << "{";
        fsw << "params" << "{";
            fsw << "hasher" << "{";
                fsw << "grid" << params.hasher.grid;
                fsw << "minVotes" << params.hasher.minVotes;
                fsw << "tablesCount" << params.hasher.tablesCount;
                fsw << "maxDistance" << params.hasher.maxDistance;
                fsw << "binCount" << params.hasher.binCount;
            fsw << "}";
            fsw << "matcher" << "{";
                fsw << "pointsCount" << params.matcher.pointsCount;
                fsw << "tMatch" << params.matcher.tMatch;
                fsw << "tOverlap" << params.matcher.tOverlap;
                fsw << "tColorTest" << params.matcher.tColorTest;
                fsw << "neighbourhood" << params.matcher.neighbourhood;
            fsw << "}";
            fsw << "objectness" << "{";
                fsw << "step" << params.objectness.step;
                fsw << "tEdgesMin" << params.objectness.tEdgesMin;
                fsw << "tEdgesMax" << params.objectness.tEdgesMax;
                fsw << "tMatch" << params.objectness.tMatch;
            fsw << "}";
        fsw << "}";
        fsw << "info" << "{";
            fsw << "smallestTemplate" << info.smallestTemplate;
            fsw << "maxTemplate" << info.maxTemplate;
            fsw << "minEdgels" << info.minEdgels;
        fsw << "}";
    fsw << "}";
}

std::ostream &operator<<(std::ostream &os, const ClassifierTerms &terms) {
    os << "Params: " << std::endl;
    os << "  hasher: " << std::endl;
    os << "    |_ grid: " << terms.params.hasher.grid << std::endl;
    os << "    |_ minVotes: " << terms.params.hasher.minVotes << std::endl;
    os << "    |_ tablesCount: " << terms.params.hasher.tablesCount << std::endl;
    os << "    |_ maxDistance: " << terms.params.hasher.maxDistance << std::endl;
    os << "    |_ binCount: " << terms.params.hasher.binCount << std::endl;
    os << "  matcher: " << std::endl;
    os << "    |_ pointsCount: " << terms.params.matcher.pointsCount << std::endl;
    os << "    |_ tMatch: " << terms.params.matcher.tMatch << std::endl;
    os << "    |_ tOverlap: " << terms.params.matcher.tOverlap << std::endl;
    os << "    |_ tColorTest: " << terms.params.matcher.tColorTest << std::endl;
    os << "    |_ neighbourhood: (" << terms.params.matcher.neighbourhood.start << ", " << terms.params.matcher.neighbourhood.end << ")" << std::endl;
    os << "  objectness: " << std::endl;
    os << "    |_ step: " << terms.params.objectness.step << std::endl;
    os << "    |_ tEdgesMin: " << terms.params.objectness.tEdgesMin << std::endl;
    os << "    |_ tEdgesMax: " << terms.params.objectness.tEdgesMax << std::endl;
    os << "    |_ tMatch: " << terms.params.objectness.tMatch << std::endl << std::endl;
    os << "Info: " << std::endl;
    os << "  |_ minEdgels: " << terms.info.minEdgels << std::endl;
    os << "  |_ smallestTemplate: " << terms.info.smallestTemplate << std::endl;
    os << "  |_ maxTemplate: " << terms.info.maxTemplate << std::endl;

    return os;
}
