#include "classifier.h"
#include "../utils/timer.h"

Classifier::Classifier(std::string basePath, std::vector<std::string> folders, std::string scenePath, std::string sceneName) {
    // Init properties
    setBasePath(basePath);
    setFolders(folders);
    setScenePath(scenePath);
    setSceneName(sceneName);

    // Init parser
    parser.setBasePath(basePath);
    parser.setFolders(folders);
    parser.setTplCount(1296);

    // Init objectness
    objectness.setStep(5);
    objectness.setTMin(0.01f);
    objectness.setTMax(0.1f);
    objectness.setTMatch(0.3f);

    // Init hasher
    hasher.setGrid(cv::Size(12, 12));
    hasher.setTablesCount(100);
    hasher.setBinCount(5);
    hasher.setMinVotes(3);
    hasher.setMaxDistance(5);

    // Init template matcher
    matcher.setFeaturePointsCount(100);
    matcher.setMatchThreshold(0.6f);
    matcher.setMatchNeighbourhood(cv::Range(-2, 2)); // 5x5 -> [-2, -1, 0, 1, 2]
    // Training constants
    matcher.setCannyThreshold1(100);
    matcher.setCannyThreshold2(200);
    matcher.setSobelMaxThreshold(50);
    matcher.setGrayscaleMinThreshold(50);
}

void Classifier::parseTemplates() {
    // Checks
    assert(basePath.length() > 0);
    assert(folders.size() > 0);

    // Parse
    std::cout << "Parsing... " << std::endl;
    parser.parse(groups, info);
    assert(groups.size() > 0);
    std::cout << "  |_ Smallest template found: " << info.smallestTemplate << std::endl;
    std::cout << "  |_ Largest template found: " << info.maxTemplate << std::endl << std::endl;
    std::cout << "DONE! " << groups.size() << " template groups parsed" << std::endl;
}

void Classifier::extractMinEdgels() {
    // Checks
    assert(groups.size() > 0);

    // Extract min edgels
    std::cout << "Extracting min edgels... ";
    objectness.extractMinEdgels(groups, info);
    std::cout << "DONE! " << std::endl;
    std::cout << "  |_ Minimum edgels found: " << info.minEdgels << std::endl << std::endl;
}

void Classifier::trainHashTables() {
    // Checks
    assert(groups.size() > 0);

    // Train hash tables
    std::cout << "Training hash tables... " << std::endl;
    Timer t;
    hasher.train(groups, tables, info);
    assert(tables.size() > 0);
    std::cout << "DONE! took: " << t.elapsed() << "s, " << tables.size() << " hash tables generated" <<std::endl << std::endl;
}

void Classifier::trainTemplates() {
    // Checks
    assert(groups.size() > 0);

    // Train hash tables
    std::cout << "Training templates for template matching... " << std::endl;
    Timer t;
    matcher.train(groups);
    std::cout << "DONE! took: " << t.elapsed() << "s" << std::endl << std::endl;
}

void Classifier::loadScene() {
    // Checks
    assert(basePath.length() > 0);
    assert(basePath.at(basePath.length() - 1) == '/');
    assert(scenePath.length() > 0);
    assert(scenePath.at(scenePath.length() - 1) == '/');
    assert(sceneName.length() > 0);

    // Load scenes
    std::cout << "Loading scene... ";
    scene = cv::imread(basePath + scenePath + "rgb/" + sceneName, CV_LOAD_IMAGE_COLOR);
    sceneDepth = cv::imread(basePath + scenePath + "depth/" + sceneName, CV_LOAD_IMAGE_UNCHANGED);

    // Convert and normalize
    cv::cvtColor(scene, sceneHSV, CV_BGR2HSV);
    cv::cvtColor(scene, sceneGray, CV_BGR2GRAY);
    sceneGray.convertTo(sceneGray, CV_32F, 1.0f / 255.0f);
    sceneDepth.convertTo(sceneDepth, CV_32F); // TODO work with 16S (int) rather than floats
    sceneDepth.convertTo(sceneDepthNorm, CV_32F, 1.0f / 65536.0f);

    // Check if conversion went ok
    assert(!sceneHSV.empty());
    assert(!sceneGray.empty());
    assert(!sceneDepthNorm.empty());
    assert(scene.type() == 16); // CV_8UC3
    assert(sceneHSV.type() == 16); // CV_8UC3
    assert(sceneGray.type() == 5); // CV_32FC1
    assert(sceneDepth.type() == 5); // CV_32FC1
    assert(sceneDepthNorm.type() == 5); // CV_32FC1

    std::cout << "DONE!" << std::endl << std::endl;
}

void Classifier::detectObjectness() {
    // Checks
    assert(info.smallestTemplate.area() > 0);
    assert(info.minEdgels > 0);

    // Objectness detection
    std::cout << "Objectness detection started... " << std::endl;
    Timer t;
    objectness.objectness(sceneDepthNorm, windows, info);
    std::cout << "  |_ Windows classified as containing object extracted: " << windows.size() << std::endl;
    std::cout << "DONE! took: " << t.elapsed() << "s" << std::endl << std::endl;

#ifndef NDEBUG
//    // Show results
//    cv::Mat objectnessLocations = scene.clone();
//    for (auto &window : windows) {
//        cv::rectangle(objectnessLocations, window.tl(), window.br(), cv::Scalar(190, 190, 190));
//    }
//    cv::imshow("Objectness locations detected:", objectnessLocations);
//    cv::waitKey(0);
#endif
}

void Classifier::verifyTemplateCandidates() {
    // Checks
    assert(tables.size() > 0);

    // Verification started
    std::cout << "Verification of template candidates, using trained HashTables started... " << std::endl;
    Timer t;
    hasher.verifyCandidates(sceneDepth, tables, windows, info);
    std::cout << "DONE! took: " << t.elapsed() << "s" << std::endl << std::endl;

#ifndef NDEBUG
//    // Show results
//    cv::Mat filteredLocations = scene.clone();
//    for (auto &window : windows) {
//        if (window.hasCandidates()) {
//            cv::rectangle(filteredLocations, window.tl(), window.br(), cv::Scalar(190, 190, 190));
//        }
//    }
//    cv::imshow("Filtered locations:", filteredLocations);
//    cv::waitKey(0);
#endif
}

void Classifier::matchTemplates() {
    // Checks
    assert(windows.size() > 0);

    // Verification started
    std::cout << "Template matching started... " << std::endl;
    Timer t;
    matcher.match(sceneHSV, sceneGray, sceneDepth, windows, matches);
    std::cout << "Template matching took: " << t.elapsed() << "s" << std::endl;
}

void Classifier::classify() {
    /// Hypothesis generation
    // Load scene images
    loadScene();

    // Parse templates
    parseTemplates();

    // Extract min edgels
    extractMinEdgels();

    // Train hash tables
    trainHashTables();

    // Train templates for template matching
    trainTemplates();

    /// Hypothesis verification
    // Start stopwatch
    Timer tTotal;

    // Objectness detection
    detectObjectness();

    // Verification and filtering of template candidates
    verifyTemplateCandidates();

    // Match templates
    matchTemplates();
    std::cout << "Classification took: " << tTotal.elapsed() << "s" << std::endl;

    /// Show matched template results
    cv::Mat sceneCopy = scene.clone();
    for (auto &match : matches) {
        cv::rectangle(sceneCopy, cv::Point(match.objBB.x, match.objBB.y), cv::Point(match.objBB.x + match.objBB.width, match.objBB.y + match.objBB.height), cv::Scalar(0, 255, 0));
    }

    cv::imshow("Match template result", sceneCopy);
    cv::waitKey(0);
}

const std::string &Classifier::getBasePath() const {
    return basePath;
}

const std::string &Classifier::getScenePath() const {
    return scenePath;
}

const std::vector<std::string> &Classifier::getFolders() const {
    return folders;
}

const cv::Mat &Classifier::getScene() const {
    return scene;
}

const cv::Mat &Classifier::getSceneDepth() const {
    return sceneDepth;
}

const std::vector<HashTable> &Classifier::getHashTables() const {
    return tables;
}

const std::string &Classifier::getSceneName() const {
    return sceneName;
}

const cv::Mat &Classifier::getSceneDepthNorm() const {
    return sceneDepthNorm;
}

const cv::Mat &Classifier::getSceneGrayscale() const {
    return sceneGray;
}

const std::vector<Group> &Classifier::getTemplateGroups() const {
    return groups;
}

const std::vector<Window> &Classifier::getWindows() const {
    return windows;
}

const std::vector<Match> &Classifier::getMatches() const {
    return matches;
}

const std::vector<int> &Classifier::getIndices() const {
    return indices;
}

void Classifier::setBasePath(const std::string &basePath) {
    assert(basePath.length() > 0);
    assert(basePath[basePath.length() - 1] == '/');
    this->basePath = basePath;
}

void Classifier::setScenePath(const std::string &scenePath) {
    assert(scenePath.length() > 0);
    assert(scenePath[scenePath.length() - 1] == '/');
    this->scenePath = scenePath;
}

void Classifier::setFolders(const std::vector<std::string> &folders) {
    assert(folders.size() > 0);
    this->folders = folders;
}

void Classifier::setSceneName(const std::string &sceneName) {
    assert(sceneName.length() > 0);
    this->sceneName = sceneName;
}

void Classifier::setIndices(const std::vector<int> &indices)  {
    assert(indices.size() > 0);
    this->indices = indices;
    parser.setIndices(indices);
}