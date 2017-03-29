#include "classifier.h"
#include "matching.h"
#include "../utils/timer.h"

Classifier::Classifier() {
    setBasePath("data/");
    setTemplateFolders({
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
        "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"
    });
}

Classifier::Classifier(std::string basePath, std::vector<std::string> templateFolders) {
    setBasePath(basePath);
    setTemplateFolders(templateFolders);
}

Classifier::Classifier(std::string basePath, std::vector<std::string> templateFolders, std::string scenePath) {
    setBasePath(basePath);
    setTemplateFolders(templateFolders);
    setScenePath(scenePath);
}

Classifier::Classifier(std::string basePath, std::vector<std::string> templateFolders, std::string scenePath, std::string sceneName) {
    setBasePath(basePath);
    setTemplateFolders(templateFolders);
    setScenePath(scenePath);
    setSceneName(sceneName);
}

void Classifier::parseTemplates() {
    // Checks
    assert(basePath.length() > 0);
    assert(templateFolders.size() > 0);

    // Parse
    std::cout << "Parsing... " << std::endl;
    parser.setTplCount(1296);
    parser.setBasePath(basePath);
    parser.setTemplateFolders(templateFolders);
    parser.parse(templateGroups);
    assert(templateGroups.size() > 0);
    std::cout << "DONE! " << templateGroups.size() << " template groups parsed" << std::endl << std::endl;
}

void Classifier::initObjectness() {
    objectness.setStep(5);
    objectness.setMinThreshold(0.01f);
    objectness.setMaxThreshold(0.1f);
    objectness.setSlidingWindowSizeFactor(1.0f);
    objectness.setMatchThresholdFactor(0.3f);
}

void Classifier::extractMinEdgels() {
    // Checks
    assert(templateGroups.size() > 0);

    // Extract min edgels
    std::cout << "Extracting min edgels... " << std::endl;
    setMinEdgels(objectness.extractMinEdgels(templateGroups));
    std::cout << "DONE! " << minEdgels << " minimum found" <<std::endl << std::endl;
}

void Classifier::trainHashTables() {
    // Checks
    assert(templateGroups.size() > 0);

    // Train hash tables
    std::cout << "Training hash tables... " << std::endl;
    hasher.setFeaturePointsGrid(cv::Size(12, 12)); // Grid of 12x12 feature points
    hasher.setHashTableCount(100);
    hasher.setHistogramBinCount(5);
    hasher.train(templateGroups, hashTables);
    assert(hashTables.size() > 0);
    std::cout << "DONE! " << hashTables.size() << " hash tables generated" <<std::endl << std::endl;
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
    setScene(cv::imread(basePath + scenePath + "rgb/" + sceneName, CV_LOAD_IMAGE_COLOR));
    setSceneDepth(cv::imread(basePath + scenePath + "depth/" + sceneName, CV_LOAD_IMAGE_UNCHANGED));

    // Convert and normalize
    cv::cvtColor(scene, sceneGrayscale, CV_BGR2GRAY);
    sceneGrayscale.convertTo(sceneGrayscale, CV_32F, 1.0f / 255.0f);
    sceneDepth.convertTo(sceneDepth, CV_32F);
    sceneDepth.convertTo(sceneDepthNormalized, CV_32F, 1.0f / 65536.0f);

    // Check if conversion went ok
    assert(!sceneGrayscale.empty());
    assert(!sceneDepthNormalized.empty());
    assert(scene.type() == 16); // CV_8UC3
    assert(sceneGrayscale.type() == 5); // CV_32FC1
    assert(sceneDepth.type() == 5); // CV_32FC1
    assert(sceneDepthNormalized.type() == 5); // CV_32FC1
    std::cout << "DONE!" << std::endl << std::endl;
}

void Classifier::detectObjectness() {
    // Checks
    assert(minEdgels[0] > 0 && minEdgels[1] > 0 && minEdgels[2] > 0);

    // Objectness detection
    std::cout << "Objectness detection started... ";
    setObjectnessROI(objectness.objectness(sceneGrayscale, scene, sceneDepthNormalized, windows, minEdgels));
    std::cout << "DONE! " << std::endl;
    std::cout << "  |_ Bounding box extracted: " << objectnessROI << std::endl;
    std::cout << "  |_ Windows classified as containing object extracted: " << windows.size() << std::endl << std::endl;
}

void Classifier::verifyTemplateCandidates() {
    // Verification started
    std::cout << "Verification of template candidates, using trained HashTables started... " << std::endl;
    hasher.verifyTemplateCandidates(sceneGrayscale, windows, hashTables, templateGroups);
    std::cout << "DONE!" << std::endl << std::endl;
}

void Classifier::classify() {
    // Load scene images
    loadScene();

    // Parse templates
    parseTemplates();

    // Extract min edgels
    initObjectness();
    extractMinEdgels();

    // Train hash tables
    trainHashTables();

    // Start stopwatch
    Timer t;

    // Objectness detection
    detectObjectness();

    // Verification and filtering of template candidates
    verifyTemplateCandidates();

    // Template Matching
    std::vector<cv::Rect> matchBBs = matchTemplate(sceneGrayscale, windows);
    cv::Mat sceneCopy = scene.clone();
    for (int i = 0; i < matchBBs.size(); i++) {
        cv::rectangle(sceneCopy, cv::Point(matchBBs[i].x, matchBBs[i].y), cv::Point(matchBBs[i].x + matchBBs[i].width, matchBBs[i].y + matchBBs[i].height), cv::Scalar(0, 255, 0));
    }

    // Show matched template results
    std::cout << "Classification took: " << t.elapsed() << "s" << std::endl;
    cv::imshow("Match template result", sceneCopy);
    cv::waitKey(0);
}

void Classifier::classifyTest(std::unique_ptr<std::vector<int>> &indices) {
    // Load scene images
    loadScene();

    // Parse templates with specific indices
    parser.setIndices(indices);
    parseTemplates();

    // Extract min edgels
    initObjectness();
    extractMinEdgels();

    // Train hash tables
    trainHashTables();

    // Print hashtables
//    for (const auto &table : hashTables) {
//        std::cout << table << std::endl;
//    }

    // Objectness detection
    detectObjectness();

    // Verification and filtering of template candidates
    verifyTemplateCandidates();
    for (const auto &window : windows) {
        std::cout << window << std::endl;
    }

    // Template Matching
    std::vector<cv::Rect> matchBBs = matchTemplate(sceneGrayscale, windows);
    cv::Mat sceneCopy = scene.clone();
    // for (auto &&bB : matchBBs) {
    //     cv::rectangle(sceneCopy, cv::Point(bB.x, bB.y), cv::Point(bB.x + bB.width, bB.y + bB.height), cv::Scalar(0, 255, 0));
    // }
     cv::rectangle(sceneCopy, cv::Point(matchBBs[0].x, matchBBs[0].y), cv::Point(matchBBs[0].x + matchBBs[0].width, matchBBs[0].y + matchBBs[0].height), cv::Scalar(0, 255, 0));

    // Show matched template results
    cv::imshow("Match template result", sceneCopy);
    cv::waitKey(0);
}

// Getters and setters
const cv::Vec3f &Classifier::getMinEdgels() const {
    return minEdgels;
}

const std::string &Classifier::getBasePath() const {
    return basePath;
}

const std::string &Classifier::getScenePath() const {
    return scenePath;
}

const std::vector<std::string> &Classifier::getTemplateFolders() const {
    return templateFolders;
}

const cv::Mat &Classifier::getScene() const {
    return scene;
}

const cv::Mat &Classifier::getSceneDepth() const {
    return sceneDepth;
}

const std::vector<HashTable> &Classifier::getHashTables() const {
    return hashTables;
}

const std::string &Classifier::getSceneName() const {
    return sceneName;
}

const cv::Mat &Classifier::getSceneDepthNormalized() const {
    return sceneDepthNormalized;
}

const cv::Mat &Classifier::getSceneGrayscale() const {
    return sceneGrayscale;
}

const std::vector<TemplateGroup> &Classifier::getTemplateGroups() const {
    return templateGroups;
}

const cv::Rect &Classifier::getObjectnessROI() const {
    return objectnessROI;
}

const std::vector<Window> &Classifier::getWindows() const {
    return windows;
}

void Classifier::setMinEdgels(const cv::Vec3f &minEdgels) {
    assert(minEdgels[0] > 0 && minEdgels[1] > 0 && minEdgels[2] > 0);
    this->minEdgels = minEdgels;
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

void Classifier::setSceneDepth(const cv::Mat &sceneDepth) {
    assert(!sceneDepth.empty());
    this->sceneDepth = sceneDepth;
}

void Classifier::setSceneDepthNormalized(const cv::Mat &sceneDepthNormalized) {
    assert(!sceneDepthNormalized.empty());
    this->sceneDepthNormalized = sceneDepthNormalized;
}

void Classifier::setTemplateGroups(const std::vector<TemplateGroup> &templateGroups) {
    assert(templateGroups.size() > 0);
    this->templateGroups = templateGroups;
}

void Classifier::setTemplateFolders(const std::vector<std::string> &templateFolders) {
    assert(templateFolders.size() > 0);
    this->templateFolders = templateFolders;
}

void Classifier::setScene(const cv::Mat &scene) {
    assert(!scene.empty());
    this->scene = scene;
}

void Classifier::setHashTables(const std::vector<HashTable> &hashTables) {
    assert(hashTables.size() > 0);
    this->hashTables = hashTables;
}

void Classifier::setSceneName(const std::string &sceneName) {
    assert(sceneName.length() > 0);
    this->sceneName = sceneName;
}

void Classifier::setSceneGrayscale(const cv::Mat &sceneGrayscale) {
    assert(!sceneGrayscale.empty());
    this->sceneGrayscale = sceneGrayscale;
}

void Classifier::setObjectnessROI(const cv::Rect &objectnessROI) {
    assert(objectnessROI.width > 0 && objectnessROI.height > 0);
    assert(objectnessROI.x >= 0 && objectnessROI.x < scene.cols);
    assert(objectnessROI.y >= 0 && objectnessROI.y < scene.rows);
    Classifier::objectnessROI = objectnessROI;
}

void Classifier::setWindows(const std::vector<Window> &windows) {
    assert(windows.size() > 0);
    this->windows = windows;
}
