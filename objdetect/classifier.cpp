#include "classifier.h"
#include "../utils/timer.h"

Classifier::Classifier(std::string basePath, std::vector<std::string> templateFolders, std::string scenePath, std::string sceneName) {
    // Init properties
    setBasePath(basePath);
    setTemplateFolders(templateFolders);
    setScenePath(scenePath);
    setSceneName(sceneName);

    // Init parser
    parser.setBasePath(basePath);
    parser.setTemplateFolders(templateFolders);
    parser.setTplCount(1296);

    // Init objectness
    objectness.setStep(5);
    objectness.setMinThreshold(0.01f);
    objectness.setMaxThreshold(0.1f);
    objectness.setSlidingWindowSizeFactor(1.0f);
    objectness.setMatchThresholdFactor(0.3f);

    // Init hasher
    hasher.setReferencePointsGrid(cv::Size(12, 12));
    hasher.setHashTableCount(100);
    hasher.setHistogramBinCount(5);
    hasher.setMinVotesPerTemplate(3);
    hasher.setMaxTripletDistance(5);

    // Init template matcher
    templateMatcher.setFeaturePointsCount(100);
    templateMatcher.setMatchThreshold(0.6f);
    templateMatcher.setMatchNeighbourhood(cv::Range(-2, 2)); // 5x5 -> [-2, -1, 0, 1, 2]
    // Training constants
    templateMatcher.setCannyThreshold1(100);
    templateMatcher.setCannyThreshold2(200);
    templateMatcher.setSobelMaxThreshold(50);
    templateMatcher.setGrayscaleMinThreshold(50);
}

void Classifier::parseTemplates() {
    // Checks
    assert(basePath.length() > 0);
    assert(templateFolders.size() > 0);

    // Parse
    std::cout << "Parsing... " << std::endl;
    parser.parse(templateGroups, info);
    assert(templateGroups.size() > 0);
    std::cout << "  |_ Smallest template found: " << info.smallestTemplateSize << std::endl;
    std::cout << "  |_ Largest template found: " << info.maxTemplateSize << std::endl << std::endl;
    std::cout << "DONE! " << templateGroups.size() << " template groups parsed" << std::endl;
}

void Classifier::extractMinEdgels() {
    // Checks
    assert(templateGroups.size() > 0);

    // Extract min edgels
    std::cout << "Extracting min edgels... ";
    objectness.extractMinEdgels(templateGroups, info);
    std::cout << "DONE! " << std::endl;
    std::cout << "  |_ Minimum edgels found: " << info.minEdgels << std::endl;
}

void Classifier::trainHashTables() {
    // Checks
    assert(templateGroups.size() > 0);

    // Train hash tables
    std::cout << "Training hash tables... " << std::endl;
    Timer t;
    hasher.train(templateGroups, hashTables, info);
    assert(hashTables.size() > 0);
    std::cout << "DONE! took: " << t.elapsed() << "s, " << hashTables.size() << " hash tables generated" <<std::endl << std::endl;
}

void Classifier::trainTemplates() {
    // Checks
    assert(templateGroups.size() > 0);

    // Train hash tables
    std::cout << "Training templates for template matching... " << std::endl;
    Timer t;
    templateMatcher.train(templateGroups);
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
    setScene(cv::imread(basePath + scenePath + "rgb/" + sceneName, CV_LOAD_IMAGE_COLOR));
    setSceneDepth(cv::imread(basePath + scenePath + "depth/" + sceneName, CV_LOAD_IMAGE_UNCHANGED));

    // Convert and normalize
    cv::cvtColor(scene, sceneHSV, CV_BGR2HSV);
    cv::cvtColor(scene, sceneGrayscale, CV_BGR2GRAY);
    sceneGrayscale.convertTo(sceneGrayscale, CV_32F, 1.0f / 255.0f);
    sceneDepth.convertTo(sceneDepth, CV_32F); // TODO work with 16S (int) rather than floats
    sceneDepth.convertTo(sceneDepthNormalized, CV_32F, 1.0f / 65536.0f);

    // Check if conversion went ok
    assert(!sceneHSV.empty());
    assert(!sceneGrayscale.empty());
    assert(!sceneDepthNormalized.empty());
    assert(scene.type() == 16); // CV_8UC3
    assert(sceneHSV.type() == 16); // CV_8UC3
    assert(sceneGrayscale.type() == 5); // CV_32FC1
    assert(sceneDepth.type() == 5); // CV_32FC1
    assert(sceneDepthNormalized.type() == 5); // CV_32FC1

    std::cout << "DONE!" << std::endl << std::endl;
}

void Classifier::detectObjectness() {
    // Checks
    assert(info.smallestTemplateSize.area() > 0);
    assert(info.minEdgels > 0);

    // Objectness detection
    std::cout << "Objectness detection started... " << std::endl;
    Timer t;
    objectness.objectness(sceneGrayscale, scene, sceneDepthNormalized, windows, info);
    std::cout << "  |_ Windows classified as containing object extracted: " << windows.size() << std::endl;
    std::cout << "DONE! took: " << t.elapsed() << "s" << std::endl << std::endl;
}

void Classifier::verifyTemplateCandidates() {
    // Checks
    assert(hashTables.size() > 0);

    // Verification started
    std::cout << "Verification of template candidates, using trained HashTables started... " << std::endl;
    Timer t;
    hasher.verifyTemplateCandidates(sceneDepth, hashTables, windows, info);
    std::cout << "DONE! took: " << t.elapsed() << "s" << std::endl << std::endl;

#ifndef NDEBUG
//    // Show results
//    cv::Mat filteredLocations = scene.clone();
//    for (auto &&window : windows) {
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
    templateMatcher.match(sceneHSV, sceneGrayscale, sceneDepth, windows, matches);
    std::cout << "Template matching took: " << t.elapsed() << "s" << std::endl;


    std::cout << "SIZE: " << matches.size() << std::endl;

    // Template TemplateMatcher
    cv::Mat sceneCopy = scene.clone();
    for (auto &&bB : matches) {
        cv::rectangle(sceneCopy, cv::Point(bB.bb.x, bB.bb.y), cv::Point(bB.bb.x + bB.bb.width, bB.bb.y + bB.bb.height), cv::Scalar(0, 255, 0));
    }

    // Show matched template results
    std::cout << "Classification took: " << t.elapsed() << "s" << std::endl;
    cv::imshow("Match template result", sceneCopy);
    cv::waitKey(0);
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
}

void Classifier::classifyTest(std::unique_ptr<std::vector<int>> &indices) {
    /// Hypothesis generation
    // Load scene images
    loadScene();

    // Parse templates with specific indices
    parser.setIndices(indices);
    parseTemplates();

    // Extract min edgels
    extractMinEdgels();

    // Train hash tables
    trainHashTables();

    // Train templates for template matching
    trainTemplates();

    /// Hypothesis verification
    // Start stopwatch
    Timer t;

    // Objectness detection
    detectObjectness();

    // Verification and filtering of template candidates
    verifyTemplateCandidates();

    // Match templates
    matchTemplates();
}

// Getters and setters
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

const std::vector<Window> &Classifier::getWindows() const {
    return windows;
}

const std::vector<TemplateMatch> &Classifier::getMatches() const {
    return matches;
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

void Classifier::setWindows(const std::vector<Window> &windows) {
    assert(windows.size() > 0);
    this->windows = windows;
}

void Classifier::setMatches(const std::vector<TemplateMatch> &matches) {
    assert(matches.size() > 0);
    this->matches = matches;
}