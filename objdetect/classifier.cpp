#include "classifier.h"
#include "../utils/timer.h"
#include "../utils/visualizer.h"

Classifier::Classifier(std::string scenePath, std::string sceneName) {
    // Init properties
    setScenePath(scenePath);
    setSceneName(sceneName);

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
    hasher.setMaxDistance(3);

    // Init template matcher
    matcher.setPointsCount(100);
    matcher.setTMatch(0.6f);
    matcher.setTOverlap(0.1f);
    matcher.setNeighbourhood(cv::Range(-2, 2)); // 5x5 -> [-2, -1, 0, 1, 2]
    matcher.setTColorTest(3);
}

void Classifier::train(std::string templatesPath, std::string resultPath, std::vector<uint> indices) {
    std::ifstream ifs(templatesPath);
    assert(ifs.is_open());

    // Init parser and common
    Parser parser;
    std::ostringstream oss;
    std::vector<Template> tpls;
    std::string line;

    parser.indices.swap(indices);

    Timer t;
    std::cout << "Training... " << std::endl;

    while (ifs >> line) {
        std::cout << "  |_ " << line;

        // Parse each object by one and persist it
        parser.parse(line, tpls, info);

        // Train features for loaded templates
        matcher.train(tpls);

        // Persist trained data
        oss.str("");
        oss << resultPath << "trained_" << std::setw(2) << std::setfill('0') << tpls[0].id / 2000 << ".yml.gz";
        std::string trainedPath = oss.str();
        cv::FileStorage fsw(trainedPath, cv::FileStorage::WRITE);

        fsw << "templates" << "[";
        for (auto &tpl : tpls) {
            tpl.persist(fsw);
        }
        fsw << "]";

        fsw.release();
        tpls.clear();
        std::cout << " -> " << trainedPath << std::endl;
    }

    std::cout << "DONE!, took: " << t.elapsed() << " s" << std::endl << std::endl;
}

void Classifier::loadScene() {
    // Checks
    assert(scenePath.length() > 0);
    assert(scenePath.at(scenePath.length() - 1) == '/');
    assert(sceneName.length() > 0);

    // Load scenes
    std::cout << "Loading scene... ";
    scene = cv::imread(scenePath + "rgb/" + sceneName, CV_LOAD_IMAGE_COLOR);
    sceneDepth = cv::imread(scenePath + "depth/" + sceneName, CV_LOAD_IMAGE_UNCHANGED);

    // Convert and normalize
    cv::cvtColor(scene, sceneHSV, CV_BGR2HSV);
    cv::cvtColor(scene, sceneGray, CV_BGR2GRAY);
    sceneGray.convertTo(sceneGray, CV_32F, 1.0f / 255.0f);
    sceneDepth.convertTo(sceneDepth, CV_32F); // TODO work with 16U (int) rather than floats
    sceneDepth.convertTo(sceneDepthNorm, CV_32F, 1.0f / 65536.0f);

    // Check if conversion went ok
    assert(!sceneHSV.empty());
    assert(!sceneGray.empty());
    assert(!sceneDepthNorm.empty());
    assert(scene.type() == CV_8UC3);
    assert(sceneHSV.type() == CV_8UC3);
    assert(sceneGray.type() == CV_32FC1);
    assert(sceneDepth.type() == CV_32FC1);
    assert(sceneDepthNorm.type() == CV_32FC1);

    std::cout << "DONE!" << std::endl << std::endl;
}

void Classifier::extractMinEdgels() {
    // Checks
    assert(!templates.empty());

    // Extract min edgels
    std::cout << "Extracting min edgels... ";
    Timer t;
    objectness.extractMinEdgels(templates, info);
    std::cout << "DONE! " << std::endl;
    std::cout << "  |_ Minimum edgels found: " << info.minEdgels << ", took: " << t.elapsed() << " s" << std::endl << std::endl;
}

void Classifier::trainHashTables() {
    // Checks
    assert(!templates.empty());

    // Train hash tables
    std::cout << "Training hash tables... " << std::endl;
    Timer t;
    hasher.train(templates, tables, info);
    assert(!tables.empty());
    std::cout << "DONE! took: " << t.elapsed() << "s, " << tables.size() << " hash tables generated" <<std::endl << std::endl;
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
//    Visualizer::visualizeWindows(this->scene, windows, false);
#endif
}

void Classifier::verifyTemplateCandidates() {
    // Checks
    assert(!tables.empty());

    // Verification started
    std::cout << "Verification of template candidates, using trained HashTables started... " << std::endl;
    Timer t;
    hasher.verifyCandidates(sceneDepth, tables, windows, info);
    std::cout << "DONE! took: " << t.elapsed() << "s" << std::endl << std::endl;

#ifndef NDEBUG
//    Visualizer::visualizeHashing(scene, sceneDepth, tables, windows, info, hasher.getGrid(), false);
//    Visualizer::visualizeWindows(this->scene, windows, false);
#endif
}

void Classifier::matchTemplates() {
    // Checks
    assert(!windows.empty());

    // Verification started
    std::cout << "Template matching started... " << std::endl;
    Timer t;
    matcher.match(sceneHSV, sceneGray, sceneDepth, windows, matches);
    std::cout << "DONE! " << matches.size() << " matches found, took: " << t.elapsed() << "s" << std::endl << std::endl;
}

void Classifier::detect(std::string trainedTemplatesPath) {
    /// Hypothesis generation
    // Load scene images
    loadScene();

    // Extract min edgels
    extractMinEdgels();

    // Train hash tables
    trainHashTables();

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
//    Visualizer::visualizeMatches(scene, matches, templates);
}

const std::string &Classifier::getScenePath() const {
    return scenePath;
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

const std::vector<Template> &Classifier::getTemplates() const {
    return templates;
}

const std::vector<Window> &Classifier::getWindows() const {
    return windows;
}

const std::vector<Match> &Classifier::getMatches() const {
    return matches;
}

void Classifier::setScenePath(const std::string &scenePath) {
    assert(scenePath.length() > 0);
    assert(scenePath[scenePath.length() - 1] == '/');
    this->scenePath = scenePath;
}

void Classifier::setSceneName(const std::string &sceneName) {
    assert(sceneName.length() > 0);
    this->sceneName = sceneName;
}