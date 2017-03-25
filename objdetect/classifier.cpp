#include "classifier.h"
#include "objectness.h"

Classifier::Classifier() {
    this->setBasePath("data/");
    this->setTemplateFolders({
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
        "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"
    });
    this->setScenePath("scene_01/");
}

Classifier::Classifier(std::string basePath, std::vector<std::string> templateFolders) {
    this->setBasePath(basePath);
    this->setTemplateFolders(templateFolders);
    this->setScenePath("scene_01/");
}

Classifier::Classifier(std::string basePath, std::vector<std::string> templateFolders, std::string scenePath) {
    this->setBasePath(basePath);
    this->setTemplateFolders(templateFolders);
    this->setScenePath(scenePath);
}

Classifier::Classifier(std::string basePath, std::vector<std::string> templateFolders, std::string scenePath, std::string sceneName) {
    this->setBasePath(basePath);
    this->setTemplateFolders(templateFolders);
    this->setScenePath(scenePath);
    this->setSceneName(sceneName);
}

void Classifier::parseTemplates() {
    // Checks
    assert(this->basePath.length() > 0);
    assert(this->templateFolders.size() > 0);

    // Parse
    std::cout << "Parsing... " << std::endl;
    this->parser.setTplCount(1296);
    this->parser.setBasePath(this->basePath);
    this->parser.setTemplateFolders(this->templateFolders);
    this->parser.parse(this->templateGroups);
    assert(this->templateGroups.size() > 0);
    std::cout << "DONE! " << templateGroups.size() << " template groups parsed" << std::endl << std::endl;
}

void Classifier::extractMinEdgels() {
    // Checks
    assert(this->templateGroups.size() > 0);

    // Extract min edgels
    std::cout << "Extracting min edgels... " << std::endl;
    this->setMinEdgels(objectness::extractMinEdgels(this->templateGroups));
    std::cout << "DONE! " << minEdgels << " minimum found" <<std::endl << std::endl;
}

void Classifier::trainHashTables() {
    // Checks
    assert(this->templateGroups.size() > 0);

    // Train hash tables
    std::cout << "Training hash tables... " << std::endl;
    hasher.setFeaturePointsGrid(cv::Size(12, 12)); // Grid of 12x12 feature points
    hasher.train(this->templateGroups, this->hashTables);
    assert(this->hashTables.size() > 0);
    std::cout << "DONE! " << this->hashTables.size() << " hash tables generated" <<std::endl << std::endl;
}

void Classifier::loadScene() {
    // Checks
    assert(this->basePath.length() > 0);
    assert(this->basePath.at(this->basePath.length() - 1) == '/');
    assert(this->scenePath.length() > 0);
    assert(this->scenePath.at(this->scenePath.length() - 1) == '/');
    assert(this->sceneName.length() > 0);

    // Load scenes
    std::cout << "Loading scene... ";
    this->setScene(cv::imread(this->basePath + this->scenePath + "rgb/" + this->sceneName, CV_LOAD_IMAGE_COLOR));
    this->setSceneDepth(cv::imread(this->basePath + this->scenePath + "depth/" + this->sceneName, CV_LOAD_IMAGE_UNCHANGED));

    // Convert and normalize
    cv::cvtColor(this->scene, sceneGrayscale, CV_BGR2GRAY);
    this->sceneGrayscale.convertTo(this->sceneGrayscale, CV_32F, 1.0f / 255.0f);
    this->sceneDepth.convertTo(this->sceneDepth, CV_32F);
    this->sceneDepth.convertTo(this->sceneDepthNormalized, CV_32F, 1.0f / 65536.0f);

    // Check if conversion went ok
    assert(!this->sceneGrayscale.empty());
    assert(!this->sceneDepthNormalized.empty());
    assert(this->scene.type() == 16); // CV_8UC3
    assert(this->sceneGrayscale.type() == 5); // CV_32FC1
    assert(this->sceneDepth.type() == 5); // CV_32FC1
    assert(this->sceneDepthNormalized.type() == 5); // CV_32FC1

    std::cout << "DONE!" << std::endl << std::endl;
}

void Classifier::classify() {
    // Load scene images
    loadScene();

    // Parse templates
    parseTemplates();

    // Extract min edgels
    extractMinEdgels();

    // Train hash tables
    trainHashTables();

    // Print hash tables
    for (auto &table : this->hashTables) {
        std::cout << table << std::endl;
    }
}

void Classifier::classifyTest(std::unique_ptr<std::vector<int>> &indices) {
    // Load scene images
    loadScene();

    // Parse templates with specific indices
    this->parser.setIndices(indices);
    parseTemplates();

    // Extract min edgels
    extractMinEdgels();

    // Train hash tables
    trainHashTables();

    // Print hash tables
    for (auto &table : this->hashTables) {
        std::cout << table << std::endl;
    }

    objectness::objectness(this->sceneGrayscale, this->sceneDepthNormalized, this->scene, this->minEdgels);
}

// Getters and setters
const cv::Vec3f &Classifier::getMinEdgels() const {
    return minEdgels;
}

void Classifier::setMinEdgels(const cv::Vec3f &minEdgels) {
    assert(minEdgels[0] > 0 && minEdgels[1] > 0 && minEdgels[2] > 0);
    this->minEdgels = minEdgels;
}

const std::string &Classifier::getBasePath() const {
    return this->basePath;
}

void Classifier::setBasePath(const std::string &basePath) {
    assert(basePath.length() > 0);
    assert(basePath[basePath.length() - 1] == '/');
    this->basePath = basePath;
}

const std::string &Classifier::getScenePath() const {
    return this->scenePath;
}

void Classifier::setScenePath(const std::string &scenePath) {
    assert(scenePath.length() > 0);
    assert(scenePath[scenePath.length() - 1] == '/');
    this->scenePath = scenePath;
}

const std::vector<std::string> &Classifier::getTemplateFolders() const {
    return this->templateFolders;
}

void Classifier::setTemplateFolders(const std::vector<std::string> &templateFolders) {
    assert(templateFolders.size() > 0);
    this->templateFolders = templateFolders;
}

const cv::Mat &Classifier::getScene() const {
    return this->scene;
}

void Classifier::setScene(const cv::Mat &scene) {
    assert(!scene.empty());
    this->scene = scene;
}

const cv::Mat &Classifier::getSceneDepth() const {
    return this->sceneDepth;
}

void Classifier::setSceneDepth(const cv::Mat &sceneDepth) {
    assert(!sceneDepth.empty());
    this->sceneDepth = sceneDepth;
}

const cv::Mat &Classifier::getSceneDepthNormalized() const {
    return this->sceneDepthNormalized;
}

void Classifier::setSceneDepthNormalized(const cv::Mat &sceneDepthNormalized) {
    assert(!sceneDepthNormalized.empty());
    this->sceneDepthNormalized = sceneDepthNormalized;
}

const std::vector<TemplateGroup> &Classifier::getTemplateGroups() const {
    return this->templateGroups;
}

void Classifier::setTemplateGroups(const std::vector<TemplateGroup> &templateGroups) {
    assert(templateGroups.size() > 0);
    this->templateGroups = templateGroups;
}

const std::vector<HashTable> &Classifier::getHashTables() const {
    return this->hashTables;
}

void Classifier::setHashTables(const std::vector<HashTable> &hashTables) {
    assert(hashTables.size() > 0);
    this->hashTables = hashTables;
}

const std::string &Classifier::getSceneName() const {
    return this->sceneName;
}

void Classifier::setSceneName(const std::string &sceneName) {
    assert(sceneName.length() > 0);
    this->sceneName = sceneName;
}

const cv::Mat &Classifier::getSceneGrayscale() const {
    return this->sceneGrayscale;
}

void Classifier::setSceneGrayscale(const cv::Mat &sceneGrayscale) {
    assert(!sceneGrayscale.empty());
    this->sceneGrayscale = sceneGrayscale;
}
