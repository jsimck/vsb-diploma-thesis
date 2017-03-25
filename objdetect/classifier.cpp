#include "classifier.h"
#include "objectness.h"

Classifier::Classifier() {
    this->basePath = "data/";
    this->templateFolders = {
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
        "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"
    };
    this->scenePath = "scene_01/";
}

Classifier::Classifier(std::string basePath, std::vector<std::string> templateFolders) {
    this->basePath = basePath;
    this->templateFolders = templateFolders;
    this->scenePath = "scene_01/";
}

Classifier::Classifier(std::string basePath, std::vector<std::string> templateFolders, std::string scenePath) {
    this->basePath = basePath;
    this->templateFolders = templateFolders;
    this->scenePath = scenePath;
}

void Classifier::parseTemplates() {
    std::cout << "Parsing... " << std::endl;
    this->parser.setTplCount(1296);
    this->parser.setBasePath(this->basePath);
    this->parser.setTemplateFolders(this->templateFolders);
    this->parser.parse(templateGroups);
    std::cout << "DONE! " << templateGroups.size() << " template groups parsed" << std::endl << std::endl;
}

void Classifier::extractMinEdgels() {
    std::cout << "Extracting min edgels... " << std::endl;
    this->minEdgels = objectness::extractMinEdgels(this->templateGroups);
    std::cout << "DONE! " << minEdgels << " minimum found" <<std::endl << std::endl;
}

void Classifier::trainHashTables() {
    std::cout << "Training hash tables... " << std::endl;
    hasher.train(this->templateGroups, this->hashTables);
    std::cout << "DONE! " << this->hashTables.size() << " hash tables generated" <<std::endl << std::endl;
}

void Classifier::classify() {
    // Parse templates
    parseTemplates();

    // Extract min edgels
    extractMinEdgels();

    // Train hash tables
    trainHashTables();
}

void Classifier::classifyTest(std::unique_ptr<std::vector<int>> &indices) {
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
}

// Getters and setters
const cv::Vec3f &Classifier::getMinEdgels() const {
    return minEdgels;
}

void Classifier::setMinEdgels(const cv::Vec3f &minEdgels) {
    this->minEdgels = minEdgels;
}

const std::string &Classifier::getBasePath() const {
    return this->basePath;
}

void Classifier::setBasePath(const std::string &basePath) {
    this->basePath = basePath;
}

const std::string &Classifier::getScenePath() const {
    return this->scenePath;
}

void Classifier::setScenePath(const std::string &scenePath) {
    this->scenePath = scenePath;
}

const std::vector<std::string> &Classifier::getTemplateFolders() const {
    return this->templateFolders;
}

void Classifier::setTemplateFolders(const std::vector<std::string> &templateFolders) {
    this->templateFolders = templateFolders;
}

const cv::Mat &Classifier::getScene() const {
    return this->scene;
}

void Classifier::setScene(const cv::Mat &scene) {
    this->scene = scene;
}

const cv::Mat &Classifier::getSceneColor() const {
    return this->sceneColor;
}

void Classifier::setSceneColor(const cv::Mat &sceneColor) {
    this->sceneColor = sceneColor;
}

const cv::Mat &Classifier::getSceneDepth() const {
    return this->sceneDepth;
}

void Classifier::setSceneDepth(const cv::Mat &sceneDepth) {
    this->sceneDepth = sceneDepth;
}

const cv::Mat &Classifier::getSceneDepthNormalized() const {
    return this->sceneDepthNormalized;
}

void Classifier::setSceneDepthNormalized(const cv::Mat &sceneDepthNormalized) {
    this->sceneDepthNormalized = sceneDepthNormalized;
}

const std::vector<TemplateGroup> &Classifier::getTemplateGroups() const {
    return this->templateGroups;
}

void Classifier::setTemplateGroups(const std::vector<TemplateGroup> &templateGroups) {
    this->templateGroups = templateGroups;
}

const std::vector<HashTable> &Classifier::getHashTables() const {
    return this->hashTables;
}

void Classifier::setHashTables(const std::vector<HashTable> &hashTables) {
    this->hashTables = hashTables;
}