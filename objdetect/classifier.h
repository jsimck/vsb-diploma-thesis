#ifndef VSB_SEMESTRAL_PROJECT_CLASSIFIER_H
#define VSB_SEMESTRAL_PROJECT_CLASSIFIER_H

#include "../core/group.h"
#include "../core/match.h"
#include "../core/hash_table.h"
#include "../utils/parser.h"
#include "hasher.h"
#include "objectness.h"
#include "../core/window.h"
#include "matcher.h"
#include "../core/dataset_info.h"

/**
 * class Classifier
 *
 * Main class which runs all other classifiers and template parsers in the correct order.
 * In this class it's also possible to fine-tune the resulted parameters of each verification stage
 * which can in the end produce different results. These params can be adapted to processed templates
 * and scenes.
 */
class Classifier {
private:
    std::string scenePath;
    std::string sceneName;

    cv::Mat scene;
    cv::Mat sceneHSV;
    cv::Mat sceneGray;
    cv::Mat sceneDepth;
    cv::Mat sceneDepthNorm;

    DataSetInfo info;
    std::vector<Template> templates;
    std::vector<HashTable> tables;
    std::vector<Window> windows;
    std::vector<Match> matches;

    // Methods
    void loadScene();
    void load(const std::string &trainedTemplatesListPath, const std::string &trainedPath);
    void trainHashTables();
    void detectObjectness();
    void verifyTemplateCandidates();
    void matchTemplates();
public:
    // Classifiers
    Objectness objectness;
    Hasher hasher;
    Matcher matcher;

    // Constructors
    explicit Classifier(std::string scenePath = "data/scene_01/", std::string sceneName = "0000.png");

    // Methods
    void train(std::string templatesListPath, std::string resultPath, std::vector<uint> indices = {});
    void detect(std::string trainedTemplatesListPath, std::string trainedPath);

    // Getters
    const std::string &getScenePath() const;
    const std::string &getSceneName() const;
    const cv::Mat &getScene() const;
    const cv::Mat &getSceneGrayscale() const;
    const cv::Mat &getSceneDepth() const;
    const cv::Mat &getSceneDepthNorm() const;
    const std::vector<Template> &getTemplates() const;
    const std::vector<HashTable> &getHashTables() const;
    const std::vector<Window> &getWindows() const;
    const std::vector<Match> &getMatches() const;

    // Setters
    void setScenePath(const std::string &scenePath);
    void setSceneName(const std::string &sceneName);
};

#endif //VSB_SEMESTRAL_PROJECT_CLASSIFIER_H
