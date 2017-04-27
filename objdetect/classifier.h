#ifndef VSB_SEMESTRAL_PROJECT_CLASSIFICATOR_H
#define VSB_SEMESTRAL_PROJECT_CLASSIFICATOR_H

#include "../core/group.h"
#include "../core/match.h"
#include "../core/hash_table.h"
#include "../utils/template_parser.h"
#include "hasher.h"
#include "objectness.h"
#include "../core/window.h"
#include "template_matcher.h"
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
    std::string basePath;
    std::string scenePath;
    std::string sceneName;
    std::vector<std::string> templateFolders;

    cv::Mat scene;
    cv::Mat sceneHSV;
    cv::Mat sceneGray;
    cv::Mat sceneDepth;
    cv::Mat sceneDepthNorma;

    DataSetInfo info;
    std::vector<Group> templateGroups;
    std::vector<HashTable> hashTables;
    std::vector<Window> windows;
    std::vector<Match> matches;

    // Methods
    void parseTemplates();
    void loadScene();
    void extractMinEdgels();
    void trainHashTables();
    void trainTemplates();
    void detectObjectness();
    void verifyTemplateCandidates();
    void matchTemplates();
public:
    // Classifiers
    TemplateParser parser;
    Objectness objectness;
    Hasher hasher;
    TemplateMatcher templateMatcher;

    // Constructors
    Classifier(std::string basePath = "data/", std::vector<std::string> templateFolders = {}, std::string scenePath = "scene_01/", std::string sceneName = "0000.png");

    // Methods
    void classify();
    void classifyTest(std::unique_ptr<std::vector<int>> &indices);

    // Getters
    const std::string &getBasePath() const;
    const std::vector<std::string> &getTemplateFolders() const;
    const std::string &getScenePath() const;
    const std::string &getSceneName() const;
    const cv::Mat &getScene() const;
    const cv::Mat &getSceneGrayscale() const;
    const cv::Mat &getSceneDepth() const;
    const cv::Mat &getSceneDepthNormalized() const;
    const std::vector<Group> &getTemplateGroups() const;
    const std::vector<HashTable> &getHashTables() const;
    const std::vector<Window> &getWindows() const;
    const std::vector<Match> &getMatches() const;

    // Setters
    void setBasePath(const std::string &basePath);
    void setTemplateFolders(const std::vector<std::string> &templateFolders);
    void setScenePath(const std::string &scenePath);
    void setSceneName(const std::string &sceneName);
    void setScene(const cv::Mat &scene);
    void setSceneGrayscale(const cv::Mat &sceneGrayscale);
    void setSceneDepth(const cv::Mat &sceneDepth);
    void setSceneDepthNormalized(const cv::Mat &sceneDepthNormalized);
    void setTemplateGroups(const std::vector<Group> &templateGroups);
    void setHashTables(const std::vector<HashTable> &hashTables);
    void setWindows(const std::vector<Window> &windows);
    void setMatches(const std::vector<Match> &matches);
};

#endif //VSB_SEMESTRAL_PROJECT_CLASSIFICATOR_H
