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
    std::vector<std::string> folders;
    std::vector<int> indices;

    cv::Mat scene;
    cv::Mat sceneHSV;
    cv::Mat sceneGray;
    cv::Mat sceneDepth;
    cv::Mat sceneDepthNorm;

    DataSetInfo info;
    std::vector<Group> groups;
    std::vector<HashTable> tables;
    std::vector<Window> windows;
    std::vector<Match> matches;

    // Methods
    inline void parseTemplates();
    inline void loadScene();
    inline void extractMinEdgels();
    inline void trainHashTables();
    inline void trainTemplates();
    inline void detectObjectness();
    inline void verifyTemplateCandidates();
    inline void matchTemplates();
public:
    // Classifiers
    TemplateParser parser;
    Objectness objectness;
    Hasher hasher;
    TemplateMatcher matcher;

    // Constructors
    Classifier(std::string basePath = "data/", std::vector<std::string> folders = {}, std::string scenePath = "scene_01/", std::string sceneName = "0000.png");

    // Methods
    void classify();

    // Getters
    const std::string &getBasePath() const;
    const std::string &getScenePath() const;
    const std::string &getSceneName() const;
    const cv::Mat &getScene() const;
    const cv::Mat &getSceneGrayscale() const;
    const cv::Mat &getSceneDepth() const;
    const cv::Mat &getSceneDepthNorm() const;
    const std::vector<std::string> &getFolders() const;
    const std::vector<Group> &getTemplateGroups() const;
    const std::vector<HashTable> &getHashTables() const;
    const std::vector<Window> &getWindows() const;
    const std::vector<Match> &getMatches() const;
    const std::vector<int> &getIndices() const;

    // Setters
    void setBasePath(const std::string &basePath);
    void setFolders(const std::vector<std::string> &folders);
    void setScenePath(const std::string &scenePath);
    void setSceneName(const std::string &sceneName);
    void setIndices(const std::vector<int> &indices);
};

#endif //VSB_SEMESTRAL_PROJECT_CLASSIFICATOR_H
