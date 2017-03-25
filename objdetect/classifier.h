#ifndef VSB_SEMESTRAL_PROJECT_CLASSIFICATOR_H
#define VSB_SEMESTRAL_PROJECT_CLASSIFICATOR_H

#include "../core/template_group.h"
#include "../core/hash_table.h"
#include "../utils/template_parser.h"
#include "hasher.h"

class Classifier {
private:
    cv::Vec3f minEdgels;
    std::string basePath;
    std::string scenePath;
    std::string sceneName;
    std::vector<std::string> templateFolders;

    cv::Mat scene;
    cv::Mat sceneGrayscale;
    cv::Mat sceneDepth;
    cv::Mat sceneDepthNormalized;

    std::vector<TemplateGroup> templateGroups;
    std::vector<HashTable> hashTables;

    TemplateParser parser;
    Hasher hasher;

    // Methods
    void parseTemplates();
    void extractMinEdgels();
    void trainHashTables();
    void loadScene();
public:
    // Constructors
    Classifier();
    Classifier(std::string basePath, std::vector<std::string> templateFolders);
    Classifier(std::string basePath, std::vector<std::string> templateFolders, std::string scenePath);
    Classifier(std::string basePath, std::vector<std::string> templateFolders, std::string scenePath, std::string sceneName);

    // Methods
    void classify();
    void classifyTest(std::unique_ptr<std::vector<int>> &indices);

    // Getters
    const cv::Vec3f &getMinEdgels() const;
    const std::string &getBasePath() const;
    const std::vector<std::string> &getTemplateFolders() const;
    const std::string &getScenePath() const;
    const std::string &getSceneName() const;
    const cv::Mat &getScene() const;
    const cv::Mat &getSceneGrayscale() const;
    const cv::Mat &getSceneDepth() const;
    const cv::Mat &getSceneDepthNormalized() const;
    const std::vector<TemplateGroup> &getTemplateGroups() const;
    const std::vector<HashTable> &getHashTables() const;

    // Setters
    void setMinEdgels(const cv::Vec3f &minEdgels);
    void setBasePath(const std::string &basePath);
    void setTemplateFolders(const std::vector<std::string> &templateFolders);
    void setScenePath(const std::string &scenePath);
    void setSceneName(const std::string &sceneName);
    void setScene(const cv::Mat &scene);
    void setSceneGrayscale(const cv::Mat &sceneGrayscale);
    void setSceneDepth(const cv::Mat &sceneDepth);
    void setSceneDepthNormalized(const cv::Mat &sceneDepthNormalized);
    void setTemplateGroups(const std::vector<TemplateGroup> &templateGroups);
    void setHashTables(const std::vector<HashTable> &hashTables);
};

#endif //VSB_SEMESTRAL_PROJECT_CLASSIFICATOR_H
