#ifndef VSB_SEMESTRAL_PROJECT_CLASSIFIER_H
#define VSB_SEMESTRAL_PROJECT_CLASSIFIER_H

#include <memory>
#include "../core/match.h"
#include "../core/hash_table.h"
#include "../utils/parser.h"
#include "hasher.h"
#include "objectness.h"
#include "../core/window.h"
#include "matcher.h"
#include "../core/classifier_criteria.h"

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
    cv::Mat scene;
    cv::Mat sceneHSV;
    cv::Mat sceneGray;
    cv::Mat sceneDepth;
    cv::Mat sceneDepthNorm;
    cv::Mat sceneMagnitudes;
    cv::Mat sceneAnglesQuantized;
    cv::Mat sceneSurfaceNormalsQuantized;

    std::vector<Template> templates;
    std::vector<HashTable> tables;
    std::vector<Window> windows;
    std::vector<Match> matches;

    // Methods
    void loadScene(const std::string &scenePath, const std::string &sceneName);
    void load(const std::string &trainedTemplatesListPath, const std::string &trainedPath);
public:
    std::shared_ptr<ClassifierCriteria> criteria;
    Objectness objectness;
    Hasher hasher;
    Matcher matcher;

    // Constructors
    Classifier();

    // Methods
    void train(std::string templatesListPath, std::string resultPath, std::string modelsPath, std::vector<uint> indices = {});
    void detect(std::string trainedTemplatesListPath, std::string trainedPath, std::string scenePath);
};

#endif //VSB_SEMESTRAL_PROJECT_CLASSIFIER_H
