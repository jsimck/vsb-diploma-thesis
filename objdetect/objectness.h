#ifndef VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
#define VSB_SEMESTRAL_PROJECT_OBJECTNESS_H

#include <string>
#include <memory>
#include <utility>
#include "../core/template.h"
#include "../core/window.h"
#include "../core/classifier_criteria.h"

/**
 * class Objetness
 *
 * Simple objectness detection algorithm, based on depth discontinuities of depth images.
 * (depth discontinuities => areas where pixel arise on the edges of objects). Sliding window
 * is used to slide through the scene (using size of a smallest template in dataset) and calculating
 * amount of depth pixels in the scene. Window is classified as containing object if it contains
 * at least 30% of edgels of the template containing least amount of them (info.minEdgels), extracted
 * throught extractMinEdgels method.
 */
class Objectness {
private:
    std::shared_ptr<ClassifierCriteria> criteria;
public:
    // Constructors
    Objectness() = default;

    // Methods
    void extractMinEdgels(std::vector<Template> &templates);
    void objectness(cv::Mat &sceneDepthNorm, std::vector<Window> &windows);

    void setCriteria(std::shared_ptr<ClassifierCriteria> criteria);
};

#endif //VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
