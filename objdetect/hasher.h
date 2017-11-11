#ifndef VSB_SEMESTRAL_PROJECT_HASHING_H
#define VSB_SEMESTRAL_PROJECT_HASHING_H

#include <opencv2/opencv.hpp>
#include <memory>
#include "../core/hash_table.h"
#include "../core/classifier_criteria.h"
#include "../core/window.h"

/**
 * class Hasher
 *
 * Used to train HashTables and quickly verify what templates should be matched
 * per each window passed form objectness detection
 */
class Hasher {
private:
    std::shared_ptr<ClassifierCriteria> criteria;

    void generateTriplets(std::vector<HashTable> &tables);
    void initializeBinRanges(std::vector<Template> &templates, std::vector<HashTable> &tables);
    void initialize(std::vector<Template> &templates, std::vector<HashTable> &tables);
public:
    // Statics
    static const int IMG_16BIT_MAX;

    // Static methods
    static cv::Vec2i relativeDepths(cv::Mat &src, cv::Point &c, cv::Point &p1, cv::Point &p2);

    // Constructors
    Hasher() = default;

    // Methods
    void train(std::vector<Template> &templates, std::vector<HashTable> &tables);
    void verifyCandidates(cv::Mat &sceneDepth, cv::Mat &sceneSurfaceNormalsQuantized, std::vector<HashTable> &tables, std::vector<Window> &windows);

    void setCriteria(std::shared_ptr<ClassifierCriteria> criteria);
};

#endif //VSB_SEMESTRAL_PROJECT_HASHING_H
