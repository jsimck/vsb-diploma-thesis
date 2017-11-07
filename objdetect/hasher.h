#ifndef VSB_SEMESTRAL_PROJECT_HASHING_H
#define VSB_SEMESTRAL_PROJECT_HASHING_H

#include <opencv2/opencv.hpp>
#include <memory>
#include "../core/hash_table.h"
#include "../core/classifier_terms.h"
#include "../core/window.h"

/**
 * class Hasher
 *
 * Used to train HashTables and quickly verify what templates should be matched
 * per each window passed form objectness detection
 */
class Hasher {
private:
    std::shared_ptr<ClassifierTerms> terms;

    void generateTriplets(std::vector<HashTable> &tables);
    void initializeBinRanges(std::vector<Template> &templates, std::vector<HashTable> &tables);
    void initialize(std::vector<Template> &templates, std::vector<HashTable> &tables);
public:
    // Statics
    static const int IMG_16BIT_MAX;

    // Static methods
    static uchar quantizeDepth(float depth, const std::vector<cv::Range> &ranges);
    static uchar quantizeSurfaceNormal(const cv::Vec3f &normal);
    static cv::Vec3f surfaceNormal(const cv::Mat &src, const cv::Point &c);
    static cv::Vec2i relativeDepths(const cv::Mat &src, const cv::Point &c, const cv::Point &p1, const cv::Point &p2);

    // Constructors
    Hasher() = default;

    // Methods
    void train(std::vector<Template> &templates, std::vector<HashTable> &tables);
    void verifyCandidates(cv::Mat &sceneDepth, std::vector<HashTable> &tables, std::vector<Window> &windows);

    void setTerms(std::shared_ptr<ClassifierTerms> terms);
};

#endif //VSB_SEMESTRAL_PROJECT_HASHING_H
