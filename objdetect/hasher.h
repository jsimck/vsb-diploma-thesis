#ifndef VSB_SEMESTRAL_PROJECT_HASHING_H
#define VSB_SEMESTRAL_PROJECT_HASHING_H

#include <opencv2/opencv.hpp>
#include "../core/hash_table.h"
#include "../core/group.h"
#include "../core/window.h"
#include "../core/dataset_info.h"

/**
 * class Hasher
 *
 * Used to train HashTables and quickly verify what templates should be matched
 * per each window passed form objectness detection
 */
class Hasher {
private:
    void generateTriplets(std::vector<HashTable> &tables);
    void initializeBinRanges(std::vector<Template> &templates, std::vector<HashTable> &tables, DataSetInfo &info);
    void initialize(std::vector<Template> &templates, std::vector<HashTable> &tables, DataSetInfo &info);
public:
    // Statics
    static const int IMG_16BIT_MAX;

    // Params
    struct {
        cv::Size grid; // Triplet grid size
        int minVotes; // Minimum number of votes to classify template as window candidate
        uint tablesCount; // Number of tables to generate
        uint maxDistance; // Max distance between each point in triplet
        uint binCount; // Number of bins for depth ranges
    } params;

    // Static methods
    static uchar quantizeDepth(float depth, const std::vector<cv::Range> &ranges, uint binCount);
    static uchar quantizeSurfaceNormal(const cv::Vec3f &normal);
    static cv::Vec3f surfaceNormal(const cv::Mat &src, const cv::Point &c);
    static cv::Vec2i relativeDepths(const cv::Mat &src, const cv::Point &c, const cv::Point &p1, const cv::Point &p2);

    // Constructors
    Hasher();

    // Methods
    void train(std::vector<Template> &templates, std::vector<HashTable> &tables, DataSetInfo &info);
    void verifyCandidates(cv::Mat &sceneDepth, std::vector<HashTable> &tables, std::vector<Window> &windows, DataSetInfo &info);
};

#endif //VSB_SEMESTRAL_PROJECT_HASHING_H
