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
    int minVotes;
    cv::Size grid;
    uint tablesCount;
    uint binCount;
    uint maxDistance;

    uchar quantizeDepth(float depth, const std::vector<cv::Range> &ranges);
    void generateTriplets(std::vector<HashTable> &tables);
    void initializeBinRanges(const std::vector<Group> &groups, std::vector<HashTable> &tables, const DataSetInfo &info);
    void initialize(const std::vector<Group> &groups, std::vector<HashTable> &tables, const DataSetInfo &info);
public:
    // Statics
    static const int IMG_16BIT_MAX;

    // Static methods
    static uchar quantizeSurfaceNormal(cv::Vec3f normal);
    static cv::Vec3f surfaceNormal(const cv::Mat &src, const cv::Point c);
    static cv::Vec2i relativeDepths(const cv::Mat &src, const cv::Point c, const cv::Point p1, const cv::Point p2);

    // Constructors
    Hasher(int minVotes = 3, cv::Size grid = cv::Size(12, 12), uint tablesCount = 100, uint binCount = 5, uint maxDistance = 3)
        : minVotes(minVotes), grid(grid), tablesCount(tablesCount), binCount(binCount), maxDistance(maxDistance) {}

    // Methods
    void train(std::vector<Group> &groups, std::vector<HashTable> &tables, DataSetInfo &info);
    void verifyCandidates(cv::Mat &sceneDepth, cv::Mat &scene, std::vector<HashTable> &tables, std::vector<Window> &windows, DataSetInfo &info);

    // Getters
    const cv::Size getGrid();
    uint getTablesCount() const;
    uint getBinCount() const;
    int getMinVotes() const;
    uint getMaxDistance() const;

    // Setters
    void setGrid(cv::Size grid);
    void setTablesCount(uint tablesCount);
    void setBinCount(uint binCount);
    void setMinVotes(int minVotes);
    void setMaxDistance(uint maxDistance);
};

#endif //VSB_SEMESTRAL_PROJECT_HASHING_H
