#ifndef VSB_SEMESTRAL_PROJECT_HASHING_H
#define VSB_SEMESTRAL_PROJECT_HASHING_H

#include <opencv2/opencv.hpp>
#include "../core/hash_table.h"
#include "../core/template_group.h"

class Hashing {
private:
    cv::Size featurePointsGrid;
    std::vector<HashTable> hashTables;

    cv::Vec3d extractSurfaceNormal(cv::Mat &src, cv::Point c);
    int quantizeSurfaceNormals(cv::Vec3f normal);
public:
    Hashing() : featurePointsGrid(cv::Size(12, 12)) {}
    Hashing(cv::Size referencePointsGrid) : featurePointsGrid(referencePointsGrid) {}

    void train(std::vector<TemplateGroup> &groups);

    const cv::Size getFeaturePointsGrid();
    const std::vector<HashTable> &getHashTables();
    void setFeaturePointsGrid(cv::Size featurePointsGrid);
    void setHashTables(const std::vector<HashTable> &hashTables);
};

#endif //VSB_SEMESTRAL_PROJECT_HASHING_H
