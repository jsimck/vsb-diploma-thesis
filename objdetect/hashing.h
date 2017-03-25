#ifndef VSB_SEMESTRAL_PROJECT_HASHING_H
#define VSB_SEMESTRAL_PROJECT_HASHING_H

#include <opencv2/opencv.hpp>
#include "../core/hash_table.h"

class Hashing {
private:
    std::vector<HashTable> hashTables;
public:
    cv::Vec3d extractSurfaceNormal(cv::Mat &src, cv::Point c);
    char quantizeSurfaceNormals(cv::Vec3f normal);
};

void trainingSetGeneration(cv::Mat &train, cv::Mat &trainDepth);

#endif //VSB_SEMESTRAL_PROJECT_HASHING_H
