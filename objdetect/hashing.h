#ifndef VSB_SEMESTRAL_PROJECT_HASHING_H
#define VSB_SEMESTRAL_PROJECT_HASHING_H

#include <opencv2/opencv.hpp>
#include "../core/hash_table.h"
#include "../core/template_group.h"

class Hashing {
private:
    std::vector<HashTable> hashTables;

    cv::Vec3d extractSurfaceNormal(cv::Mat &src, cv::Point c);
    int quantizeSurfaceNormals(cv::Vec3f normal);
public:
    void train(std::vector<TemplateGroup> &groups);
};

void trainingSetGeneration(cv::Mat &train, cv::Mat &trainDepth);

#endif //VSB_SEMESTRAL_PROJECT_HASHING_H
