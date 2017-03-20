#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_H

#include <string>
#include <opencv2/opencv.hpp>
#include <ostream>

struct Template {
public:
    int id;
    std::string fileName;
    cv::Mat src;
    cv::Mat srcDepth;

    // Template .yml parameters
    cv::Rect objBB; // Object bounding box
    cv::Mat camK; // Intrinsic camera matrix K
    cv::Mat camRm2c; // Rotation matrix R_m2c
    cv::Vec3d camTm2c; // Translation vector t_m2c
    int elev;
    int mode;

    Template(int id, std::string fileName, cv::Mat src, cv::Mat srcDepth, cv::Rect objBB, cv::Mat camK, cv::Mat camRm2c, cv::Vec3d camTm2c, int elev, int mode)
            : id(id), fileName(fileName), src(src), srcDepth(srcDepth), objBB(objBB), camK(camK), camRm2c(camRm2c), camTm2c(camTm2c), elev(elev), mode(mode) {}

    friend std::ostream &operator<<(std::ostream &os, const Template &t);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_H
