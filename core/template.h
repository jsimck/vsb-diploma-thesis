#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_H

#include <string>
#include <opencv2/opencv.hpp>
#include <ostream>

/**
 * struct Template
 *
 * Template parse and downloaded from dataset http://cmp.felk.cvut.cz/t-less/
 * used across all matching process at all sorts of places
 */
struct Template {
public:
    int id;
    std::string fileName;
    cv::Mat src;
    cv::Mat srcDepth;

    // Template matching feature points
    std::vector<cv::Point> edgePoints;
    std::vector<cv::Point> stablePoints;

    // Template .yml parameters
    cv::Rect objBB; // Object bounding box
    cv::Mat camK; // Intrinsic camera matrix K
    cv::Mat camRm2c; // Rotation matrix R_m2c
    cv::Vec3f camTm2c; // Translation vector t_m2c
    int elev;
    int mode;

    // Hashing args
    int votes;

    // Constructors
    Template(int id, std::string fileName, cv::Mat src, cv::Mat srcDepth, cv::Rect objBB, cv::Mat camRm2c, cv::Vec3d camTm2c)
            : votes(0), id(id), fileName(fileName), src(src), srcDepth(srcDepth), objBB(objBB), camRm2c(camRm2c), camTm2c(camTm2c) {}

    // Methods
    void voteUp();
    void resetVotes();
    void applyROI();
    void resetROI();

    // Operators
    bool operator==(const Template &rhs) const;
    bool operator!=(const Template &rhs) const;
    friend std::ostream &operator<<(std::ostream &os, const Template &t);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_H
