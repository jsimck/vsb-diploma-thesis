#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_H

#include <string>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <utility>

/**
 * struct Template
 *
 * Holds one template and all it's data that are either extracted at initialization
 * or earned throught the process of template matching.
 */
struct Template {
public:
    int id;
    std::string fileName;
    float diameter;

    // Image sources
    cv::Mat srcGray;
    cv::Mat srcHSV;
    cv::Mat srcDepth;

    // Extracted template feature points
    cv::Mat quantizedGradients;
    cv::Mat quantizedNormals;
    std::vector<cv::Point> edgePoints;
    std::vector<cv::Point> stablePoints;

    // Matching Features
    struct {
        float depthMedian; // median value over all feature points
        std::vector<uchar> gradients; // quantized oriented quantizedGradients
        std::vector<uchar> normals; // quantized surface quantizedNormals
        std::vector<float> depths; // depth value
        std::vector<cv::Vec3b> colors; // HSV color space value
    } features;

    // Template .yml parameters
    cv::Rect objBB; // Object bounding box
    cv::Mat camK; // Intrinsic camera matrix K
    cv::Mat camRm2c; // Rotation matrix R_m2c
    cv::Vec3f camTm2c; // Translation vector t_m2c
    int elev;
    int azimuth;
    int mode;

    // Constructors
    Template() {}
    Template(int id, std::string &fileName, float diameter, cv::Mat src, cv::Mat srcHSV, cv::Mat srcDepth,
                 cv::Mat quantizedGradients, cv::Mat normals, cv::Rect &objBB, cv::Mat camRm2c, const cv::Vec3d &camTm2c)
        : id(id), fileName(fileName), diameter(diameter), srcGray(src), srcHSV(srcHSV), srcDepth(srcDepth), quantizedGradients(quantizedGradients),
          quantizedNormals(normals), objBB(objBB), camRm2c(camRm2c), camTm2c(camTm2c), elev(0), azimuth(0), mode(0) {}

    // Persist and load methods
    static Template load(cv::FileNode node);
    void save(cv::FileStorage &fs);

    // Methods
    void applyROI();
    void resetROI();

    // Dynamic image loading for visualization purposes mostly
    static cv::Mat loadSrc(const std::string &basePath, const Template &tpl, int ddepth);

    // Operators
    bool operator==(const Template &rhs) const;
    bool operator!=(const Template &rhs) const;
    friend std::ostream &operator<<(std::ostream &os, const Template &t);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_H
