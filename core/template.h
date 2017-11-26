#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_H

#include <string>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <utility>

namespace tless {
    /**
     * @brief Template wrapper, holds all template images and info available.
     */
    struct Template {
    public:
        uint id = 0;
        std::string fileName;
        float diameter = 0;

        // Image sources
        cv::Mat srcGray;
        cv::Mat srcHSV;
        cv::Mat srcDepth;

        // Extracted template feature points
        cv::Mat srcGradients;
        cv::Mat srcNormals;
        std::vector<cv::Point> edgePoints;
        std::vector<cv::Point> stablePoints;

        // Matching Features
        struct {
            float depthMedian = 0; //!< median value over all feature points
            std::vector<uchar> gradients; //!< quantized oriented gradients
            std::vector<uchar> normals; //!< quantized surface normals
            std::vector<float> depths; //!< depth value
            std::vector<cv::Vec3b> colors; //!< HSV color space value
        } features;

        // Template .yml parameters
        cv::Rect objBB; //!< Object bounding box
        cv::Mat camK; //!< Intrinsic camera matrix K
        cv::Mat camRm2c; //!< Rotation matrix R_m2c
        cv::Vec3f camTm2c; //!< Translation vector t_m2c
        int elev = 0;
        int azimuth = 0;
        int mode = 0;

        // Constructors
        Template() {}
        Template(int id, std::string &fileName, float diameter, cv::Mat src, cv::Mat srcHSV, cv::Mat srcDepth,
                 cv::Mat srcGradients, cv::Mat srcNormals, cv::Rect &objBB, cv::Mat camRm2c, const cv::Vec3d &camTm2c)
                : id(id), fileName(fileName), diameter(diameter), srcGray(src), srcHSV(srcHSV), srcDepth(srcDepth),
                  srcGradients(srcGradients),
                  srcNormals(srcNormals), objBB(objBB), camRm2c(camRm2c), camTm2c(camTm2c) {}

        // Dynamic image loading for visualization purposes mostly
        static cv::Mat loadSrc(const std::string &basePath, const Template &tpl, int ddepth);

        // Operators
        bool operator==(const Template &rhs) const;
        bool operator!=(const Template &rhs) const;

        // Friends
        friend void operator>>(const cv::FileNode &node, Template &t);
        friend cv::FileStorage &operator<<(cv::FileStorage &fs, const Template &t);
        friend std::ostream &operator<<(std::ostream &os, const Template &t);
    };
}

#endif
