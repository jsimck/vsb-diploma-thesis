#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_H

#include <string>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <utility>
#include "camera.h"
#include "triplet.h"

namespace tless {
    /**
     * @brief Template wrapper, holds all template images and info available.
     */
    struct Template {
    public:
        int id = 0;
        int objId = 0;
        std::string fileName;
        float diameter = 0;
        float resizeRatio = 1.0f;

        // Image sources
        cv::Mat srcRGB;
        cv::Mat srcGray;
        cv::Mat srcHue;
        cv::Mat srcDepth;

        // Extracted template feature points
        cv::Mat srcGradients;
        cv::Mat srcNormals;
        std::vector<cv::Point> edgePoints;
        std::vector<cv::Point> stablePoints;

        // Matching Features
        struct {
            std::vector<uchar> gradients; //!< quantized oriented gradients
            std::vector<uchar> normals; //!< quantized surface normals
            std::vector<ushort> depths; //!< depth value
            std::vector<uchar> hue; //!< hue value from HSV color space
        } features;

        // Other params
        cv::Rect objBB; //!< Object bounding box
        Camera camera; //!< Camera parameters
        float objArea = 0; //!< Area object covers relative to it's window
        int votes = 0;
        ushort minDepth = std::numeric_limits<unsigned short>::max(), maxDepth = 0; //!< Minimum and maximum depth of the object in this template

#ifdef VIZ_HASHING
        std::vector<Triplet> triplets;
#endif

        Template() = default;

        bool operator==(const Template &rhs) const;
        bool operator!=(const Template &rhs) const;
        friend void operator>>(const cv::FileNode &node, Template &t);
        friend cv::FileStorage &operator<<(cv::FileStorage &fs, const Template &t);
        friend std::ostream &operator<<(std::ostream &os, const Template &t);
    };
}

#endif
