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
        uint id = 0;
        std::string fileName;
        float diameter = 0;
        float resizeRatio = 1.0f;

        // Image sources
        cv::Mat srcRGB;
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
            ushort depthMedian = 0; //!< median value over all feature points
            std::vector<uchar> gradients; //!< quantized oriented gradients
            std::vector<uchar> normals; //!< quantized surface normals
            std::vector<ushort> depths; //!< depth value
            std::vector<cv::Vec3b> colors; //!< HSV color space value
        } features;

        cv::Rect objBB; //!< Object bounding box
        int objArea = 0;
        Camera camera; //!< Camera parameters
        int votes = 0;
        std::vector<Triplet> triplets; // TODO - remove/refactor, mostly for debugging, Triplets that were equal in hashing verification

        Template() = default;

        // TODO REMOVE -> moved to visualizer
        /**
         * @brief Dynamic image loading when t.src are not load (mainly for debugging)
         *
         * @param[in]  basePath Full path to template source image
         * @param[in]  tpl      Input template, used to compute path based on it's id and obj type
         * @param[out] dst      Destination of loaded template image
         * @param[in]  iscolor  Color of the loaded source image (CV_LOAD_IMAGE_COLOR, CV_LOAD_IMAGE_UNCHANGED)
         */
        static void loadSrc(const std::string &basePath, const Template &tpl, cv::Mat &dst, int iscolor);

        bool operator==(const Template &rhs) const;
        bool operator!=(const Template &rhs) const;
        friend void operator>>(const cv::FileNode &node, Template &t);
        friend cv::FileStorage &operator<<(cv::FileStorage &fs, const Template &t);
        friend std::ostream &operator<<(std::ostream &os, const Template &t);
    };
}

#endif
