#include <opencv2/imgproc.hpp>
#include "utils.h"
#include "visualizer.h"

std::string Utils::matType2Str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

cv::Vec3b Utils::hsv2bgr(cv::Vec3f &hsv) {
    cv::Mat src_hsv(1, 1, CV_32FC3), dst_bgr(1, 1, CV_8UC3);

    // Set color and convert
    src_hsv.at<cv::Vec3f>(0) = hsv;
    cv::cvtColor(src_hsv, dst_bgr, CV_HSV2BGR);

    return dst_bgr.at<cv::Vec3b>(0);
}
