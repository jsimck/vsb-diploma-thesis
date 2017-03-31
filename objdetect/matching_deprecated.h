#ifndef VSB_SEMESTRAL_PROJECT_MATCHING_H
#define VSB_SEMESTRAL_PROJECT_MATCHING_H

#include <opencv2/opencv.hpp>
#include "../core/template.h"
#include "../core/template_group.h"
#include "../core/hash_table.h"
#include "../core/window.h"

#define MATCH_NORMED_CROSS_CORRELATION
// #define MATCH_NORMED_CORRELATION_COEF

// Sort bounding boxes by their matching score (DESC)
void sortBBByScore(std::vector<cv::Rect> &matchBB, std::vector<float> &scoreBB);
// Suppresses matched bounding boxes which are overlapped by given threshold,
std::vector<cv::Rect> nonMaximaSuppression(std::vector<cv::Rect> &matchBB, std::vector<float> &scoreBB, float overlapThresh = 0.1f);
// Calculates mean of given image ROI
cv::Scalar matRoiMean(cv::Size maskSize, cv::Rect roi);

// Concludes template matching using CROSS CORRELATION function on given template groups and input image
std::vector<cv::Rect> matchTemplate(const cv::Mat &input, std::vector<Window> &windows);

#endif //VSB_SEMESTRAL_PROJECT_MATCHING_H