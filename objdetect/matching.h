#ifndef VSB_SEMESTRAL_PROJECT_MATCHING_H
#define VSB_SEMESTRAL_PROJECT_MATCHING_H

#include <opencv2/opencv.hpp>
#include "../core/template.h"

#define MATCH_NORMED_CROSS_CORRELATION
// #define MATCH_NORMED_CORRELATION_COEF

void sortBBByScore(std::vector<cv::Rect> &matchBB, std::vector<double> &scoreBB);
std::vector<cv::Rect> nonMaximaSuppression(std::vector<cv::Rect> &matchBB, std::vector<double> &scoreBB, double overlapThresh = 0.5);

cv::Scalar matRoiMean(cv::Size maskSize, cv::Rect roi);
std::vector<cv::Rect> matchTemplate(cv::Mat &input, cv::Rect inputRoi, std::vector<Template> &templates);

#endif //VSB_SEMESTRAL_PROJECT_MATCHING_H