#include "processing.h"
#include "../utils/utils.h"
#include <cassert>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv/cv.hpp>

void Processing::filterSobel(const cv::Mat &src, cv::Mat &dst, bool xFilter, bool yFilter) {
    assert(!src.empty());
    assert(src.type() == CV_32FC1);

    int filterX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int filterY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    if (dst.empty()) {
        dst = cv::Mat(src.size(), src.type());
    }

    // Blur image little bit to reduce noise
    cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0);

    #pragma omp parallel for
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            int i = 0;
            float sumX = 0, sumY = 0;

            for (int yy = 0; yy < 3; yy++) {
                for (int xx = 0; xx < 3; xx++) {
                    float px = src.at<float>(yy + y - 1, x + xx - 1);

                    if (xFilter) { sumX += px * filterX[i]; }
                    if (yFilter) { sumY += px * filterY[i]; }

                    i++;
                }
            }

            dst.at<float>(y, x) = std::sqrt(SQR(sumX) + SQR(sumY));
        }
    }
}

void Processing::thresholdMinMax(const cv::Mat &src, cv::Mat &dst, float min, float max) {
    assert(!src.empty());
    assert(!dst.empty());
    assert(src.type() == CV_32FC1);
    assert(dst.type() == CV_32FC1);
    assert(min >= 0);
    assert(max >= 0 && max > min);

    // Apply very simple min/max thresholding for the source image
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (src.at<float>(y, x) >= min && src.at<float>(y, x) <= max) {
                dst.at<float>(y, x) = 1.0;
            } else {
                dst.at<float>(y, x) = 0.0;
            }
        }
    }
}

void Processing::orientationGradients(const cv::Mat &src, cv::Mat &angle, cv::Mat &magnitude, bool angleInDegrees) {
    // Checks
    assert(!src.empty());
    assert(src.type() == CV_32FC1);

    // Calc sobel
    cv::Mat sobelX, sobelY;
    filterSobel(src, sobelX, true, false);
    filterSobel(src, sobelY, false, true);

    // Calc orientationGradients
    cv::cartToPolar(sobelX, sobelY, magnitude, angle, angleInDegrees);
}
