#include "processing.h"
#include "../utils/utils.h"
#include "../objdetect/hasher.h"
#include <cassert>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv/cv.hpp>

void Processing::filterSobel(const cv::Mat &src, cv::Mat &dst, bool xFilter, bool yFilter) {
    assert(!src.empty());
    assert(src.type() == CV_32FC1);

    const int filterX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int filterY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    if (dst.empty()) {
        dst = cv::Mat(src.size(), src.type());
    }

    // Blur image little bit to reduce noise
    cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0);

    #pragma omp parallel for default(none) shared(src, dst, filterX, filterY) firstprivate(xFilter, yFilter)
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
    #pragma omp parallel for default(none) firstprivate(min, max) shared(dst, src)
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (src.at<float>(y, x) >= min && src.at<float>(y, x) <= max) {
                dst.at<float>(y, x) = 1.0f;
            } else {
                dst.at<float>(y, x) = 0.0f;
            }
        }
    }
}

void Processing::quantizedSurfaceNormals(const cv::Mat &srcDepth, cv::Mat &quantizedSurfaceNormals) {
    assert(!srcDepth.empty());
    assert(srcDepth.type() == CV_32FC1);

    // Blur image to reduce noise
    cv::Mat srcBlurred;
    cv::GaussianBlur(srcDepth, srcBlurred, cv::Size(3, 3), 0, 0);
    quantizedSurfaceNormals = cv::Mat::zeros(srcDepth.size(), CV_8UC1);

    #pragma omp parallel for default(none) shared(srcBlurred, quantizedSurfaceNormals)
    for (int y = 1; y < srcBlurred.rows - 1; y++) {
        for (int x = 1; x < srcBlurred.cols - 1; x++) {
            float dzdx = (srcBlurred.at<float>(y, x + 1) - srcBlurred.at<float>(y, x - 1)) / 2.0f;
            float dzdy = (srcBlurred.at<float>(y + 1, x) - srcBlurred.at<float>(y - 1, x)) / 2.0f;
            cv::Vec3f d(-dzdy, -dzdx, 1.0f);

            // Normalize and save normal
            quantizedSurfaceNormals.at<uchar>(y, x) = quantizeSurfaceNormal(cv::normalize(d));
        }
    }
}

void Processing::quantizedOrientationGradients(const cv::Mat &srcGray, cv::Mat &quantizedOrientations, cv::Mat &magnitude) {
    // Checks
    assert(!srcGray.empty());
    assert(srcGray.type() == CV_32FC1);

    // Calc sobel
    cv::Mat sobelX, sobelY, angles;
    cv::Sobel(srcGray, sobelX, CV_32F, 1, 0, 3, 1, 0);
    cv::Sobel(srcGray, sobelY, CV_32F, 0, 1, 3, 1, 0);

    // Calc orientationGradients
    cv::cartToPolar(sobelX, sobelY, magnitude, angles, true);

    // Quantize orientations
    quantizedOrientations = cv::Mat(angles.size(), CV_8UC1);

    #pragma omp parallel for default(none) shared(quantizedOrientations, angles)
    for (int y = 0; y < angles.rows; y++) {
        for (int x = 0; x < angles.cols; x++) {
            quantizedOrientations.at<uchar>(y, x) = quantizeOrientationGradient(angles.at<float>(y, x));
        }
    }
}

uchar Processing::quantizeDepth(float depth, std::vector<cv::Range> &ranges) {
    // Depth should have max value of <-65536, +65536>
    assert(depth >= -Hasher::IMG_16BIT_MAX && depth <= Hasher::IMG_16BIT_MAX);
    assert(!ranges.empty());

    // Loop through histogram ranges and return quantized index
    const size_t iSize = ranges.size();
    for (size_t i = 0; i < iSize; i++) {
        if (ranges[i].start >= depth && depth < ranges[i].end) {
            return static_cast<uchar>(i);
        }
    }

    // If value is IMG_16BIT_MAX it belongs to last bin
    return static_cast<uchar>(iSize - 1);
}

uchar Processing::quantizeSurfaceNormal(const cv::Vec3f &normal) {
    // Normal z coordinate should not be < 0
    assert(normal[2] >= 0);

    // In our case z is always positive, that's why we're using
    // 8 octants in top half of sphere only to quantize into 8 bins
    static cv::Vec3f octantNormals[8] = {
            cv::Vec3f(0.707107f, 0.0f, 0.707107f), // 0. octant
            cv::Vec3f(0.57735f, 0.57735f, 0.57735f), // 1. octant
            cv::Vec3f(0.0f, 0.707107f, 0.707107f), // 2. octant
            cv::Vec3f(-0.57735f, 0.57735f, 0.57735f), // 3. octant
            cv::Vec3f(-0.707107f, 0.0f, 0.707107f), // 4. octant
            cv::Vec3f(-0.57735f, -0.57735f, 0.57735f), // 5. octant
            cv::Vec3f(0.0f, -0.707107f, 0.707107f), // 6. octant
            cv::Vec3f(0.57735f, -0.57735f, 0.57735f), // 7. octant
    };

    uchar minIndex = 9;
    float maxDot = 0, dot = 0;
    for (uchar i = 0; i < 8; i++) {
        // By doing dot product between octant octantNormals and calculated normal
        // we can find maximum -> index of octant where the vector belongs to
        dot = normal.dot(octantNormals[i]);

        if (dot > maxDot) {
            maxDot = dot;
            minIndex = i;
        }
    }

    // Index should in interval <0,7>
    assert(minIndex >= 0 && minIndex < 8);

    return minIndex;
}

uchar Processing::quantizeOrientationGradient(float deg) {
    // Checks
    assert(deg >= 0);
    assert(deg <= 360);

    // We only work in first 2 quadrants (PI)
    int degPI = static_cast<int>(deg) % 180;

    // Quantize
    if (degPI >= 0 && degPI < 36) {
        return 0;
    } else if (degPI >= 36 && degPI < 72) {
        return 1;
    } else if (degPI >= 72 && degPI < 108) {
        return 2;
    } else if (degPI >= 108 && degPI < 144) {
        return 3;
    } else {
        return 4;
    }
}

cv::Vec2i Processing::relativeDepths(const cv::Mat &src, cv::Point &c, cv::Point &p1, cv::Point &p2) {
    assert(!src.empty());
    assert(src.type() == CV_32FC1);

    return cv::Vec2i(
        static_cast<int>(src.at<float>(p1) - src.at<float>(c)),
        static_cast<int>(src.at<float>(p2) - src.at<float>(c))
    );
}
