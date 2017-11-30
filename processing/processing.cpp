#include "processing.h"
#include "../objdetect/hasher.h"
#include "computation.h"
#include <cassert>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv/cv.hpp>

namespace tless {
    /**
     * Applies bilateral filtering around each point A computing optimal gradient in b.
     *
     * Each point A represents a filtered depth to compute normal from, where
     * A[0] is origin. b represents optimal gradient computed across origin b[0]
     * and
     *
     * @param[in]  delta         Current pixel depth value
     * @param[in]  xShift        Patch shift in X direction (+/- patch), if shifted
     * @param[in]  yShift        Patch shift in Y direction (+/- patch), if shifted
     * @param[out] A             3 Points to compute bilateral filter for
     * @param[out] b             2 Values, containing optimal[1] and depth gradient[0]
     * @param[in]  maxDifference Ignore contributions of pixels whose depth difference with central
     *                            pixel is above this threshold
     */
    static void accumulateBilateral(long delta, long xShift, long yShift, long *A, long *b, int maxDifference) {
        long f = std::abs(delta) < maxDifference ? 1 : 0;

        const long fx = f * xShift;
        const long fy = f * yShift;

        A[0] += fx * xShift;
        A[1] += fx * yShift;
        A[2] += fy * yShift;
        b[0] += fx * delta;
        b[1] += fy * delta;
    }

    void quantizedNormals(const cv::Mat &src, cv::Mat &dst, float fx, float fy, int maxDepth, int maxDifference) {
        assert(!src.empty());
        assert(src.type() == CV_16UC1);

        int PS = 5; // patch size
        dst = cv::Mat::zeros(src.size(), CV_8UC1);
        auto offsetX = static_cast<int>(NORMAL_LUT_SIZE * 0.5f);
        auto offsetY = static_cast<int>(NORMAL_LUT_SIZE * 0.5f);

        #pragma omp parallel for default(none) shared(src, dst, NORMAL_LUT) firstprivate(fx, fy, maxDepth, maxDifference, PS, offsetX, offsetY)
        for (int y = PS; y < src.rows - PS; y++) {
            for (int x = PS; x < src.cols - PS; x++) {
                // Get depth value at (x,y)
                long d = src.at<ushort>(y, x);

                if (d < maxDepth) {
                    long A[3], b[2];
                    A[0] = A[1] = A[2] = 0;
                    b[0] = b[1] = 0;

                    // Get 8 points around computing points in defined patch of size PS
                    accumulateBilateral(src.at<ushort>(y - PS, x - PS) - d, -PS, -PS, A, b, maxDifference);
                    accumulateBilateral(src.at<ushort>(y - PS, x) - d, 0, -PS, A, b, maxDifference);
                    accumulateBilateral(src.at<ushort>(y - PS, x + PS) - d, +PS, -PS, A, b, maxDifference);
                    accumulateBilateral(src.at<ushort>(y, x - PS) - d, -PS, 0, A, b, maxDifference);
                    accumulateBilateral(src.at<ushort>(y, x + PS) - d, +PS, 0, A, b, maxDifference);
                    accumulateBilateral(src.at<ushort>(y + PS, x - PS) - d, -PS, +PS, A, b, maxDifference);
                    accumulateBilateral(src.at<ushort>(y + PS, x) - d, 0, +PS, A, b, maxDifference);
                    accumulateBilateral(src.at<ushort>(y + PS, x + PS) - d, +PS, +PS, A, b, maxDifference);

                    // Solve
                    long det = A[0] * A[2] - A[1] * A[1];
                    long Dx = A[2] * b[0] - A[1] * b[1];
                    long Dy = -A[1] * b[0] + A[0] * b[1];

                    // Multiply differences by focal length
                    float Nx = fx * Dx;
                    float Ny = fy * Dy;
                    auto Nz = static_cast<float>(-det * d);

                    // Get normal vector size
                    float norm = std::sqrt(Nx * Nx + Ny * Ny + Nz * Nz);

                    if (norm > 0) {
                        float normInv = 1.0f / (norm);

                        // Normalize normal
                        Nx *= normInv;
                        Ny *= normInv;
                        // Nz *= normInv;

                        // Get values for pre-generated Normal look up table
                        auto vX = static_cast<int>(Nx * offsetX + offsetX);
                        auto vY = static_cast<int>(Ny * offsetY + offsetY);
                        // auto vZ = static_cast<int>(Nz * NORMAL_LUT_SIZE + NORMAL_LUT_SIZE);

                        // Save quantized normals, ignore vZ, we quantize only in top half of sphere (cone)
                        dst.at<uchar>(y, x) = NORMAL_LUT[vY][vX];
                        // dst.at<uchar>(y, x) = static_cast<uchar>(std::fabs(Nz) * 255); // Lambert
                    } else {
                        dst.at<uchar>(y, x) = 0; // Discard shadows & distant objects from depth sensor
                    }
                } else {
                    dst.at<uchar>(y, x) = 0; // Wrong depth
                }
            }
        }

        cv::medianBlur(dst, dst, 5);
    }

    void relativeDepths(const cv::Mat &src, cv::Point c, cv::Point p1, cv::Point p2, int *depths) {
        assert(!src.empty());
        assert(src.type() == CV_16U);

        depths[0] = static_cast<int>(src.at<ushort>(p1) - src.at<ushort>(c));
        depths[1] = static_cast<int>(src.at<ushort>(p2) - src.at<ushort>(c));
    }

    void depthEdgelsIntegral(const cv::Mat &src, cv::Mat &sum, int minDepth, int maxDepth, int minMag) {
        assert(!src.empty());
        assert(src.type() == CV_16U);

        const int filterX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        const int filterY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

        cv::Mat srcBlurred = src.clone();
        cv::Mat edgels = cv::Mat::zeros(src.size(), CV_8UC1);

        #pragma omp parallel for default(none) shared(srcBlurred, edgels, filterX, filterY) firstprivate(minDepth, maxDepth, minMag)
        for (int y = 1; y < srcBlurred.rows - 1; y++) {
            for (int x = 1; x < srcBlurred.cols - 1; x++) {
                int i = 0, sumX = 0, sumY = 0;
                bool skip = false;

                for (int yy = 0; yy < 3 && !skip; yy++) {
                    for (int xx = 0; xx < 3; xx++) {
                        int px = srcBlurred.at<ushort>(yy + y - 1, x + xx - 1);

                        // Skip pixels out range
                        if (px < minDepth || px > maxDepth) {
                            skip = true;
                            break;
                        }

                        sumX += px * filterX[i];
                        sumY += px * filterY[i];
                        i++;
                    }
                }

                if (skip) {
                    continue;
                }

                edgels.at<uchar>(y, x) = static_cast<uchar>((std::sqrt(sqr<float>(sumX) + sqr<float>(sumY)) > minMag) ? 1 : 0);
            }
        }

        cv::integral(edgels, sum, CV_32S);
    }

    float depthNormalizationFactor(float depth, std::vector<cv::Vec2f> errorFunction) {
        float ratio = 0;

        for (size_t j = 0; j < errorFunction.size() - 1; j++) {
            if (depth < errorFunction[j + 1][0]) {
                ratio = (1 - errorFunction[j + 1][1]);
                break;
            }
        }

        return ratio;
    }

    uchar quantizedDepth(int depth, std::vector<cv::Range> &ranges) {
        // Depth should have max value of <-65536, +65536>
        assert(depth >= -Hasher::IMG_16BIT_MAX && depth <= Hasher::IMG_16BIT_MAX);
        assert(!ranges.empty());

        // Loop through histogram ranges and return quantized index
        for (size_t i = 0; i < ranges.size(); i++) {
            if (depth >= ranges[i].start && depth < ranges[i].end) {
                return DEPTH_LUT[i];
            }
        }

        // If value is IMG_16BIT_MAX it belongs to last bin
        return DEPTH_LUT[ranges.size() - 1];
    }

    void filterSobel(const cv::Mat &src, cv::Mat &dst, bool xFilter, bool yFilter) {
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

                dst.at<float>(y, x) = std::sqrt(sqr<float>(sumX) + sqr<float>(sumY));
            }
        }
    }

    // TODO define param for min magnitude
    void quantizedOrientationGradients(const cv::Mat &srcGray, cv::Mat &quantizedOrientations, cv::Mat &magnitude) {
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

    uchar quantizeOrientationGradient(float deg) {
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
}