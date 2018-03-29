#include "processing.h"
#include "../objdetect/hasher.h"
#include "computation.h"
#include "../utils/timer.h"
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
        dst.create(src.size(), CV_8UC1);
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

    void depthEdgels(const cv::Mat &src, cv::Mat &dst, int minDepth, int maxDepth, int minMag) {
        assert(!src.empty());
        assert(src.type() == CV_16U);

        const int filterX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        const int filterY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        dst = cv::Mat::zeros(src.size(), CV_8UC1);

        #pragma omp parallel for default(none) shared(src, dst, filterX, filterY) firstprivate(minDepth, maxDepth, minMag)
        for (int y = 1; y < src.rows - 1; y++) {
            for (int x = 1; x < src.cols - 1; x++) {
                int i = 0, sumX = 0, sumY = 0;
                bool skip = false;

                for (int yy = 0; yy < 3 && !skip; yy++) {
                    for (int xx = 0; xx < 3; xx++) {
                        int px = src.at<ushort>(yy + y - 1, x + xx - 1);

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

                dst.at<uchar>(y, x) = static_cast<uchar>((std::sqrt(sqr<float>(sumX) + sqr<float>(sumY)) > minMag) ? 1 : 0);
            }
        }
    }

    uchar quantizeDepth(int depth, const std::vector<cv::Range> &ranges) {
        // Loop through histogram ranges and return quantized index
        for (size_t i = 0; i < ranges.size(); i++) {
            if (depth >= ranges[i].start && depth < ranges[i].end) {
                return DEPTH_LUT[i];
            }
        }

        // If value doesn't belong to any of those bins, it's invalid
        return 0;
    }

    void normalizeHSV(const cv::Mat &src, cv::Mat &dst, uchar value, uchar saturation) {
        dst.create(src.size(), CV_8UC1);

        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                cv::Vec3b hsv = src.at<cv::Vec3b>(y, x);

                // Normalize hue value
                if (hsv[2] < value) {
                    hsv[0] = 120; // Set from black to blue
                } else if (hsv[1] < saturation) {
                    hsv[0] = 30; // Set from white to yellow
                }

                // Save only hue to dst
                dst.at<uchar>(y, x) = hsv[0];
            }
        }
    }

    void nms(std::vector<Match> &matches, float maxOverlap) {
        if (matches.empty()) return;

        // Sort all matches by their highest score
        std::sort(matches.rbegin(), matches.rend());

        std::vector<Match> pick;
        std::vector<int> suppress(matches.size()); // Indexes of matches to remove
        std::vector<int> idx(matches.size()); // Indexes of bounding boxes to check
        std::iota(idx.begin(), idx.end(), 0); // Fill idx array with range 0..idx.size()

        while (!idx.empty()) {
            // Pick first element with highest score
            Match &firstMatch = matches[idx[0]];

            // Store this index into suppress array and push to final matches, we won't check against this match again
            suppress.push_back(idx[0]);
            pick.push_back(firstMatch);

            // Check overlaps with all other bounding boxes, skipping first one (since it is the one we're checking with)
            for (size_t i = 1; i < idx.size(); i++) {
                // If overlap is bigger than min threshold or smaller windows are in bigger ones, retain the one with larger score
                if (matches[idx[i]].overlap(firstMatch) > maxOverlap || matches[idx[i]].overlap(firstMatch) >= 1.0f) {
                    suppress.push_back(idx[i]);
                }
            }

            // Remove all suppress indexes from idx array
            idx.erase(std::remove_if(idx.begin(), idx.end(), [&suppress](int v) -> bool {
                return std::find(suppress.begin(), suppress.end(), v) != suppress.end();
            }), idx.end());
            suppress.clear();
        }

        matches.swap(pick);
    }

    void filterEdges(const cv::Mat &src, cv::Mat &dst, int kSize) {
        // Checks
        assert(!src.empty());
        assert(src.type() == CV_8UC1);

        // Blur image to minimize noise
        cv::Mat blurred, gradX, gradY;
        cv::GaussianBlur(src, blurred, cv::Size(kSize, kSize), 0, 0, cv::BORDER_REPLICATE);

        // Compute sobel
        cv::Sobel(blurred, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
        cv::convertScaleAbs(gradX, gradX);
        cv::Sobel(blurred, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
        cv::convertScaleAbs(gradY, gradY);

        // Total Gradient (approximate)
        cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, dst);
    }

    void quantizedGradients(const cv::Mat &src, cv::Mat &dst, float minMag) {
        assert(src.type() == CV_8UC1);

        // Compute sobel
        cv::Mat gradX, gradY, mags, angles;
        cv::Sobel(src, gradX, CV_32F, 1, 0, 3, 1, 0);
        cv::Sobel(src, gradY, CV_32F, 0, 1, 3, 1, 0);

        // Compute angles and magnitudes
        cv::cartToPolar(gradX, gradY, mags, angles, true);

        // Quantize orientations
        dst = cv::Mat::zeros(src.size(), CV_8UC1);
        cv::Mat grad = cv::Mat(src.size(), CV_32FC1);

        #pragma omp parallel for default(none) shared(dst, angles, mags) firstprivate(minMag)
        for (int y = 0; y < dst.rows; y++) {
            for (int x = 0; x < dst.cols; x++) {
                // Check for min
                if (mags.at<float>(y, x) < minMag) { continue; }

                // Quantize orientations
                dst.at<uchar>(y, x) = quantizeGradientOrientation(angles.at<float>(y, x));
            }
        }
    }

    uchar quantizeGradientOrientation(float deg) {
        assert(deg >= 0 && deg <= 360);

        // We only work in first 2 quadrants (PI)
        int degPI = static_cast<int>(deg) % 180;

        // Quantize orientations
        if (degPI >= 0 && degPI < 36) {
            return 1;
        } else if (degPI >= 36 && degPI < 72) {
            return 2;
        } else if (degPI >= 72 && degPI < 108) {
            return 4;
        } else if (degPI >= 108 && degPI < 144) {
            return 8;
        } else {
            return 16;
        }
    }
}