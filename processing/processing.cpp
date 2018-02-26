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

    // TODO - consider refactoring and sending scale along with other params, max depth and difference can be than modified inside this function ranther than outside
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

    uchar quantizeDepth(int depth, std::vector<cv::Range> &ranges) {
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

    void normalizeHSV(const cv::Mat &src, cv::Mat &dst, uchar value, uchar saturation) {
        dst = cv::Mat(src.size(), CV_8UC1);

        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                cv::Vec3b hsv = src.at<cv::Vec3b>(y, x);

                // Normalize hue value
                if (hsv[2] <= value) {
                    hsv[0] = 120; // Set from black to blue
                } else if (hsv[1] < saturation) {
                    hsv[0] = 30; // Set from white to yellow
                }

                // Save only hue to dst
                dst.at<uchar>(y, x) = hsv[0];
            }
        }
    }

    void nonMaximaSuppression(std::vector<Match> &matches, float maxOverlap) {
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
            cv::Rect firstMatchBB = firstMatch.scaledBB(1.0f);

            // Store this index into suppress array and push to final matches, we won't check against this match again
            suppress.push_back(idx[0]);
            pick.push_back(firstMatch);

            // Check overlaps with all other bounding boxes, skipping first one (since it is the one we're checking with)
            #pragma omp parallel for default(none) shared(firstMatch, matches, idx, suppress) firstprivate(maxOverlap, firstMatchBB)
            for (size_t i = 1; i < idx.size(); i++) {
                // Get overlap BB coordinates of each other bounding box and compare with the first one
                cv::Rect bb = matches[idx[i]].scaledBB(1.0f);
                int x1 = std::min<int>(bb.br().x, firstMatchBB.br().x);
                int x2 = std::max<int>(bb.tl().x, firstMatchBB.tl().x);
                int y1 = std::min<int>(bb.br().y, firstMatchBB.br().y);
                int y2 = std::max<int>(bb.tl().y, firstMatchBB.tl().y);

                // Calculate overlap area
                int h = std::max<int>(0, y1 - y2);
                int w = std::max<int>(0, x1 - x2);
                float overlap = static_cast<float>(h * w) / static_cast<float>(firstMatchBB.area());

                // If overlap is bigger than min threshold, remove the match
                if (overlap > maxOverlap) {
                    #pragma omp critical
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

        // First erode image and blur to minimize noise
        cv::Mat eroded, gradX, gradY;
        cv::erode(src, eroded, cv::Mat(), cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
        cv::GaussianBlur(eroded, eroded, cv::Size(kSize, kSize), 0, 0, cv::BORDER_REPLICATE);

        // Compute sobel
        cv::Sobel(eroded, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
        cv::convertScaleAbs(gradX, gradX);
        cv::Sobel(eroded, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
        cv::convertScaleAbs(gradY, gradY);

        // Total Gradient (approximate)
        cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, dst);
    }

    // TODO define param for min magnitude, try compute edgels in 8-bit to avoid convesion to 32f image
    void quantizedOrientationGradients(const cv::Mat &srcGray, cv::Mat &quantizedOrientations, cv::Mat &magnitude) {
        // Checks
        assert(!srcGray.empty());
        assert(srcGray.type() == CV_8UC1);

        // Convert to 32FC1
        cv::Mat srcNorm;
        srcGray.convertTo(srcNorm, CV_32FC1, 1.0f / 255.0f);

        // Calc sobel
        cv::Mat sobelX, sobelY, angles;
        cv::Sobel(srcNorm, sobelX, CV_32F, 1, 0, 3, 1, 0);
        cv::Sobel(srcNorm, sobelY, CV_32F, 0, 1, 3, 1, 0);

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