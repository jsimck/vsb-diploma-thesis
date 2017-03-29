#include "objectness.h"
#include <cassert>
#include "../utils/utils.h"

void Objectness::filterSobel(cv::Mat &src, cv::Mat &dst) {
    // Src should not be empty
    assert(!src.empty());
    assert(src.type() == 5); // CV_32FC1

    // Sobel masks
    int filterX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int filterY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    // Create dst matrix if its empty
    if (dst.empty()) {
        dst = cv::Mat(src.size(), src.type());
    }

    // Blur image little bit to reduce noise
    cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0);

    // Apply sobel filter
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            int i = 0;
            float sumX = 0, sumY = 0;
            for (int yy = 0; yy < 3; yy++) {
                for (int xx = 0; xx < 3; xx++) {
                    float px = src.at<float>(yy + y - 1, x + xx - 1);
                    sumX += px * filterX[i];
                    sumY += px * filterY[i];
                    i++;
                }
            }

            // Add sum of x and y derivatives
            dst.at<float>(y, x) = sqrt(SQR(sumX) + SQR(sumY));
        }
    }
}

void Objectness::thresholdMinMax(cv::Mat &src, cv::Mat &dst, float minThreshold, float maxThreshold) {
    // Check matrices type and if they're not empty
    assert(!src.empty());
    assert(!dst.empty());
    assert(src.type() == 5); // CV_32FC1
    assert(dst.type() == 5); // CV_32FC1

    // Check thresholds
    assert(minThreshold >= 0);
    assert(maxThreshold >= 0 && maxThreshold > minThreshold);

    // Apply very simple min/max thresholding for the source image
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (src.at<float>(y, x) >= minThreshold && src.at<float>(y, x) <= maxThreshold) {
                dst.at<float>(y, x) = 1.0;
            } else {
                dst.at<float>(y, x) = 0.0;
            }
        }
    }
}

cv::Vec3f Objectness::extractMinEdgels(std::vector<TemplateGroup> &templateGroups) {
    // Checks
    assert(!templateGroups.empty());

    // Extract edgels
    int edgels = 0;
    cv::Vec3i output(INT_MAX, 0, 0);
    cv::Mat tplSobel, tplIntegral, tplNormalized;

    // Find template which contains least amount of the edgels and get his bounding box
    for (auto &group : templateGroups) {
        for (auto &t : group.templates) {
            // Normalize input image into <0, 1> values
            t.srcDepth.convertTo(tplNormalized, CV_32F, 1.0f / 65536.0f);

            // Apply sobel filter and thresholding
            filterSobel(tplNormalized, tplSobel);
            thresholdMinMax(tplSobel, tplSobel, this->minThreshold, this->maxThreshold);

            // Compute integral image for easier computation of edgels
            cv::integral(tplNormalized, tplIntegral, CV_32F);
            edgels = static_cast<int>(tplIntegral.at<float>(tplIntegral.rows - 1, tplIntegral.cols - 1));

            if (edgels < output[0]) {
                output[0] = edgels;
                output[1] = t.srcDepth.cols;
                output[2] = t.srcDepth.rows;
            }
        }
    }

    // Save output
    return output;
}

// TODO - we should sent only the specific window locations for further matching
cv::Rect Objectness::objectness(cv::Mat &sceneGrayscale, cv::Mat &sceneColor, cv::Mat &sceneDepthNormalized, std::vector<Window> &windows, cv::Vec3f minEdgels) {
    // Check thresholds and min edgels
    assert(minEdgels[0] > 0);
    assert(minEdgels[1] > 0);
    assert(minEdgels[2] > 0);
    assert(matchThresholdFactor > 0);
    assert(slidingWindowSizeFactor > 0);

    // Matrices should not be empty
    assert(!sceneGrayscale.empty());
    assert(!sceneDepthNormalized.empty());
    assert(!sceneColor.empty());

    // Check channels
    assert(sceneGrayscale.type() == 5); // CV_32FC1
    assert(sceneDepthNormalized.type() == 5); // CV_32FC1
    assert(sceneColor.type() == 16); // CV_8UC3

#ifndef NDEBUG
    cv::Mat resultScene = sceneColor.clone();
#endif

    // Apply sobel filter and thresholding on normalized Depth scene (<0, 1> px values)
    cv::Mat sceneSobel;
    filterSobel(sceneDepthNormalized, sceneSobel);
    thresholdMinMax(sceneSobel, sceneSobel, this->minThreshold, this->maxThreshold);

    // Calculate image integral
    cv::Mat sceneIntegral;
    cv::integral(sceneSobel, sceneIntegral, CV_32F);

    // Init helper variables
    std::vector<cv::Vec4i> windowBBs;
    minEdgels[0] *= matchThresholdFactor;
    int sizeX = static_cast<int>(minEdgels[1] * slidingWindowSizeFactor), sizeY = static_cast<int>(minEdgels[2] * slidingWindowSizeFactor);

    // Slide window over scene and calculate edgel count for each overlap
    for (int y = 0; y < sceneSobel.rows - sizeY; y += step) {
        for (int x = 0; x < sceneSobel.cols - sizeX; x += step) {

            // Calc edgel value in current sliding window with help of image integral
            unsigned int sceneEdgels = static_cast<unsigned int>(
                sceneIntegral.at<float>(y + sizeY, x + sizeX)
                - sceneIntegral.at<float>(y, x + sizeX)
                - sceneIntegral.at<float>(y + sizeY, x)
                + sceneIntegral.at<float>(y, x)
            );

            if (sceneEdgels >= minEdgels[0]) {
                windowBBs.push_back(cv::Vec4i(x, y, x + sizeX, y + sizeY));
                windows.push_back(Window(cv::Point(x, y), cv::Size(sizeX, sizeY), sceneEdgels));
#ifndef NDEBUG
            cv::rectangle(resultScene, cv::Point(x, y), cv::Point(x + sizeX, y + sizeY), cv::Vec3b(190, 190, 190));
#endif
            }
        }
    }

    // Calculate coordinates of outer BB
    int minX = sceneSobel.cols, maxX = 0;
    int minY = sceneSobel.rows, maxY = 0;
    for (int i = 0; i < windowBBs.size(); i++) {
        minX = std::min(minX, windowBBs[i][0]);
        minY = std::min(minY, windowBBs[i][1]);
        maxX = std::max(maxX, windowBBs[i][2]);
        maxY = std::max(maxY, windowBBs[i][3]);
    }

    // Create outer BB
    cv::Rect outerBB(minX, minY, maxX - minX, maxY - minY);
    assert(outerBB.width > 0 && outerBB.height > 0);

#ifndef NDEBUG
    // Draw outer BB based on max/min values of all smaller boxes
    cv::rectangle(resultScene, cv::Point(minX, minY), cv::Point(maxX, maxY), cv::Vec3b(0, 255, 0), 2);

    // Show results
    cv::imshow("Objectness::Result", resultScene);
    cv::imshow("Objectness::Depth Scene", sceneDepthNormalized);
    cv::imshow("Objectness::Sobel Scene", sceneSobel);
    cv::imshow("Objectness::Scene", sceneColor);
    cv::waitKey(0);
#endif

    return outerBB;
}

float Objectness::getMinThreshold() const {
    return minThreshold;
}

float Objectness::getMaxThreshold() const {
    return maxThreshold;
}

float Objectness::getMatchThresholdFactor() const {
    return matchThresholdFactor;
}

float Objectness::getSlidingWindowSizeFactor() const {
    return slidingWindowSizeFactor;
}

unsigned int Objectness::getStep() const {
    return step;
}

void Objectness::setMinThreshold(float minThreshold) {
    assert(minThreshold >= 0);
    this->minThreshold = minThreshold;
}

void Objectness::setMaxThreshold(float maxThreshold) {
    assert(maxThreshold >= 0);
    this->maxThreshold = maxThreshold;
}

void Objectness::setMatchThresholdFactor(float matchThresholdFactor) {
    assert(matchThresholdFactor > 0);
    this->matchThresholdFactor = matchThresholdFactor;
}

void Objectness::setSlidingWindowSizeFactor(float slidingWindowSizeFactor) {
    assert(slidingWindowSizeFactor > 0);
    this->slidingWindowSizeFactor = slidingWindowSizeFactor;
}
void Objectness::setStep(unsigned int step) {
    assert(step > 0);
    this->step = step;
}
