#include "objectness.h"
#include <cassert>
#include "../utils/utils.h"

void Objectness::filterSobel(cv::Mat &src, cv::Mat &dst) {
    assert(!src.empty());
    assert(src.type() == 5); // CV_32FC1

    int filterX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int filterY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    if (dst.empty()) {
        dst = cv::Mat(src.size(), src.type());
    }

    // Blur image little bit to reduce noise
    cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0);

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

            dst.at<float>(y, x) = sqrt(SQR(sumX) + SQR(sumY));
        }
    }
}

void Objectness::thresholdMinMax(cv::Mat &src, cv::Mat &dst, float min, float max) {
    assert(!src.empty());
    assert(!dst.empty());
    assert(src.type() == 5); // CV_32FC1
    assert(dst.type() == 5); // CV_32FC1
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

void Objectness::extractMinEdgels(std::vector<Group> &groups, DataSetInfo &info) {
    assert(!groups.empty());

    int edgels = 0;
    cv::Mat tSobel, tIntegral, tNorm;

    // Find template which contains least amount of the edgels and get his bounding box
    for (auto &group : groups) {
        for (auto &t : group.templates) {
            // Normalize input image into <0, 1> values and crop it
            t.srcDepth.convertTo(tNorm, CV_32F, 1.0f / 65536.0f);
            tNorm = tNorm(t.objBB);

            filterSobel(tNorm, tSobel);
            thresholdMinMax(tSobel, tSobel, tMin, tMax);

            // Compute integral image for easier computation of edgels
            cv::integral(tNorm, tIntegral, CV_32F);
            edgels = static_cast<int>(tIntegral.at<float>(tIntegral.rows - 1, tIntegral.cols - 1));

            if (edgels < info.minEdgels) {
                info.minEdgels = edgels;
            }
        }
    }
}

void Objectness::objectness(cv::Mat &sceneDepthNorm, std::vector<Window> &windows, DataSetInfo &info) {
    // Check thresholds and min edgels
    assert(info.smallestTemplate.area() > 0);
    assert(info.minEdgels > 0);
    assert(tMatch > 0);

    assert(!sceneDepthNorm.empty());
    assert(sceneDepthNorm.type() == 5); // CV_32FC1

    // Apply sobel filter and thresholding on normalized Depth scene (<0, 1> px values)
    cv::Mat sSobel;
    filterSobel(sceneDepthNorm, sSobel);
    thresholdMinMax(sSobel, sSobel, tMin, tMax);

    // Calculate image integral
    cv::Mat sIntegral;
    cv::integral(sSobel, sIntegral, CV_32F);

    int edgels = static_cast<int>(info.minEdgels * tMatch);
    int sizeX = info.smallestTemplate.width;
    int sizeY = info.smallestTemplate.height;

    // Slide window over scene and calculate edgel count for each overlap
    for (int y = 0; y < sSobel.rows - sizeY; y += step) {
        for (int x = 0; x < sSobel.cols - sizeX; x += step) {

            // Calc edgel value in current sliding window with help of image integral
            unsigned int sceneEdgels = static_cast<unsigned int>(
                sIntegral.at<float>(y + sizeY, x + sizeX)
                - sIntegral.at<float>(y, x + sizeX)
                - sIntegral.at<float>(y + sizeY, x)
                + sIntegral.at<float>(y, x)
            );

            if (sceneEdgels >= edgels) {
                windows.push_back(Window(x, y, sizeX, sizeY, sceneEdgels));
            }
        }
    }
}

float Objectness::getTMin() const {
    return tMin;
}

float Objectness::getTMax() const {
    return tMax;
}

float Objectness::getTMatch() const {
    return tMatch;
}

uint Objectness::getStep() const {
    return step;
}

void Objectness::setTMin(float tMin) {
    assert(tMin >= 0);
    this->tMin = tMin;
}

void Objectness::setTMax(float tMax) {
    assert(tMax >= 0);
    this->tMax = tMax;
}

void Objectness::setTMatch(float tMatch) {
    assert(tMatch > 0);
    this->tMatch = tMatch;
}

void Objectness::setStep(uint step) {
    assert(step > 0);
    this->step = step;
}
