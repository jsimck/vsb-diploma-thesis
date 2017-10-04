#include "objectness.h"
#include "../processing/processing.h"

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

            Processing::filterSobel(tNorm, tSobel, true, true);
            Processing::thresholdMinMax(tSobel, tSobel, tMin, tMax);

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
    assert(sceneDepthNorm.type() == CV_32FC1);

    // Apply sobel filter and thresholding on normalized Depth scene (<0, 1> px values)
    cv::Mat sSobel;
    Processing::filterSobel(sceneDepthNorm, sSobel, true, true);
    Processing::thresholdMinMax(sSobel, sSobel, tMin, tMax);

    // Calculate image integral
    cv::Mat sIntegral;
    cv::integral(sSobel, sIntegral, CV_32F);

    auto edgels = static_cast<uint>(info.minEdgels * tMatch);
    int sizeX = info.smallestTemplate.width;
    int sizeY = info.smallestTemplate.height;

    // Slide window over scene and calculate edgel count for each overlap
    for (int y = 0; y < sSobel.rows - sizeY; y += step) {
        for (int x = 0; x < sSobel.cols - sizeX; x += step) {

            // Calc edgel value in current sliding window with help of image integral
            uint sceneEdgels = static_cast<uint>(
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
