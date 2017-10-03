#include "objectness.h"
#include "../utils/utils.h"

void Objectness::filterSobel(const cv::Mat &src, cv::Mat &dst) {
    assert(!src.empty());
    assert(src.type() == CV_32FC1);

    if (dst.empty()) {
        dst = cv::Mat(src.size(), src.type());
    }

    assert(dst.type() == CV_32FC1);

    // Blur image little bit to reduce noise
    cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0);

    // Sobel
    cv::Mat gradX, gradY;
    cv::Sobel(src, gradX, CV_32FC1, 1, 0, 3, 2, 0);
    cv::Sobel(src, gradY, CV_32FC1, 0, 1, 3, 2, 0);

    // Convert to abs values and merge XY masks to one image
    cv::absdiff(gradX, cv::Scalar(0.0f), gradX);
    cv::absdiff(gradY, cv::Scalar(0.0f), gradY);
    addWeighted(gradX, 0.5, gradY, 0.5, 0, dst);
}

void Objectness::thresholdMinMax(const cv::Mat &src, cv::Mat &dst, float min, float max) {
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
    assert(sceneDepthNorm.type() == CV_32FC1);

    // Apply sobel filter and thresholding on normalized Depth scene (<0, 1> px values)
    cv::Mat sSobel;
    filterSobel(sceneDepthNorm, sSobel);
    thresholdMinMax(sSobel, sSobel, tMin, tMax);

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
