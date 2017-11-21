#include "objectness.h"

#include <utility>
#include "../processing/processing.h"
#include "../core/classifier_criteria.h"

// TODO move to parser
void Objectness::extractMinEdgels(std::vector<Template> &templates) {
    assert(!templates.empty());

    int edgels = 0;
    cv::Mat tSobel, tIntegral, tNorm;

    // Find template which contains least amount of the edgels and get his bounding box
    for (auto &t : templates) {
        // Normalize input image into <0, 1> values and crop it
        t.srcDepth.convertTo(tNorm, CV_32F, 1.0f / 65536.0f);
        tNorm = tNorm(t.objBB);

        Processing::filterSobel(tNorm, tSobel, true, true);
        Processing::thresholdMinMax(tSobel, tSobel, criteria->train.objectness.tEdgesMin, criteria->train.objectness.tEdgesMax);

        // Compute integral image for easier computation of edgels
        cv::integral(tNorm, tIntegral, CV_32F);
        edgels = static_cast<int>(tIntegral.at<float>(tIntegral.rows - 1, tIntegral.cols - 1));

        if (edgels < criteria->info.minEdgels) {
            criteria->info.minEdgels = edgels;
        }
    }
}

void Objectness::objectness(cv::Mat &sceneDepthNorm, std::vector<Window> &windows) {
    // Check thresholds and min edgels
    assert(criteria->info.smallestTemplate.area() > 0);
    assert(criteria->info.minEdgels > 0);
    assert(criteria->detect.objectness.tMatch > 0);
    assert(!sceneDepthNorm.empty());
    assert(sceneDepthNorm.type() == CV_32FC1);

    // Apply sobel filter and thresholding on normalized Depth scene (<0, 1> px values)
    cv::Mat sSobel;
    Processing::filterSobel(sceneDepthNorm, sSobel, true, true);
    Processing::thresholdMinMax(sSobel, sSobel, criteria->train.objectness.tEdgesMin, criteria->train.objectness.tEdgesMax);

    // Calculate image integral
    cv::Mat sIntegral;
    cv::integral(sSobel, sIntegral, CV_32F);

    auto edgels = static_cast<uint>(criteria->info.minEdgels * criteria->detect.objectness.tMatch);
    int sizeX = criteria->info.smallestTemplate.width;
    int sizeY = criteria->info.smallestTemplate.height;

    // Slide window over scene and calculate edge count for each overlap
    for (int y = 0; y < sSobel.rows - sizeY; y += criteria->detect.objectness.step) {
        for (int x = 0; x < sSobel.cols - sizeX; x += criteria->detect.objectness.step) {

            // Calc edge value in current sliding window with help of image integral
            auto sceneEdgels = static_cast<uint>(
                sIntegral.at<float>(y + sizeY, x + sizeX)
                - sIntegral.at<float>(y, x + sizeX)
                - sIntegral.at<float>(y + sizeY, x)
                + sIntegral.at<float>(y, x)
            );

            if (sceneEdgels >= edgels) {
                windows.emplace_back(x, y, sizeX, sizeY, sceneEdgels);
            }
        }
    }
}

void Objectness::setCriteria(std::shared_ptr<ClassifierCriteria> criteria) {
    this->criteria = criteria;
}