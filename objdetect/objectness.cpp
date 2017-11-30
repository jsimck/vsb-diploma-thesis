#include "objectness.h"

#include "../processing/processing.h"

namespace tless {
    void Objectness::objectness(cv::Mat &src, std::vector<Window> &windows, float scale) {
        assert(criteria->info.smallestTemplate.area() > 0);
        assert(criteria->info.minEdgels > 0);
        assert(criteria->objectnessFactor > 0);
        assert(!src.empty());
        assert(src.type() == CV_16U);

        // Normalize min and max depths to look for objectness in
        float minDepth = criteria->info.minDepth * scale;
        float maxDepth = criteria->info.maxDepth * scale;
        minDepth *= depthNormalizationFactor(minDepth, criteria->depthDeviationFun);
        maxDepth /= depthNormalizationFactor(maxDepth, criteria->depthDeviationFun);

        // Generate integral image of detected edgels
        cv::Mat integral;
        depthEdgelsIntegral(src, integral, static_cast<int>(minDepth), static_cast<int>(maxDepth));

        const auto minEdgels = static_cast<const int>(criteria->info.minEdgels * criteria->objectnessFactor);
        const int sizeX = criteria->info.smallestTemplate.width;
        const int sizeY = criteria->info.smallestTemplate.height;

        // Slide window over scene and calculate edge count for each overlap
        for (int y = 0; y < integral.rows - sizeY; y += criteria->windowStep) {
            for (int x = 0; x < integral.cols - sizeX; x += criteria->windowStep) {

                // Calc edgel count value in current sliding window with help of image integral
                int sceneEdgels = integral.at<int>(y + sizeY, x + sizeX) - integral.at<int>(y, x + sizeX)
                                  - integral.at<int>(y + sizeY, x) + integral.at<int>(y, x);

                if (sceneEdgels >= minEdgels) {
                    windows.emplace_back(x, y, sizeX, sizeY, sceneEdgels);
                }
            }
        }
    }
}