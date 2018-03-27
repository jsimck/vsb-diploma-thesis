#include "objectness.h"

#include "../processing/processing.h"

namespace tless {
    void Objectness::objectness(cv::Mat &src, std::vector<Window> &windows) {
        assert(criteria->info.smallestTemplate.area() > 0);
        assert(criteria->info.minEdgels > 0);
        assert(criteria->objectnessFactor > 0);
        assert(!src.empty());
        assert(src.type() == CV_16U);

        // Generate integral image of detected edgels
        cv::Mat edgels, integral;
        auto minMag = static_cast<int>(criteria->objectnessDiameterThreshold * criteria->info.smallestDiameter * criteria->info.depthScaleFactor);
        depthEdgels(src, edgels, criteria->info.minDepth, criteria->info.maxDepth, minMag);
        cv::integral(edgels, integral, CV_32S);

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