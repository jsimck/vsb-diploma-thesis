#include "objectness.h"

#include <utility>
#include "../processing/processing.h"
#include "../core/classifier_criteria.h"

namespace tless {
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

            filterSobel(tNorm, tSobel, true, true);

            // Compute integral image for easier computation of edgels
            cv::integral(tNorm, tIntegral, CV_32F);
            edgels = static_cast<int>(tIntegral.at<float>(tIntegral.rows - 1, tIntegral.cols - 1));

            if (edgels < criteria->info.minEdgels) {
                criteria->info.minEdgels = edgels;
            }
        }
    }

    void Objectness::objectness(cv::Mat &src, std::vector<Window> &windows, float scale) {
        assert(criteria->info.smallestTemplate.area() > 0);
        assert(criteria->info.minEdgels > 0);
        assert(criteria->objectnessFactor > 0);
        assert(!src.empty());
        assert(src.type() == CV_16U);

        // TODO FIX DEVIATION FUNCTION
        // Normalize min and max depths
        float minDepth = (criteria->info.minDepth * scale) * 0.8f;
        float maxDepth = (criteria->info.maxDepth * scale) * 1.2f;

        // Generate minEdgels integral image
        cv::Mat integral;
        filterDepthEdgels(src, integral, static_cast<int>(minDepth), static_cast<int>(maxDepth));

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