#include "objectness.h"
#include <cassert>
#include "../utils/utils.h"

void objectness::filterSobel(cv::Mat &src, cv::Mat &dst) {
    // Src should not be empty
    assert(!src.empty());

    // Sobel masks
    int filterX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int filterY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    // Create dst matrix if its empty
    if (dst.empty()) {
        dst = cv::Mat(src.size(), src.type());
    }

    int rows = src.rows, cols = src.cols;
    cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0); // Reduce noise

    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
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

void objectness::thresholdMinMax(cv::Mat &src, cv::Mat &dst, float min, float max) {
    // Both matrices should not be empty
    assert(!src.empty());
    assert(!dst.empty());

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

cv::Vec3i objectness::extractMinEdgels(std::vector<TemplateGroup> &templateGroups, float minThresh, float maxThresh) {
    assert(!templateGroups.empty());
    assert(minThresh >= 0);
    assert(maxThresh >= 0);

    int edgels = 0;
    cv::Vec3i output(INT_MAX, 0, 0);
    cv::Mat tplSobel, tplIntegral, tplNormalized;

    for (auto &group : templateGroups) {
        for (auto &t : group.templates) {
            t.srcDepth.convertTo(tplNormalized, CV_32F, 1.0f / 65536.0f);
            filterSobel(tplNormalized, tplSobel);
            thresholdMinMax(tplSobel, tplSobel, minThresh, maxThresh);
            cv::integral(tplNormalized, tplIntegral, CV_32F);
            edgels = static_cast<int>(tplIntegral.at<float>(tplIntegral.rows - 1, tplIntegral.cols - 1));

            if (edgels < output[0]) {
                output[0] = edgels;
                output[1] = t.srcDepth.cols;
                output[2] = t.srcDepth.rows;
            }
        }
    }

    return output;
}

cv::Rect objectness::objectness(cv::Mat &scene, cv::Mat &sceneDepthNormalized, cv::Mat &sceneColor, cv::Vec3i minEdgels, float minThresh, float maxThresh) {
    // Edgels count and template bounding box should be greater than 0
    assert(minEdgels[0] > 0);
    assert(minEdgels[1] > 0);
    assert(minEdgels[2] > 0);

    // Matrices should not be empty
    assert(!scene.empty());
    assert(!sceneDepthNormalized.empty());
    assert(!sceneColor.empty());

    // Take first template [just for demonstration]
    cv::Mat sceneSobel;

    // Apply sobel filter on template and scene
    filterSobel(sceneDepthNormalized, sceneSobel);

    // Apply thresh
    thresholdMinMax(sceneSobel, sceneSobel, minThresh, maxThresh);

    // Calculate image integral
    cv::Mat sceneIntegral;
    cv::integral(sceneSobel, sceneIntegral, CV_32F);

    // Set min number of edgels to 30% of original
    minEdgels[0] *= 0.30;

    // Helper vars
    std::vector<cv::Vec4i> windows;
    int sizeX = minEdgels[1] / 2, sizeY = minEdgels[2] / 2; // Set sliding window size to half of original window
    int stepX = sizeX / 3, stepY = sizeY / 3; // Set step to third of template width for better BB detection

    // Slide window over scene and calculate edgel count for each overlap
    for (int y = 0; y < sceneSobel.rows - sizeY; y += stepY) {
        for (int x = 0; x < sceneSobel.cols - sizeX; x += stepX) {

            // Calc edgel value in current sliding window with help of image integral
            float sceneEdgels = sceneIntegral.at<float>(y + sizeY, x + sizeX)
                - sceneIntegral.at<float>(y, x + sizeX)
                - sceneIntegral.at<float>(y + sizeY, x)
                + sceneIntegral.at<float>(y, x);

            // Check if current window contains at least 30% of tpl edgels, if yes, save window variables
            if (sceneEdgels >= minEdgels[0]) {
                windows.push_back(cv::Vec4i(x, y, x + sizeX, y + sizeY));
#ifdef DEBUG
            cv::rectangle(sceneColor, cv::Point(x, y), cv::Point(x + sizeX, y + sizeY), cv::Vec3b(190, 190, 190));

            // Draw text into corresponding rect with edgel count
            std::stringstream ss;
            ss << "T: " << minEdgels[0];
            cv::putText(sceneColor, ss.str(), cv::Point(x + 3, y + sizeY - 20), CV_FONT_HERSHEY_SIMPLEX, 0.45, cv::Vec3b(190, 190, 190));
            ss.str("");
            ss << "S: " << sceneEdgels;
            cv::putText(sceneColor, ss.str(), cv::Point(x + 3, y + sizeY - 5), CV_FONT_HERSHEY_SIMPLEX, 0.45, cv::Vec3b(190, 190, 190));
#endif
            }
        }
    }

    // Calculate coordinates of outer BB
    int minX = sceneSobel.cols, maxX = 0;
    int minY = sceneSobel.rows, maxY = 0;
    for (int i = 0; i < windows.size(); i++) {
        minX = std::min(minX, windows[i][0]);
        minY = std::min(minY, windows[i][1]);
        maxX = std::max(maxX, windows[i][2]);
        maxY = std::max(maxY, windows[i][3]);
    }

#ifdef DEBUG
    // Draw outer BB based on max/min values of all smaller boxes
    cv::rectangle(sceneColor, cv::Point(minX, minY), cv::Point(maxX, maxY), cv::Vec3b(0, 255, 0), 2);

    // Show results
    cv::imshow("Depth Scene", sceneDepthNormalized);
    cv::imshow("Sobel Scene", sceneSobel);
    cv::imshow("Scene", sceneColor);
    cv::waitKey(0);
#endif

    // Return resulted BB window
    return cv::Rect(minX, minY, maxX - minX, maxY - minY);
}