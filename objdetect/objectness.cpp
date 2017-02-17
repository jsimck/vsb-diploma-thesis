#include "objectness.h"
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include "../utils/utils.h"

#ifndef DEBUG
#define DEBUG
#endif

void filterSobel(cv::Mat &src, cv::Mat &dst) {
    // Sobel masks
    int filterX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int filterY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    int rows = src.rows, cols = src.cols;
    cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0); // Reduce noise

    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            int i = 0;
            double sumX = 0.0, sumY = 0.0;
            for (int yy = 0; yy < 3; yy++) {
                for (int xx = 0; xx < 3; xx++) {
                    double px = src.at<double>(yy + y - 1, x + xx - 1);
                    sumX += px * filterX[i];
                    sumY += px * filterY[i];
                    i++;
                }
            }

            // Add sum of x and y derivatives
            dst.at<double>(y, x) = sqrt(SQR(sumX) + SQR(sumY));
        }
    }
}

void thresholdMinMax(cv::Mat &src, cv::Mat &dst, double min, double max) {
    src = dst.clone();

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (src.at<double>(y, x) >= min && src.at<double>(y, x) <= max) {
                dst.at<double>(y, x) = 1.0;
            } else {
                dst.at<double>(y, x) = 0.0;
            }
        }
    }
}

cv::Vec4i edgeBasedObjectness(cv::Mat &scene, cv::Mat &sceneDepth, cv::Mat &sceneColor, std::vector<Template> &templates,
                              double thresh) {
    // Take first template [just for demonstration]
    cv::Mat sceneSobel, tplSobel;
    cv::Mat tpl = templates[0].srcDepth;

    // Apply sobel filter on template and scene
    filterSobel(tpl, tplSobel);
    filterSobel(sceneDepth, sceneSobel);

    // Apply thresh
    cv::threshold(tplSobel, tplSobel, thresh, 1.0, CV_THRESH_BINARY);
    thresholdMinMax(sceneSobel, sceneSobel, 0.01, 0.1);

    // Calculate image integral
    cv::Mat tplIntegral, sceneIntegral;
    cv::integral(tplSobel, tplIntegral, CV_64F);
    cv::integral(sceneSobel, sceneIntegral, CV_64F);

    // Set min number of edgels to 30% of original
    tplIntegral *= 0.3;
    double tplEdgels = tplIntegral.at<double>(tplIntegral.rows - 1, tplIntegral.cols - 1);

    // Helper vars
    std::vector<cv::Vec4i> windows;
    int sizeX = tpl.cols, sizeY = tpl.rows;
    int stepX = tpl.cols / 3, stepY = tpl.rows / 3;// Set step to half of template width for better BB detection

    // Slide window over scene and calculate edgel count for each overlap
    for (int y = 0; y < sceneSobel.rows - sizeY; y += stepY) {
        for (int x = 0; x < sceneSobel.cols - sizeX; x += stepX) {

            // Calc edgel value in current sliding window with help of image integral
            double sceneEdgels = sceneIntegral.at<double>(y + sizeY, x + sizeX)
                - sceneIntegral.at<double>(y, x + sizeX)
                - sceneIntegral.at<double>(y + sizeY, x)
                + sceneIntegral.at<double>(y, x);

            // Check if current window contains at least 30% of tpl edgels, if yes, save window variables
            if (sceneEdgels >= tplEdgels) {
                windows.push_back(cv::Vec4i(x, y, x + sizeX, y + sizeY));
#ifdef DEBUG
            // Draw rect
//            cv::rectangle(scene, cv::Point(x, y), cv::Point(x + sizeX, y + sizeY), 0.8);
            cv::rectangle(sceneColor, cv::Point(x, y), cv::Point(x + sizeX, y + sizeY), cv::Vec3b(190, 190, 190));

            // Draw text into corresponding rect with edgel count
            std::stringstream ss;
            ss << "T: " << tplEdgels;
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
        if (minX > windows[i][0]) minX = windows[i][0];
        if (minY > windows[i][1]) minY = windows[i][1];
        if (maxX < windows[i][2]) maxX = windows[i][2];
        if (maxY < windows[i][3]) maxY = windows[i][3];
    }

#ifdef DEBUG
    // Draw outer BB based on max/min values of all smaller boxes
//    cv::rectangle(scene, cv::Point(minX, minY), cv::Point(maxX, maxY), 1.0f, 3);
    cv::rectangle(sceneColor, cv::Point(minX, minY), cv::Point(maxX, maxY), cv::Vec3b(0, 255, 0), 2);

    // Show results
    cv::imshow("Depth Scene", sceneDepth);
    cv::imshow("Sobel Scene", sceneSobel);
    cv::imshow("Scene", sceneColor);
    cv::waitKey(1);
#endif

    // Return resulted BB window
    return cv::Vec4i(minX, minY, maxX, maxY);
}
