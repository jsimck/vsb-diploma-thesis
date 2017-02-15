#include "objectness.h"
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include "../utils/utils.h"

#ifndef DEBUG
#define DEBUG
#endif

void generateBINGTrainingSet(std::string destPath, std::vector<Template> &templates) {
    // Create file storage
    cv::FileStorage fsWrite;

    for (int i = 0; i < templates.size(); i++) {
        // Load template and crop
        cv::Mat templ = templates[i].src;

        // Run sobel
        cv::Mat templSobel_x, templSobel_y, templSobel;
        cv::Sobel(templ, templSobel_x, -1, 1, 0);
        cv::Sobel(templ, templSobel_y, -1, 0, 1);
        cv::addWeighted(templSobel_x, 0.5, templSobel_y, 0.5, 0, templSobel);

        // Calculate helper variables
        bool biggerHeight = templ.rows > templ.cols;
        int size = biggerHeight ? templ.rows : templ.cols;
        int offset = biggerHeight ? (size - templ.cols) / 2 : (size - templ.rows) / 2;
        int top = biggerHeight ? 0 : offset;
        int left = biggerHeight ? offset : 0;

        // Copy to keep ratio and Resize
        cv::Mat dest = cv::Mat::zeros(size, size, CV_32FC1);
        templSobel.copyTo(dest(cv::Rect(left, top, templSobel.cols, templSobel.rows)));
        cv::resize(templSobel, templSobel, cv::Size(8, 8));

//        cv::imshow("Image resized", templSobel);
//        cv::waitKey(0);

        // Save file
        fsWrite.open(destPath + "/ObjNessB2W8I." + templates[i].fileName + ".yml.gz", cv::FileStorage::WRITE);
        fsWrite << "ObjNessB2W8I" + templates[i].fileName << templSobel;
    }

    // Release
    fsWrite.release();
}

void computeBING(std::string trainingPath, cv::Mat &scene, std::vector<cv::Vec4i> &resultBB) {
    cv::saliency::ObjectnessBING objectnessBING;
    std::vector<cv::Vec4i> objectnessBoundingBox;
    objectnessBING.setTrainingPath(trainingPath);

    if (objectnessBING.computeSaliency(scene, objectnessBoundingBox) ) {
        std::vector<float> values = objectnessBING.getobjectnessValues();
        printf("detected candidates: %d\n", (int) objectnessBoundingBox.size());
        printf("scores: %d\n", (int) values.size());

        for (int i = 0; i < 20; i++) {
            cv::Vec4i bb = objectnessBoundingBox[i];
            printf("index=%d, value=%f\n", i, values[i]);
            rectangle(scene, cv::Point(bb[0], bb[1]), cv::Point(bb[2], bb[3]), cv::Scalar(0, 0, 255), 4);

            cv::imshow("BING", scene);
            cv::waitKey(0);
        }
    }
}

void filterSobel(cv::Mat &src, cv::Mat &dst) {
    // Sobel masks
    int filterX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int filterY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    int rows = src.rows, cols = src.cols;
    dst = cv::Mat(rows, cols, CV_32FC1);

    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {

            int i = 0;
            float sumX = 0.0f, sumY = 0.0f;
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

cv::Vec4i edgeBasedObjectness(cv::Mat &scene, cv::Mat &sceneDepth, std::vector<Template> &templates) {
    // Set threshold
    const float THRESHOLD = 0.01f;

    // Take first template [just for demonstration]
    cv::Mat sceneSobel, tplSobel;
    cv::Mat tpl = templates[0].srcDepth;

    // Apply sobel filter on template and scene
    filterSobel(tpl, tplSobel);
    filterSobel(sceneDepth, sceneSobel);

    // Count edgels in template
    int tplEdgels = 0;

    for (int y = 0; y < tplSobel.rows; y++) {
        for (int x = 0; x < tplSobel.cols; x++) {
            if (tplSobel.at<float>(y, x) > THRESHOLD) {
                tplEdgels++;
            }
        }
    }

    // Set min number of edgels to 30% of original
    tplEdgels *= 0.3;

    // Helper vars
    std::vector<cv::Vec4i> windows;
    int sizeX = tpl.cols;
    int sizeY = tpl.rows;

    // Slide window over scene and calculate edgel count for each overlap
    for (int y = 0; y < sceneSobel.rows - sizeY; y += sizeY) {
        for (int x = 0; x < sceneSobel.cols - sizeX; x += sizeX) {
            int sceneEdgels = 0;

            // Count edgels in sliding window
            for (int yy = y; yy < y + sizeY; yy++) {
                for (int xx = x; xx < x + sizeX; xx++) {
                    if (sceneSobel.at<float>(yy, xx) > THRESHOLD) {
                        sceneEdgels++;
                    }
                }
            }

            float color = 0.35f;
            // Check if current window contains at least 30% of tpl edgels, if yes, save window variables
            if (sceneEdgels >= tplEdgels) {
                windows.push_back(cv::Vec4i(x, y, x + sizeX, y + sizeY));
                color = 8.0f;
            }

#ifdef DEBUG
            // Draw rect
            cv::rectangle(scene, cv::Point(x, y), cv::Point(x + sizeX, y + sizeY), color);

            // Draw text into corresponding rect with edgel count
            std::stringstream ss;
            ss << "T: " << tplEdgels;
            cv::putText(scene, ss.str(), cv::Point(x + 3, y + sizeY - 20), CV_FONT_HERSHEY_SIMPLEX, 0.45f, color);
            ss.str("");
            ss << "S: " << sceneEdgels;
            cv::putText(scene, ss.str(), cv::Point(x + 3, y + sizeY - 5), CV_FONT_HERSHEY_SIMPLEX, 0.45f, color);
#endif
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
    cv::rectangle(scene, cv::Point(minX, minY), cv::Point(maxX, maxY), 1.0f, 3);

    // Show results
    cv::imshow("Scene", scene);
    cv::imshow("Sobel Scene", sceneSobel);
    cv::imshow("Depth Scene", sceneDepth);
    cv::waitKey(0);
#endif

    // Return resulted BB window
    return cv::Vec4i(minX, minY, maxX, maxY);
}
