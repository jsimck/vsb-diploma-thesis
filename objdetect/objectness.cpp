#include "objectness.h"
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>

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

void edgeBasedObjectness(cv::Mat &scene, cv::Mat &sceneDepth, std::vector<Template> &templates) {
    // Set threshold
    uchar threshold = 100;

    // Take first template [just for demonstration]
    cv::Mat tpl = templates[0].srcDepth;

    // Run sobel over the image
    cv::Mat tplSobel_x, tplSobel_y, tplSobel;
    cv::Sobel(tpl, tplSobel_x, CV_32F, 1, 0);
    cv::Sobel(tpl, tplSobel_y, CV_32F, 0, 1);
    cv::addWeighted(tplSobel_x, 0.5, tplSobel_y, 0.5, 0, tplSobel);
    cv::threshold(tplSobel, tplSobel, 0, 255, CV_THRESH_BINARY);

    // Count edgels in template
    int tplEdgels = 0;
    for (int y = 0; y < tplSobel.rows; y++) {
        for (int x = 0; x < tplSobel.cols; x++) {
            if (tplSobel.at<uchar>(y, x) > threshold) {
                tplEdgels++;
            }
        }
    }

//    tplEdgels *= 0.1;

    // Calculate edgels in sceneDepth over each sliding window
    cv::Mat sceneSobel_x, sceneSobel_y, sceneSobel;
    cv::Sobel(sceneDepth, sceneSobel_x, -1, 1, 0);
    cv::Sobel(sceneDepth, sceneSobel_y, -1, 0, 1);
    cv::addWeighted(sceneSobel_x, 0.5, sceneSobel_y, 0.5, 0, sceneSobel);
    cv::threshold(sceneSobel, sceneSobel, 0, 255, CV_THRESH_BINARY);

    // Helper vars
    int sizeX = tpl.cols;
    int sizeY = tpl.rows;
    int step = sizeX;

    // Slide window over scene and calculate edgel count for each overlap
    std::vector<cv::Rect> windows;
    for (int y = 0; y < sceneSobel.rows - sizeY; y += step) {
        for (int x = 0; x < sceneSobel.cols - sizeX; x += step) {
            int sceneEdgels = 0;

            for (int yy = y; yy < y + sizeY; yy++) {
                for (int xx = y; xx < x + sizeX; xx++) {
                    if (sceneSobel.at<uchar>(yy, xx) > threshold) {
                        sceneEdgels++;
                    }
                }
            }


            if (sceneEdgels >= tplEdgels) {
                cv::rectangle(scene, cv::Point(x, y), cv::Point(x + sizeX, y + sizeY), 255);
                cv::imshow("Scene", scene);
                windows.push_back(cv::Rect(x, y, x + sizeX, y + sizeY));
            } else {
                std::cout << sceneEdgels << std::endl;
                std::cout << tplEdgels << std::endl;
            }
        }
    }

    // Draw rectanges on matched windows
    for (int j = 0; j < windows.size(); j++) {
        cv::Rect window = windows[j];
        cv::rectangle(scene, cv::Point(window.x, window.y), cv::Point(window.width, window.height), 255);
    }

    cv::imshow("Sobel Scene", sceneSobel);
    cv::imshow("Sobel Template", tplSobel);
    cv::imshow("Scene", scene);
    cv::imshow("Depth", sceneDepth);
    cv::waitKey(0);
}
