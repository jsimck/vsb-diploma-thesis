#include <iostream>
#include <opencv2/opencv.hpp>
#include "objdetect/classifier.h"
#include "processing/processing.h"
#include "utils/converter.h"

//using namespace cv;
//using namespace tless;
//
///// Global Variables
//int scale = 100, depthDiff = 10, f = 107, maxDepth = 200;
//
///// Matrices to store images
//Mat srcNormals, roi, roiNormals;
//
//cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());
//Parser parser(criteria);
//Scene scene = parser.parseScene("data/scenes/02/", 0, 1.0f);
//
//void on_trackbar(int, void *) {
//    std::cout << "scale: " << scale / 100.0f;
//    std::cout << " | depthDiff: " << depthDiff * 10;
//    std::cout << " | maxDepth: " << maxDepth * 100;
//    std::cout << " | f: " << f * 10 * (scale / 100.0f);
//    std::cout << std::endl;
//
//    Scene scene = parser.parseScene("data/scenes/02/", 0, scale / 100.0f);
//    roi = scene.srcDepth(cv::Rect(270 * (scale / 100.0f), 220 * (scale / 100.0f), 290 * (scale / 100.0f), 250 * (scale / 100.0f)));
//
//    quantizedNormals(scene.srcDepth, srcNormals, f * 10 * (scale / 100.0f), f * 10 * (scale / 100.0f), maxDepth * 100,
//                     static_cast<int>((depthDiff * 10) / (scale / 100.0f)));
//    quantizedNormals(roi, roiNormals, f * 10 * (scale / 100.0f), f * 10 * (scale / 100.0f), maxDepth * 100,
//                     static_cast<int>((depthDiff * 10) / (scale / 100.0f)));
//
//    imshow("Scene - rgb", scene.srcRGB);
//    imshow("Scene - srcNormals", srcNormals);
//    imshow("Scene - roiNormals", roiNormals);
//}
//
//int main(int argc, char **argv) {
//    std::cout << *criteria << std::endl;
//
//    /// Create Trackbars
//    namedWindow("Controls", cv::WINDOW_FREERATIO);
//    createTrackbar("scale", "Controls", &scale, 200, on_trackbar);
//    createTrackbar("depthDiff", "Controls", &depthDiff, 200, on_trackbar);
//    createTrackbar("maxDepth", "Controls", &maxDepth, 400, on_trackbar);
//    createTrackbar("f", "Controls", &f, 200, on_trackbar);
//
//    /// Show some stuff
//    on_trackbar(scale, 0);
//
//    /// Wait until user press some key
//    waitKey(0);
//    return 0;
//}

int main() {
    // Convert templates from t-less to custom format
//    tless::Converter converter;
//    converter.convert("data/convert.txt", "data/400x400/models/", "data/108x108/", 108);

    // Custom criteria
    cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());

    // Training params
    criteria->tablesCount = 100;
    criteria->minVotes = 3;
    criteria->depthBinCount = 5;

    // Detect params
    criteria->matchFactor = 0.6f;

    // Init classifier
    tless::Classifier classifier(criteria);

    // Run classifier
//    classifier.train("data/templates.txt", "data/trained/", { 0, 20, 25, 29, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 });
//    classifier.train("data/templates.txt", "data/trained/");
    classifier.detect("data/trained_templates.txt", "data/trained/", "data/scenes/02/");

    return 0;
}