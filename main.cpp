#include <iostream>
#include <opencv2/opencv.hpp>
#include "objdetect/classifier.h"
#include "processing/processing.h"
#include "utils/converter.h"

//using namespace cv;
//using namespace tless;
//
//// Global Variables
//int minMag = 100;
//
//cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());
//Parser parser(criteria);
//
//void on_trackbar(int, void *) {
//    std::cout << "scale: " << minMag;
//    std::cout << std::endl;
//
//    criteria->minMagnitude = minMag;
//    Scene scene = parser.parseScene("data/scenes/02/", 400, 1.0f);
//
//    cv::Mat testGray, mags, testRGB;
//    quantizedGradients(scene.srcRGB, testRGB, criteria->minMagnitude);
//    quantizedOrientationGradients(scene.srcGray, testGray, mags);
//
//    testGray *= 16;
//    testRGB *= 16;
//
//    imshow("Scene - testGray", testGray);
//    imshow("Scene - testRGB", testRGB);
//}
//
//int main(int argc, char **argv) {
//    std::cout << *criteria << std::endl;
//
//    /// Create Trackbars
//    namedWindow("Controls", cv::WINDOW_FREERATIO);
//    createTrackbar("minMag", "Controls", &minMag, 200, on_trackbar);
//
//    /// Show some stuff
//    on_trackbar(minMag, 0);
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
    classifier.detect("data/trained.txt", "data/trained/", "data/scenes/02/");

    return 0;
}