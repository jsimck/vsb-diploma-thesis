#include <iostream>
#include <opencv2/opencv.hpp>
#include "objdetect/classifier.h"
#include "utils/converter.h"

int main() {
    // Convert templates from t-less to custom format
//    tless::Converter converter;
//    converter.convert("data/convert.txt", "data/models/", "data/108x108/", 108);

    // Custom criteria
    cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());
    criteria->matchFactor = 0.4f;

    // Init classifier
    tless::Classifier classifier(criteria);

    // Run classifier
//    classifier.train("data/templates.txt", "data/trained/", { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 });
    classifier.train("data/templates.txt", "data/trained/");
//    classifier.detect("data/trained_templates.txt", "data/trained/", "data/scene_01/");

    return 0;
}