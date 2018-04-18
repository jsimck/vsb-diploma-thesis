#include <iostream>
#include <opencv2/opencv.hpp>
#include "objdetect/classifier.h"
#include "processing/processing.h"
#include "utils/converter.h"
#include "utils/evaluator.h"

int main() {
    // Convert templates from t-less to custom format
//    tless::Converter converter;
//    converter.convert("data/convert_kinectv2.txt", "data/models/", "data/108x108/kinectv2/", 108);
//    converter.convert("data/convert_primesense.txt", "data/models/", "data/108x108/primesense/", 108);

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
    // Primesense
//    classifier.train("data/templates_primesense.txt", "data/trained/primesense/");
//    classifier.detect("data/trained_primesense.txt", "data/trained/primesense/", "data/scenes/primesense/02/");

    // Kinect
//    classifier.train("data/templates_kinectv2.txt", "data/trained/kinectv2/");
    classifier.detect("data/trained_kinectv2.txt", "data/trained/kinectv2/", "data/shaders/", "data/meshes.txt",
                      "data/scenes/kinectv2/02/", "data/results.yml");

    tless::Evaluator eval("data/scenes/kinectv2/", 0.5f);
    eval.evaluate("data/results.yml", 2);

    return 0;
}