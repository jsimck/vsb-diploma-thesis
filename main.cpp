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

    // Train
//    classifier.train("data/108x108/kinectv2/", {5, 6, 7, 25, 29, 30});
//    classifier.save("data/trained/kinectv2/");

    // Detect
    classifier.load("data/trained/kinectv2/");
    classifier.detect("data/scenes/kinectv2/", {1, 2}, "data/results/");

    // Evaluate
    tless::Evaluator eval("data/scenes/kinectv2/", 0.5f);
    eval.evaluate("data/results/results_01.yml.gz", 1);
    eval.evaluate("data/results/results_02.yml.gz", 2);

    return 0;
}