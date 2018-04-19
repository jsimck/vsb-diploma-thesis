#include <iostream>
#include <opencv2/opencv.hpp>
#include "objdetect/classifier.h"
#include "processing/processing.h"
#include "utils/converter.h"
#include "utils/evaluator.h"

static const int SENSOR_KINECT = 0;
static const int SENSOR_PRIMESENSE = 1;
static const int SENSOR_CURRENT = SENSOR_PRIMESENSE;

int main() {
    // Custom criteria
    cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());

    // Training params
    criteria->tablesCount = 100;
    criteria->minVotes = 3;
    criteria->depthBinCount = 5;
    criteria->tablesTrainingMultiplier = 15;
    criteria->depthK = 0.5f;

    // Detect params
    criteria->matchFactor = 0.6f;

    // Define path to files
    std::string modelsPath = "data/models/";
    std::string trainedPath = "data/trained/";
    std::string templatesPath = "data/108x108/";
    std::string convertPath = "data/400x400/";
    std::string resultsPath = "data/results_no_errode/";
    std::string scenesPath = "data/scenes/";

    // Update paths based on current sensor
    if (SENSOR_CURRENT == SENSOR_KINECT) {
        trainedPath += "kinectv2/";
        templatesPath += "kinectv2/";
        convertPath += "kinectv2/";
        resultsPath += "kinectv2/";
        scenesPath += "kinectv2/";
    } else {
        trainedPath += "primesense/";
        templatesPath += "primesense/";
        convertPath += "primesense/";
        resultsPath += "primesense/";
        scenesPath += "primesense/";
    }

    // Convert templates from t-less to custom format
//    tless::Converter converter;
//    converter.convert(convertPath, modelsPath, {5, 6, 7, 25, 29, 30}, templatesPath, 108);

    // Init classifier
    tless::Classifier classifier(criteria);

    // Train
//    classifier.train(templatesPath, {5, 6, 7, 25, 29, 30});
    classifier.train(templatesPath, {5, 6, 7});
    classifier.save(trainedPath);

    // Detect
    classifier.load(trainedPath);
    classifier.detect(scenesPath, {2}, resultsPath, 504);

    // Evaluate
    tless::Evaluator eval(scenesPath, 0.5f);
    eval.evaluate(resultsPath, {2});

    return 0;
}