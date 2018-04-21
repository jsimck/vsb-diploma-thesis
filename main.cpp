#include <iostream>
#include <opencv2/opencv.hpp>
#include "objdetect/classifier.h"
#include "processing/processing.h"
#include "utils/converter.h"
#include "utils/evaluator.h"

static const int SENSOR_KINECT = 0;
static const int SENSOR_PRIMESENSE = 1;
static const int SENSOR_CURRENT = SENSOR_KINECT;

int main() {

    // Dataset pairs (sceneId, templates)
    std::vector<std::pair<int, std::vector<int>>> data = {
//        {1, {2, 25, 29, 30}},
        {2, {5, 6, 7}},
//        {3, {5, 8, 11, 12, 18}},
//        {4, {5, 8, 26, 28}},
//        {5, {1, 4, 9, 10, 27}},
//        {6, {6, 7, 11, 12}},
//        {7, {1, 3, 13, 14, 15, 16, 17, 18}},
//        {8, {19, 20, 21, 22, 23, 24}},
//        {9, {1, 2, 3, 4}},
//        {10, {19, 20, 21, 22, 23, 24}},
//        {11, {5, 8, 9, 10}},
//        {12, {2, 3, 7, 9}},
//        {13, {19, 20, 21, 23, 28}},
//        {14, {19, 20, 22, 23, 24}},
//        {15, {25, 26, 27, 28, 29, 30}},
//        {16, {10, 11, 12, 13, 14, 15, 16, 17}},
//        {17, {1, 4, 7, 9}},
//        {18, {1, 4, 7, 9}},
//        {19, {13, 14, 15, 16, 17, 18, 24, 30}},
//        {20, {1, 2, 3, 4}},
    };

    // Custom criteria
    cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());

    // Training params
    criteria->tablesCount = 100;
    criteria->minVotes = 3;
    criteria->depthBinCount = 5;
    criteria->tablesTrainingMultiplier = 15;
    criteria->depthK = 0.5f;
    criteria->depthDeviation = 0.85f;

    // Detect params
    criteria->matchFactor = 0.55f;

    // Define path to files
    const char *sensorPath = (SENSOR_CURRENT == SENSOR_KINECT) ? "kinectv2" : "primesense";
    std::string templatesPath = cv::format("data/108x108/%s/", sensorPath);
    std::string scenesPath = cv::format("data/scenes/%s/", sensorPath);
    std::string convertPath = cv::format("data/400x400/%s/", sensorPath);
    std::string trainedPath = "data/trained/%s/%02d/";
    std::string resultsPath = "data/results/%s/%02d/";
    std::string modelsPath = "data/models/";

    // Init classifier
    tless::Evaluator eval(scenesPath, 0.3f);
    tless::Classifier classifier(criteria);

    // Run classification on defined dataset
    for (auto &scene : data) {
        // Results and trained paths
        std::string sceneResultsPath = cv::format(resultsPath.c_str(), sensorPath, scene.first);
        std::string sceneTrainedPath = cv::format(trainedPath.c_str(), sensorPath, scene.first);

        // Train
//        classifier.train(templatesPath, scene.second);
//        classifier.save(sceneTrainedPath);

        // Detect
        classifier.load(sceneTrainedPath);
        classifier.detect(scenesPath, {scene.first}, sceneResultsPath, 0, 504);

        // Evaluate
        eval.evaluate(sceneResultsPath, { scene.first });
    }

    return 0;
}