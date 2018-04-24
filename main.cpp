#include <iostream>
#include <opencv2/opencv.hpp>
#include "objdetect/classifier.h"
#include "processing/processing.h"
#include "utils/converter.h"
#include "utils/evaluator.h"
#include "utils/timer.h"

static const int SENSOR_PRIMESENSE = 0;
static const int SENSOR_KINECT = 1;
static const int SENSOR_CURRENT = SENSOR_KINECT;
static const int RUNS = 1;

int main() {
    // Dataset pairs (sceneId, templates)
    std::vector<std::pair<int, std::vector<int>>> data = {
//        {1, {2, 25, 29, 30}},
//        {2, {5, 6, 7}},
//        {3, {5, 8, 11, 12, 18}},
//        {4, {5, 8, 26, 28}},
//        {5, {1, 4, 9, 10, 27}},
//        {6, {6, 7, 11, 12}},
        {7, {1, 3, 13, 14, 15, 16, 17, 18}},
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
        {19, {13, 14, 15, 16, 17, 18, 24, 30}},
//        {20, {1, 2, 3, 4}},
//        {0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}},
    };

    for (int sensor = 1; sensor < 2; ++sensor) {
        // Custom criteria
        cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());

        // Start date
        std::stringstream buffer;
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        buffer << std::put_time(&tm, "%j_%H_%M");
        std::string currDate = buffer.str();
//        currDate = "set_per_scene_" + currDate;
        currDate = "set_per_scene_1";

        // Define path to files
        const char *sensorPath = (sensor == SENSOR_KINECT) ? "kinectv2" : "primesense";
        std::string templatesPath = cv::format("data/108x108/%s/", sensorPath);
        std::string scenesPath = cv::format("data/scenes/%s/", sensorPath);
        std::string convertPath = cv::format("data/400x400/%s/", sensorPath);
        std::string trainedPath = "data/trained/" + currDate + "/%s/%02d/";
        std::string resultsPath = "data/results/" + currDate + "/%s/%02d/";
        std::string modelsPath = "data/models/";

        // Init classifier
        tless::Evaluator eval(scenesPath, 0.3f);

        // Run classification on defined dataset
        for (auto &scene : data) {
            for (int i = 0; i < RUNS; ++i) {
                // Results and trained paths
                tless::Timer tt;
                std::string sceneResultsPath = cv::format(resultsPath.c_str(), sensorPath, scene.first);
                std::string sceneTrainedPath = cv::format(trainedPath.c_str(), sensorPath, scene.first);
                std::string resultsFileFormat = cv::format("results_%02d", i) + "_%02d.yml.gz";
                std::string trainedFileFormat = cv::format("trained_%02d", i) + "_%02d.yml.gz";
                std::string classifierFileName = cv::format("classifier_%02d.yml.gz", i);

                // Train
    //            std::cout.setstate(std::ios_base::failbit); // Disable cout
                tless::Classifier classifier(criteria);
//                classifier.train(templatesPath, scene.second);
//                classifier.save(sceneTrainedPath, classifierFileName, trainedFileFormat);

                // Detect
                classifier.load(sceneTrainedPath, classifierFileName, trainedFileFormat);
                if (scene.first == 0) {
                    classifier.detect(scenesPath, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 }, sceneResultsPath, 0, 504, resultsFileFormat);
                } else {
                    classifier.detect(scenesPath, { scene.first }, sceneResultsPath, 0, 504, resultsFileFormat);
                }

                // Evaluate
    //            std::cout.clear();
                std::cout << "Run: " << (i + 1) << ", " << "took: " << tt.elapsed() << "s" << ", ";
                eval.evaluate(sceneResultsPath, { scene.first }, resultsFileFormat);
            }
        }
    }

    return 0;
};