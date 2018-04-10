#include <iostream>
#include <opencv2/opencv.hpp>
#include "objdetect/classifier.h"
#include "glcore/mesh.h"
#include "utils/glutils.h"
#include "utils/converter.h"
#include "core/particle.h"
#include "processing/processing.h"

int main() {
    // Convert templates from t-less to custom format
//    tless::Converter converter;
//    converter.convert("data/convert_primesense.txt", "data/models/", "data/108x108/primesense/", 108);
//    converter.convert("data/convert_kinectv2.txt", "data/models/", "data/398x398/kinectv2/", 398);

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

//     Run classifier
    // Primesense
//    classifier.train("data/templates_primesense.txt", "data/trained/primesense/");
//    classifier.detect("data/trained_primesense.txt", "data/trained/primesense/", "data/scenes/primesense/02/");

    // Kinect
//    classifier.train("data/templates_kinectv2.txt", "data/trained/kinectv2/");
    classifier.detect("data/trained_kinectv2.txt", "data/trained/kinectv2/", "data/scenes/kinectv2/02/");

    return 0;
}