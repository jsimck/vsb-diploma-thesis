#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/functional/hash.hpp>
#include "objdetect/matching_deprecated.h"
#include "objdetect/objectness.h"
#include "utils/timer.h"
#include "objdetect/hasher.h"
#include "objdetect/classifier.h"


int main() {
    // Init classifier
    const std::vector<std::string> tplNames = { "02", "25", "29", "30" };
    Classifier classifier("data/", tplNames, "scene_01/", "0000.png");

    // Init indices
    std::unique_ptr<std::vector<int>> indices { new std::vector<int>() };
    const int indiciesDataSize = 14;
    int indiciesData[] = { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 };
    std::copy(&indiciesData[0], &indiciesData[indiciesDataSize], std::back_inserter(*indices));

    // Run classifier
    classifier.classify();
//    classifier.classifyTest(indices);

    return 0;
}