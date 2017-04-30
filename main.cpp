#include <iostream>
#include "utils/timer.h"
#include "objdetect/classifier.h"

int main() {
    // Init classifier
    const std::vector<std::string> tplNames = { "02", "25", "29", "30" };
    Classifier classifier("data/", tplNames, "scene_01/", "0000.png");

    // Init indices
    std::vector<uint> indices = { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 };

    // Run classifier
//    classifier.setIndices(indices);
    classifier.classify();

    return 0;
}