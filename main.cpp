#include <iostream>
#include "utils/timer.h"
#include "objdetect/classifier.h"

int main() {
    // Init classifier
    const std::vector<std::string> tplNames = { "02", "30", "29", "25" };
    Classifier classifier("scene_01/", "_0000.png");

    // Init indices
//    std::vector<uint> indices = { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 };

    // Run classifier
//    classifier.setIndices(indices);
//    classifier.train("data/templates.txt", "trained/", { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 });
//    classifier.train("data/templates.txt", "data/trained/", { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 });
    classifier.detect("data/trained/");

    return 0;
}