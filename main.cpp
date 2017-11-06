#include <iostream>
#include "utils/timer.h"
#include "objdetect/classifier.h"

int main() {
    // Init classifier
    Classifier classifier;

    // Init indices
//    std::vector<uint> indices = { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 };

    // Run classifier
//    classifier.train("data/templates.txt", "data/trained/", { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 });
//    classifier.train("data/templates.txt", "data/trained/");
    classifier.detect("data/trained_templates.txt", "data/trained/");

    return 0;
}