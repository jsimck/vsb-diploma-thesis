#include <iostream>
#include "utils/timer.h"
#include "objdetect/classifier.h"

int main() {
    // Init classifier
    Classifier classifier;

    std::set<float*> testSet;
    float *n1 = new float();
    float *n2 = new float();
    float *n3 = new float();
    float *n4 = new float();

    testSet.insert(n1);
    testSet.insert(n1);
    testSet.insert(n2);
    testSet.insert(n3);
    testSet.insert(n2);
    testSet.insert(n3);
    testSet.insert(n4);
    testSet.insert(n4);
    testSet.insert(n4);
    testSet.insert(n3);
    testSet.insert(n3);
    testSet.insert(n2);
    testSet.insert(n2);
    testSet.insert(n2);
    testSet.insert(n3);
    testSet.insert(n4);
    testSet.insert(n1);
    testSet.insert(n1);
    testSet.insert(n1);
    testSet.insert(n4);

    std::cout << "n1: " << n1 << std::endl;
    std::cout << "n2: " << n2 << std::endl;
    std::cout << "n3: " << n3 << std::endl;
    std::cout << "n4: " << n4 << std::endl;

    for (auto &item : testSet) {
        std::cout << item << std::endl;
    }

    // Run classifier
//    classifier.train("data/templates.txt", "data/trained/", "data/models/", { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 });
//    classifier.train("data/templates.txt", "data/trained/", "data/models/");
//    classifier.detect("data/trained_templates.txt", "data/trained/", "data/scene_01/");

    // TODO
    // 1) Test I - depth not being filtered as it should
    // 2) Visualizer for hashing is probably broken

    return 0;
}