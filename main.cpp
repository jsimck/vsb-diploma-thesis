#include <iostream>
#include "objdetect/classifier.h"

int main() {
    // Custom criteria
    cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());
    criteria->matchFactor = 0.6f;

    // Init classifier
    tless::Classifier classifier(criteria);

    // Run classifier
//    classifier.train("data/templates.txt", "data/trained/", "data/models/", { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478, 1095, 1015, 72 });
//    classifier.train("data/templates.txt", "data/trained/", "data/models/");
    classifier.detect("data/trained_templates.txt", "data/trained/", "data/scene_01/");

    return 0;
}