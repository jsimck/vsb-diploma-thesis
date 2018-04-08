#include <iostream>
#include <opencv2/opencv.hpp>
#include "objdetect/classifier.h"
#include "glcore/mesh.h"
#include "utils/glutils.h"
#include "utils/converter.h"
#include "core/particle.h"

#define SQR(x) ((x) * (x))
using namespace tless;

float fitness(const cv::Mat &gt, cv::Mat &pose) {
    float sum = 0;

    for (int y = 0; y < gt.rows; y++) {
        for (int x = 0; x < gt.cols; x++) {
            sum += SQR(gt.at<float>(y, x) - pose.at<float>(y, x));
        }
    }

    return sum;
}

int main() {
    // Convert templates from t-less to custom format
//    tless::Converter converter;
//    converter.convert("data/convert_kinectv2.txt", "data/models/", "data/108x108/kinectv2/", 398);

    // Custom criteria
    cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());

    // Load templates
    std::vector<Template> templates;
    Parser parser(criteria);
    parser.parseObject("data/398x398/kinectv2/07/", templates, {28});

    // Init GT depth
    cv::Mat gt;
    drawDepth(templates[0], gt);

    // Random
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dR(-0.5f, 0.5f);
    static std::uniform_real_distribution<float> dT(-50, 50);

    // Init particles
    std::vector<Particle> particles;
    for (int i = 0; i < 10; ++i) {
        particles.emplace_back(dT(gen), dT(gen), dT(gen), dR(gen), dR(gen), dR(gen));
    }

    // Test draw each pose
    cv::Mat pose;
    for (auto &particle : particles) {
        // Draw depth
        drawDepth(templates[0], pose, particle.model());

        // Print fitness
        std::cout << fitness(gt, pose) << std::endl;
    }

    // Ground truth
    cv::imshow("GT", gt);
    cv::waitKey(0);

    return 0;
}

//int main() {
//    // Convert templates from t-less to custom format
//    tless::Converter converter;
////    converter.convert("data/convert_primesense.txt", "data/models/", "data/108x108/primesense/", 108);
//    converter.convert("data/convert_kinectv2.txt", "data/models/", "data/398x398/kinectv2/", 398);
//
//    // Custom criteria
//    cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());
//
//    // Training params
//    criteria->tablesCount = 100;
//    criteria->minVotes = 3;
//    criteria->depthBinCount = 5;
//
//    // Detect params
//    criteria->matchFactor = 0.6f;
//
//    // Init classifier
//    tless::Classifier classifier(criteria);
//
////     Run classifier
//    // Primesense
////    classifier.train("data/templates_primesense.txt", "data/trained/primesense/");
////    classifier.detect("data/trained_primesense.txt", "data/trained/primesense/", "data/scenes/primesense/02/");
//
//    // Kinect
////    classifier.train("data/templates_kinectv2.txt", "data/trained/kinectv2/");
////    classifier.detect("data/trained_kinectv2.txt", "data/trained/kinectv2/", "data/scenes/kinectv2/02/");
//
//    return 0;
//}