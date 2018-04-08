#include <iostream>
#include <opencv2/opencv.hpp>
#include "objdetect/classifier.h"
#include "glcore/mesh.h"
#include "utils/glutils.h"
#include "utils/converter.h"
#include "core/particle.h"
#include "processing/processing.h"

#define SQR(x) ((x) * (x))
using namespace tless;

// Random
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> dR(-0.2f, 0.2f);
static std::uniform_real_distribution<float> dT(-15, 15);
static std::uniform_real_distribution<float> dVT(0, 3);
static std::uniform_real_distribution<float> dVR(0, 0.1f);
static std::uniform_real_distribution<float> dRand(0, 1.0f);

float fitness(const cv::Mat &gt, cv::Mat &pose) {
    float sum = 0;

    cv::Mat edges;
    cv::Laplacian(pose, edges, CV_32FC1);
    cv::threshold(edges, edges, 0.01f, 1, CV_THRESH_BINARY);

    for (int y = 0; y < gt.rows; y++) {
        for (int x = 0; x < gt.cols; x++) {
            sum += gt.at<float>(y, x) > 0 && edges.at<float>(y, x) > 0;
        }
    }

    return sum;
}

float computeVelocity(float w, float vi, float xi, float pBest, float gBest, float c1, float c2, float r1, float r2) {
    return w * vi + (c1 * r1) * (pBest - xi) + (c2 * r2) * (gBest - xi);
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
    parser.parseObject("data/108x108/kinectv2/07/", templates, {28, 60});

    // Init GT depth
    cv::Mat gt, gtEdge, org;
    drawDepth(templates[0], gt);
    drawDepth(templates[1], org);

    // Do laplace
    cv::Laplacian(gt, gtEdge, CV_32FC1);
    cv::threshold(gtEdge, gtEdge, 0.01f, 1, CV_THRESH_BINARY);

    // Init particles
    cv::Mat pose;
    std::vector<Particle> particles;
    Particle gBest;
    gBest.fitness = 0;

    for (int i = 0; i < 50; ++i) {
        particles.emplace_back(dT(gen), dT(gen), dT(gen), dR(gen), dR(gen), dR(gen),
                               dVT(gen), dVT(gen), dVT(gen), dVR(gen), dVR(gen), dVR(gen));
        drawDepth(templates[1], pose, particles[i].model());
        particles[i].fitness = fitness(gtEdge, pose);

        if (particles[i].fitness > gBest.fitness) {
            gBest = particles[i];
        }
    }

    // Gbest before PSO
    std::cout << "pre-PSO - gBest: " << gBest.fitness << std::endl;
    Particle preGBest = gBest;
    cv::Mat imGBest;
    drawDepth(templates[1], imGBest, gBest.model());

    // PSO
    const float C1 = 0.25f, C2 = 0.25f, W = 0.95f;

    // Generations
    for (int i = 0; i < 50; i++) {
        std::cout << "Iteration: " << i << std::endl;

        for (auto &p : particles) {
            drawDepth(templates[1], pose, p.model());

            cv::imshow("pose 1", pose);

            // Compute velocity
            p.v1 = computeVelocity(W, p.v1, p.rx, p.pBest.rx, gBest.rx, C1, C2, dRand(gen), dRand(gen));
            p.v2 = computeVelocity(W, p.v2, p.ty, p.pBest.ty, gBest.ty, C1, C2, dRand(gen), dRand(gen));
            p.v3 = computeVelocity(W, p.v3, p.tz, p.pBest.tz, gBest.tz, C1, C2, dRand(gen), dRand(gen));
            p.v4 = computeVelocity(W, p.v4, p.rx, p.pBest.rx, gBest.rx, C1, C2, dRand(gen), dRand(gen));
            p.v5 = computeVelocity(W, p.v5, p.ry, p.pBest.ry, gBest.ry, C1, C2, dRand(gen), dRand(gen));
            p.v6 = computeVelocity(W, p.v6, p.rz, p.pBest.rz, gBest.rz, C1, C2, dRand(gen), dRand(gen));

            // Update
            p.update();

            // Fitness
            drawDepth(templates[1], pose, p.model());
            p.fitness = fitness(gtEdge, pose);

            // Check for pBest
            if (p.fitness > p.pBest.fitness) {
                p.updatePBest();
                drawDepth(templates[1], imGBest, gBest.model());
            }

            // Check for gBest
            if (p.fitness > gBest.fitness) {
                std::cout << gBest.fitness << " - ";
                gBest = p;
                std::cout << gBest.fitness << std::endl;
                drawDepth(templates[1], imGBest, gBest.model());
            }

            cv::imshow("org", org);
            cv::imshow("GT", gt);
            cv::imshow("gBest", imGBest);
            cv::imshow("pose 2", pose);
            cv::waitKey(1);
        }
    }



    // Test draw each pose
    std::cout << "gBest: " << gBest << std::endl;
    for (auto &particle : particles) {
        std::cout << particle << std::endl;
    }

    cv::Mat imPreGBest;
    drawDepth(templates[1], imPreGBest, preGBest.model());

    // Ground truth
    cv::imshow("imPreGBest", imPreGBest);
    cv::imshow("imGBest", imGBest);
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