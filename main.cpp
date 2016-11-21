#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include "utils/template_parser.h"
#include "objdetect/matching.h"
#include "objdetect/objectness.h"
#include "utils/timer.h"

int main() {
    // Parse rgb templates
    std::vector<Template> templates;
    TemplateParser parser = TemplateParser("data/obj_01");
    parser.setTplCount(10);

    // Load scene
    cv::Mat scene = cv::imread("data/scene_01/rgb/0000.png");
    cv::Mat sceneDepth = cv::imread("data/scene_01/depth/0000.png");

    // Load templates
    std::cout << "Parsing... ";
    parser.parse(templates);
    std::cout << "DONE!" << std::endl;

    // Edge based objectness
//    edgeBasedObjectness(scene, sceneDepth, templates);

    // BING generate NG
//    std::cout << "Generating BING normed gradients... ";
//    generateBINGTrainingSet("data/objectness", templates);
//    std::cout << "DONE!" << std::endl;

    // Compute BING
//    std::cout << "BING... ";
//    std::vector<cv::Vec4i> bingBB;
//    computeBING("data/objectness", scene, bingBB);
//    computeBING("data/bing", scene, bingBB);
//    std::cout << "DONE!";

    // Match templates
//    std::vector<cv::Rect> matchBB;
//    Timer t;
//
//    std::cout << "Matching... ";
//    matchTemplate(scene, templates, matchBB);
//    std::cout << "DONE!" << std::endl;
//    std::cout << "Elapsed: " << t.elapsed() << "s";
//
//    // Show results
//    for (int i = 0; i < matchBB.size(); i++) {
//        cv::rectangle(scene, matchBB[i], cv::Vec3b(0, 255, 0), 1);
//    }
//
//    cv::imshow("Result", scene);
//    cv::waitKey(0);
    return 0;
}