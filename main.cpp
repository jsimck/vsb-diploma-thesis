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

    Timer t;
    // Load scene
    cv::Mat scene = cv::imread("/Users/jansimecek/ClionProjects/vsb-semestral-project/data/scene_01/rgb/0000.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat sceneDepth = cv::imread("data/scene_01/depth/0000.png", CV_LOAD_IMAGE_UNCHANGED);

    // Convert to float
    scene.convertTo(scene, CV_32FC1, 1.0f / 255.0f);
    sceneDepth.convertTo(sceneDepth, CV_32FC1, 1.0f / 65536.0f);

    // Load templates
    std::cout << "Parsing... ";
    parser.parse(templates);
    std::cout << "DONE!" << std::endl;

    // Edge based objectness
    edgeBasedObjectness(scene, sceneDepth, templates);
    std::cout << "Time: " << t.elapsed() << std::endl;

    // BING generate NG
//    std::cout << "Generating BING normed gradients... ";
//    generateBINGTrainingSet("data/objectness", templates);
//    std::cout << "DONE!" << std::endl;

//     Compute BING
//    std::cout << "BING... ";
//    std::vector<cv::Vec4i> bingBB;
//    computeBING("/Users/jansimecek/ClionProjects/vsb-semestral-project/data/objectness", scene, bingBB);
//    computeBING("/Users/jansimecek/ClionProjects/vsb-semestral-project/data/bing", scene, bingBB);
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