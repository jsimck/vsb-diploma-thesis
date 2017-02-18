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
    cv::Mat scene;
    cv::Mat sceneColor = cv::imread("/Users/jansimecek/ClionProjects/vsb-semestral-project/data/scene_01/rgb/0244.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat sceneDepth = cv::imread("data/scene_01/depth/0244.png", CV_LOAD_IMAGE_UNCHANGED);

    // Convect to grayscale
    cv::cvtColor(sceneColor, scene, CV_BGR2GRAY);

    // Convert to double
    scene.convertTo(scene, CV_64FC1, 1.0f / 255.0f);
    sceneDepth.convertTo(sceneDepth, CV_64FC1, 1.0f / 65536.0f);

    // Load templates
    std::cout << "Parsing... ";
    parser.parse(templates);
    std::cout << "DONE!" << std::endl;

    // Stop Matching time
    Timer t;

    // Edge based objectness
    cv::Rect objectsRoi;
    objectsRoi = edgeBasedObjectness(scene, sceneDepth, sceneColor, templates, 0.01);

    // Match templates
    std::vector<cv::Rect> matchBB;

    std::cout << "Matching... ";
    matchBB = matchTemplate(scene, objectsRoi, templates);
    std::cout << "DONE!" << std::endl;

    // Print elapsed time
    std::cout << "Elapsed: " << t.elapsed() << "s";

    // Show results
    for (int i = 0; i < matchBB.size(); i++) {
        cv::rectangle(sceneColor, matchBB[i], cv::Vec3b(0, 255, 0), 1);
    }

    cv::imshow("Result", sceneColor);
    cv::waitKey(0);
    return 0;
}