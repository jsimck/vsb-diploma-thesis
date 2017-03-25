#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include "utils/template_parser.h"
#include "objdetect/matching.h"
#include "objdetect/objectness.h"
#include "utils/timer.h"
#include "objdetect/hashing.h"

int main() {
    std::vector<TemplateGroup> templateGroups;
    std::vector<Template> templates;
    TemplateParser parser = TemplateParser("data/");

    // Load scene
    cv::Mat scene;
    cv::Mat sceneColor = cv::imread("data/scene_01/rgb/0000.png", CV_LOAD_IMAGE_COLOR);
    cv::Mat sceneDepth = cv::imread("data/scene_01/depth/0000.png", CV_LOAD_IMAGE_UNCHANGED);

    // Convect to grayscale
    cv::cvtColor(sceneColor, scene, CV_BGR2GRAY);

    // Convert to double
    scene.convertTo(scene, CV_32FC1, 1.0f / 255.0f);
    sceneDepth.convertTo(sceneDepth, CV_32FC1, 1.0f / 65536.0f);

    /// ***** PREPARATION STAGE - START *****
    // Parse templates (groups)
//    std::cout << "Parsing... " << std::endl;
//    std::vector<std::string> tplNames = { "02", "25", "29", "30" };
//    parser.parse(templateGroups, tplNames);
//    std::cout << "DONE! " << templateGroups.size() << " template groups parsed" << std::endl << std::endl;
//
//    // Get number of edgels of template containing least amount of them
//    std::cout << "Extracting minimum of template edgels... " << std::endl;
//    cv::Vec3i minEdgels = extractMinEdgels(templateGroups);
//    std::cout << "DONE! " << minEdgels << " minimum found" <<std::endl << std::endl;
    /// ***** PREPARATION STAGE - END *****


    /// Test DATA - START
    std::vector<int> indices = { 0, 20, 25, 23, 120, 250, 774, 998, 1100, 400, 478 };

    std::vector<Template> templatesObj1, templatesObj9;
    parser.parseTemplate(templatesObj1, "02", indices);
    parser.parseTemplate(templatesObj9, "25", indices);

    TemplateGroup group1("02", templatesObj1);
    TemplateGroup group2("25", templatesObj9);

    templateGroups.push_back(group1);
    templateGroups.push_back(group2);

    // Get number of edgels of template containing least amount of them
    std::cout << "Extracting minimum of template edgels... ";
    cv::Vec3i minEdgels = extractMinEdgels(templateGroups);
    std::cout << "extracted: " << minEdgels << std::endl;

//    std::cout << "Hashing";
//    // Hashing test
//    Hashing h;
//    h.train(templateGroups);
//    /// Test DATA - END
//    return 0;


    /// ***** MATCHING - START *****
    // Stop Matching time
    Timer t;

    // Edge based objectness
    cv::Rect objectsRoi;
    objectsRoi = objectness(scene, sceneDepth, sceneColor, minEdgels);

    // Match templates
    std::vector<cv::Rect> matchBB;

    std::cout << "Matching... ";
    matchBB = matchTemplate(scene, objectsRoi, templateGroups);
    std::cout << "DONE!" << std::endl;
    /// ***** MATCHING - END *****


    /// ***** RESULTS - START *****
    // Print elapsed time
    std::cout << "Elapsed: " << t.elapsed() << "s" << std::endl;
    std::cout << "BB Size: " << matchBB.size() << std::endl;

    // Show results
    for (int i = 0; i < matchBB.size(); i++) {
        cv::rectangle(sceneColor, matchBB[i], cv::Scalar(0, 255, 0), 1);
    }

    cv::imshow("Result", sceneColor);
    cv::waitKey(0);
    /// ***** RESULTS - END *****

    return 0;
}