#ifndef VSB_SEMESTRAL_PROJECT_VISUALIZER_H
#define VSB_SEMESTRAL_PROJECT_VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include "../core/window.h"
#include "../core/template.h"
#include "../core/dataset_info.h"
#include "../core/hash_table.h"
#include "../core/match.h"
#include "../core/group.h"

class Visualizer {
private:
    static void visualizeWindow(cv::Mat &scene, Window &window);
    static cv::Vec3b heatMapValue(int min, int max, int value);
public:
    static void visualizeHashing(cv::Mat &scene, cv::Mat &sceneDepth, std::vector<HashTable> &tables, std::vector<Window> &windows, DataSetInfo &info, cv::Size &grid, const char *title = nullptr);
    static void visualizeMatches(cv::Mat &scene, std::vector<Match> &matches, std::vector<Group> &groups);
    static void visualizeWindows(cv::Mat &scene, std::vector<Window> &windows, bool continuous = true, const char *title = nullptr);

    // Templates
    static void visualizeTemplate(Template &tpl, const char *title = nullptr);

    // Utils
    static void setLabel(cv::Mat &im, const std::__cxx11::string label, const cv::Point &origin, int padding = 1, int fontFace = CV_FONT_HERSHEY_SIMPLEX, double scale = 0.4
        , cv::Scalar fColor = cv::Scalar(255, 255, 255), cv::Scalar bColor = cv::Scalar(0, 0, 0), int thickness = 1);
};


#endif //VSB_SEMESTRAL_PROJECT_VISUALIZER_H
