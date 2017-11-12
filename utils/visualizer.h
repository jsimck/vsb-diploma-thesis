#ifndef VSB_SEMESTRAL_PROJECT_VISUALIZER_H
#define VSB_SEMESTRAL_PROJECT_VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <memory>
#include "../core/window.h"
#include "../core/template.h"
#include "../core/classifier_criteria.h"
#include "../core/hash_table.h"
#include "../core/match.h"

class Visualizer {
private:
    static void visualizeWindow(cv::Mat &scene, Window &window);
    static cv::Vec3b heatMapValue(int min, int max, int value);
public:
    static void visualizeHashing(cv::Mat &scene, cv::Mat &sceneDepth, std::vector<HashTable> &tables, std::vector<Window> &windows,
                                     std::shared_ptr<ClassifierCriteria> &criteria, bool continuous, int wait = 0, const char *title = nullptr);
    static void visualizeMatches(cv::Mat &scene, std::vector<Match> &matches, const std::string &templatesPath = "data/", int wait = 0, const char *title = nullptr);
    static void visualizeWindows(cv::Mat &scene, std::vector<Window> &windows, bool continuous, int wait = 0, const char *title = nullptr);
    static void visualizeTests(Template &tpl, const cv::Mat &sceneHSV, Window &window, std::vector<cv::Point> &stablePoints, std::vector<cv::Point> &edgePoints,
                               cv::Range &neighbourhood, std::vector<int> &scoreII, std::vector<int> &scoreIII, float scoreIV, std::vector<int> &scoreV,
                               int pointsCount, int minThreshold, bool continuous = true, const std::string &templatesPath = "data/",
                               int wait = 0, const char *title = nullptr);

    // Templates
    static void visualizeTemplate(Template &tpl, const std::string &templatesPath = "data/", int wait = 0, const char *title = nullptr);

    // Utils
    static void setLabel(cv::Mat &im, const std::string &label, const cv::Point &origin, int padding = 1, int fontFace = CV_FONT_HERSHEY_SIMPLEX, double scale = 0.4,
                         const cv::Scalar &fColor = cv::Scalar(255, 255, 255), const cv::Scalar &bColor = cv::Scalar(0, 0, 0), int thickness = 1);
};


#endif //VSB_SEMESTRAL_PROJECT_VISUALIZER_H
