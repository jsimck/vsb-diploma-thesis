#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_H


#include <string>
#include <opencv2/opencv.hpp>

struct Template {
public:
    Template(std::string fileName, cv::Rect bounds, cv::Mat &src, cv::Mat &srcDepth);

    std::string fileName;
    cv::Rect bounds;
    cv::Mat src;
    cv::Mat srcDepth;

    void print();
};


#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_H
