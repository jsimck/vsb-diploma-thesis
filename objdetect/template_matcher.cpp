#include "template_matcher.h"
#include "../utils/utils.h"
#include "objectness.h"

uint TemplateMatcher::getFeaturePointsCount() const {
    return featurePointsCount;
}

void TemplateMatcher::setFeaturePointsCount(uint featurePointsCount) {
    assert(featurePointsCount > 0);
    this->featurePointsCount = featurePointsCount;
}

void TemplateMatcher::match(const cv::Mat &srcColor, const cv::Mat &srcGrayscale, const cv::Mat &srcDepth,
                     std::vector<Window> &windows, std::vector<TemplateMatch> &matches) {

}

void TemplateMatcher::train(std::vector<TemplateGroup> &groups) {
    cv::Mat canny, src_8uc1;

    for (auto &group : groups) {
        for (auto &t : group.templates) {
            cv::convertScaleAbs(t.src, src_8uc1, 255);
            cv::blur(src_8uc1, src_8uc1, cv::Size(3, 3));
            cv::Canny(src_8uc1, canny, 100, 200, 3, false);

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            findContours(canny, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cv::Point(0, 0));

//            std::vector<cv::Point> featurePoints;
//            for (int y = 0; y < canny.rows; y++) {
//                for (int x = 0; x < canny.cols; x++) {
//                    if (canny.at<uchar>(y, x) > 0) {
//                        featurePoints.push_back(cv::Point(x, y));
//                    }
//                }
//            }

            cv::Mat colorChannels(t.src.size(), CV_32FC3);
            cv::cvtColor(t.src, colorChannels, CV_GRAY2BGR);


            std::cout << hierarchy.size() << std::endl;
            // Draw random 100 points
            for (auto &contour : contours) {
                for (int i = 0; i < contour.size(); i++) {
                    cv::circle(colorChannels, contour[i], 1, cv::Scalar(0, 1, 0), -1);
                    cv::imshow("Sobel on template (canny)", canny);
                    cv::imshow("Sobel on template (orig)", colorChannels);
                    cv::waitKey(0);
                }
            }


            cv::imshow("Sobel on template (orig)", colorChannels);
            cv::imshow("Sobel on template (canny)", canny);
            cv::waitKey(0);
        }
    }
}
