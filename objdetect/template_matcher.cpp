#include <random>
#include "template_matcher.h"
#include "../utils/utils.h"
#include "objectness.h"
#include "../core/triplet.h"
#include "hasher.h"

float TemplateMatcher::extractGradientOrientation(cv::Mat &src, cv::Point &point) {
    float dx = (src.at<float>(point.y, point.x - 1) - src.at<float>(point.y, point.x + 1)) / 2.0f;
    float dy = (src.at<float>(point.y - 1, point.x) - src.at<float>(point.y + 1, point.x)) / 2.0f;

    return cv::fastAtan2(dy, dx);
}

void TemplateMatcher::match(const cv::Mat &srcColor, const cv::Mat &srcGrayscale, const cv::Mat &srcDepth,
                     std::vector<Window> &windows, std::vector<TemplateMatch> &matches) {

}

void TemplateMatcher::train(std::vector<TemplateGroup> &groups) {
    // Generate canny and stable feature points
    generateFeaturePoints(groups);

    // Extract gradient orientations
    extractGradientOrientations(groups);
}

void TemplateMatcher::generateFeaturePoints(std::vector<TemplateGroup> &groups) {
    // Init engine
    typedef std::mt19937 engine;

    for (auto &group : groups) {
        for (auto &t : group.templates) {
            std::vector<cv::Point> cannyPoints;
            std::vector<cv::Point> stablePoints;
            cv::Mat canny, sobelX, sobelY, sobel, src_8uc1;

            // Convert to uchar and apply canny to detect edges
            cv::convertScaleAbs(t.src, src_8uc1, 255);

            // Apply canny to detect edges
            cv::blur(src_8uc1, src_8uc1, cv::Size(3, 3));
            cv::Canny(src_8uc1, canny, cannyThreshold1, cannyThreshold2, 3, false);

            // Apply sobel to get mask for stable areas
            cv::Sobel(src_8uc1, sobelX, CV_8U, 1, 0, 3);
            cv::Sobel(src_8uc1, sobelY, CV_8U, 0, 1, 3);
            cv::addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobel);

            // Get all stable and edge points based on threshold
            for (int y = 0; y < canny.rows; y++) {
                for (int x = 0; x < canny.cols; x++) {
                    if (canny.at<uchar>(y, x) > 0) {
                        cannyPoints.push_back(cv::Point(x, y));
                    }

                    if (src_8uc1.at<uchar>(y, x) > grayscaleMinThreshold && sobel.at<uchar>(y, x) <= sobelMaxThreshold) {
                        stablePoints.push_back(cv::Point(x, y));
                    }
                }
            }

            // There should be more than MIN points for each template
            assert(stablePoints.size() > featurePointsCount);
            assert(cannyPoints.size() > featurePointsCount);

            // Shuffle
            std::shuffle(stablePoints.begin(), stablePoints.end(), engine(1));
            std::shuffle(cannyPoints.begin(), cannyPoints.end(), engine(1));

            // Save random points into the template arrays
            for (int i = 0; i < featurePointsCount; i++) {
                int ri = (int) Triplet::random(0, stablePoints.size() - 1);
                t.stablePoints.push_back(stablePoints[ri]);

                // Randomize once more
                ri = (int) Triplet::random(0, cannyPoints.size() - 1);
                t.edgePoints.push_back(cannyPoints[ri]);
            }

#ifndef NDEBUG
            // Visualize extracted features
            cv::Mat visualizationMat;
            cv::cvtColor(t.src, visualizationMat, CV_GRAY2BGR);

            for (int i = 0; i < featurePointsCount; ++i) {
                cv::circle(visualizationMat, t.edgePoints[i], 1, cv::Scalar(0, 0, 255), -1);
                cv::circle(visualizationMat, t.stablePoints[i], 1, cv::Scalar(0, 255, 0), -1);
            }

            cv::imshow("TemplateMatcher::train Sobel", sobel);
            cv::imshow("TemplateMatcher::train Canny", canny);
            cv::imshow("TemplateMatcher::train Feature points", visualizationMat);
            cv::waitKey(1);
#endif
        }
    }
}

void TemplateMatcher::extractGradientOrientations(std::vector<TemplateGroup> &groups) {
    for (auto &group : groups) {
        for (auto &t : group.templates) {
            for (auto &p : t.edgePoints) {
                // Print orientation
                std::cout << extractGradientOrientation(t.src, p) << " deg" << std::endl;
            }

            for (auto &p : t.stablePoints) {
                // Print orientation
                std::cout << Hasher::extractSurfaceNormal(t.srcDepth, p) << " normal" << std::endl;
            }
        }
    }
}

uint TemplateMatcher::getFeaturePointsCount() const {
    return featurePointsCount;
}

void TemplateMatcher::setFeaturePointsCount(uint featurePointsCount) {
    assert(featurePointsCount > 0);
    this->featurePointsCount = featurePointsCount;
}
