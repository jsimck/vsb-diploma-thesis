#include <random>
#include <algorithm>
#include "template_matcher.h"
#include "../core/triplet.h"
#include "hasher.h"
#include "../core/template.h"

float TemplateMatcher::extractGradientOrientation(cv::Mat &src, cv::Point &point) {
    assert(!src.empty());

    float dx = (src.at<float>(point.y, point.x - 1) - src.at<float>(point.y, point.x + 1)) / 2.0f;
    float dy = (src.at<float>(point.y - 1, point.x) - src.at<float>(point.y + 1, point.x)) / 2.0f;

    return cv::fastAtan2(dy, dx);
}

int TemplateMatcher::quantizeOrientationGradients(float deg) {
    // Checks
    assert(deg >= 0);
    assert(deg <= 360);

    // We work only in first 2 quadrants
    int degNormalized = static_cast<int>(deg) % 180;

    // Quantize
    if (degNormalized >= 0 && degNormalized < 36) {
        return 0;
    } else if (degNormalized >= 36 && degNormalized < 72) {
        return 1;
    } else if (degNormalized >= 72 && degNormalized < 108) {
        return 2;
    } else if (degNormalized >= 108 && degNormalized < 144) {
        return 3;
    } else {
        return 4;
    }
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
                int ri;
                // If points extracted are on the part of depth image corrupted by noise (black spots)
                // regenerate new points, until
                bool falseStablePointGenerated;
                do {
                    falseStablePointGenerated = false;
                    ri = (int) Triplet::random(0, stablePoints.size() - 1);
                    cv::Point stablePoint = stablePoints[ri];

                    // Check if point is at black spot
                    if (t.srcDepth.at<float>(stablePoint) <= 0) {
                        falseStablePointGenerated = true;
                    } else {
                        t.stablePoints.push_back(stablePoint);
                    }

                    stablePoints.erase(stablePoints.begin() + ri - 1); // Remove from array of points
                } while (falseStablePointGenerated);

                // Randomize once more
                ri = (int) Triplet::random(0, cannyPoints.size() - 1);
                cv::Point edgePoint = cannyPoints[ri];
                t.edgePoints.push_back(edgePoint);
                cannyPoints.erase(cannyPoints.begin() + ri - 1); // Remove from array of points
            }

            assert(t.stablePoints.size() == featurePointsCount);
            assert(t.edgePoints.size() == featurePointsCount);

#ifndef NDEBUG
            // Visualize extracted features
//            cv::Mat visualizationMat;
//            cv::cvtColor(t.src, visualizationMat, CV_GRAY2BGR);
//
//            for (int i = 0; i < featurePointsCount; ++i) {
//                cv::circle(visualizationMat, t.edgePoints[i], 1, cv::Scalar(0, 0, 255), -1);
//                cv::circle(visualizationMat, t.stablePoints[i], 1, cv::Scalar(0, 255, 0), -1);
//            }
//
//            cv::imshow("TemplateMatcher::train Sobel", sobel);
//            cv::imshow("TemplateMatcher::train Canny", canny);
//            cv::imshow("TemplateMatcher::train Feature points", visualizationMat);
//            cv::waitKey(0);
#endif
        }
    }
}

void TemplateMatcher::extractTemplateFeatures(std::vector<TemplateGroup> &groups) {
    // Checks
    assert(groups.size() > 0);

    for (auto &group : groups) {
        for (auto &t : group.templates) {
            // Init tmp array to store depth values to compute median
            std::vector<int> depthArray;

            // Quantize surface normal and gradient orientations and extract other features
            for (int i = 0; i < featurePointsCount; i++) {
                // Checks
                assert(!t.src.empty());
                assert(!t.srcHSV.empty());
                assert(!t.srcDepth.empty());

                // TODO - consider refactoring the code to work with sources in original 400x400 size (so without bounding box mask applied)
                // Check points are either on template edge in which case surface normal and gradient orientation
                // extraction would fail due to central derivation -> reset roi applied on template and restore it after
                // feature has been extracted
                bool edgePoint = false;

                if ((t.edgePoints[i].x == 0 || t.edgePoints[i].y == 0 || t.edgePoints[i].x == t.objBB.width - 1 || t.edgePoints[i].y == t.objBB.height - 1) ||
                 (t.stablePoints[i].x == 0 || t.stablePoints[i].y == 0 || t.stablePoints[i].x == t.objBB.width - 1 || t.stablePoints[i].y == t.objBB.height - 1)) {
                    t.resetROI();
                    t.edgePoints[i].x += t.objBB.x;
                    t.edgePoints[i].y += t.objBB.y;
                    t.stablePoints[i].x += t.objBB.x;
                    t.stablePoints[i].y += t.objBB.y;
                    edgePoint = true;
                }

                // Save features to template
                t.features.orientationGradients[i] = quantizeOrientationGradients(extractGradientOrientation(t.src, t.edgePoints[i]));
                t.features.surfaceNormals[i] = Hasher::quantizeSurfaceNormals(Hasher::extractSurfaceNormal(t.srcDepth, t.stablePoints[i]));
                t.features.depth[i] = t.srcDepth.at<float>(t.stablePoints[i]);
                t.features.color[i] = t.srcHSV.at<cv::Vec3b>(t.stablePoints[i]);
                depthArray.push_back(static_cast<int>(t.features.depth[i]));

                // Checks
                assert(t.features.orientationGradients[i] >= 0);
                assert(t.features.orientationGradients[i] < 5);
                assert(t.features.surfaceNormals[i] >= 0);
                assert(t.features.surfaceNormals[i] < 8);

                // Restore matrix roi and delete offsets on feature points
                if (edgePoint) {
                    t.applyROI();
                    t.edgePoints[i].x -= t.objBB.x;
                    t.edgePoints[i].y -= t.objBB.y;
                    t.stablePoints[i].x -= t.objBB.x;
                    t.stablePoints[i].y -= t.objBB.y;
                }
            }

            // Save median value
            t.features.depthMedian = static_cast<uint>(extractMedian(depthArray));
            for (auto &item : depthArray) {
                std::cout << item << ", ";
            }
            std::cout << std::endl;
            std::cout << t.features.depthMedian << std::endl << std::endl;
        }
    }
}

void TemplateMatcher::train(std::vector<TemplateGroup> &groups) {
    // Generate edge and stable points for features extraction
    generateFeaturePoints(groups);

    // Extract features for all templates in template group
    extractTemplateFeatures(groups);
}

bool TemplateMatcher::testObjectSize(float scale) {
    return true; // TODO implement object size test
}

float TemplateMatcher::testSurfaceNormalOrientation() {
    return 0;
}

float TemplateMatcher::testIntensityGradients() {
    return 0;
}

float TemplateMatcher::testDepth() {
    return 0;
}

float TemplateMatcher::testColor() {
    return 0;
}

void TemplateMatcher::match(const cv::Mat &srcColor, const cv::Mat &srcGrayscale, const cv::Mat &srcDepth,
                            std::vector<Window> &windows, std::vector<TemplateMatch> &matches) {
    const float minThreshold = 0.6f; // 60%

    for (auto &window : windows) {
        // Skip empty windows
        if (!window.hasCandidates()) continue;

        for (auto &candidate : window.candidates) {
            // TODO implement 5x5 local neighbourhood
            // Do template matching

            // Test I. object size
            if (!testObjectSize(1.0f)) continue;

            // Test II.
        }
    }
}

uint TemplateMatcher::getFeaturePointsCount() const {
    return featurePointsCount;
}

uchar TemplateMatcher::getCannyThreshold1() const {
    return cannyThreshold1;
}

uchar TemplateMatcher::getCannyThreshold2() const {
    return cannyThreshold2;
}

uchar TemplateMatcher::getSobelMaxThreshold() const {
    return sobelMaxThreshold;
}

uchar TemplateMatcher::getGrayscaleMinThreshold() const {
    return grayscaleMinThreshold;
}

void TemplateMatcher::setFeaturePointsCount(uint featurePointsCount) {
    assert(featurePointsCount > 0);
    this->featurePointsCount = featurePointsCount;
}

void TemplateMatcher::setCannyThreshold1(uchar cannyThreshold1) {
    assert(featurePointsCount > 0);
    assert(featurePointsCount < 256);
    this->cannyThreshold1 = cannyThreshold1;
}

void TemplateMatcher::setCannyThreshold2(uchar cannyThreshold2) {
    assert(featurePointsCount > 0);
    assert(featurePointsCount < 256);
    this->cannyThreshold2 = cannyThreshold2;
}

void TemplateMatcher::setSobelMaxThreshold(uchar sobelMaxThreshold) {
    assert(featurePointsCount > 0);
    assert(featurePointsCount < 256);
    this->sobelMaxThreshold = sobelMaxThreshold;
}

void TemplateMatcher::setGrayscaleMinThreshold(uchar grayscaleMinThreshold) {
    assert(featurePointsCount > 0);
    assert(featurePointsCount < 256);
    this->grayscaleMinThreshold = grayscaleMinThreshold;
}

int TemplateMatcher::extractMedian(std::vector<int> &depths) {
    std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
    return depths[depths.size() / 2];
}
