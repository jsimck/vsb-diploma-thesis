#include <random>
#include <algorithm>
#include "template_matcher.h"
#include "../core/triplet.h"
#include "hasher.h"
#include "../core/template.h"
#include "../utils/utils.h"

float TemplateMatcher::extractOrientationGradient(const cv::Mat &src, cv::Point &point) {
    float dx = (src.at<float>(point.y, point.x - 1) - src.at<float>(point.y, point.x + 1)) / 2.0f;
    float dy = (src.at<float>(point.y - 1, point.x) - src.at<float>(point.y + 1, point.x)) / 2.0f;

    return cv::fastAtan2(dy, dx);
}

int TemplateMatcher::median(std::vector<int> &values) {
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());

    return values[values.size() / 2];
}

int TemplateMatcher::quantizeOrientationGradient(float deg) {
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

    // TODO - better generation of feature points
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

                    // Check if point is at black spot
                    if (t.srcDepth.at<float>(stablePoints[ri]) <= 0) {
                        falseStablePointGenerated = true;
                    } else {
                        // Remove offset so the feature point locations are relative to template BB
                        t.stablePoints.push_back(cv::Point(stablePoints[ri].x - t.objBB.tl().x, stablePoints[ri].y - t.objBB.tl().y));
                    }

                    stablePoints.erase(stablePoints.begin() + ri - 1); // Remove from array of points
                } while (falseStablePointGenerated);

                // Randomize once more
                ri = (int) Triplet::random(0, cannyPoints.size() - 1);

                // Push to feature points array
                t.edgePoints.push_back(cv::Point(cannyPoints[ri].x - t.objBB.tl().x, cannyPoints[ri].y - t.objBB.tl().y));
                cannyPoints.erase(cannyPoints.begin() + ri - 1); // Remove from array of points
            }

            assert(t.stablePoints.size() == featurePointsCount);
            assert(t.edgePoints.size() == featurePointsCount);

#ifndef NDEBUG
            // Visualize extracted features
            cv::Mat visualizationMat;
            cv::cvtColor(t.src, visualizationMat, CV_GRAY2BGR);

            for (int i = 0; i < featurePointsCount; ++i) {
                cv::Point ePOffset(t.edgePoints[i].x + t.objBB.tl().x, t.edgePoints[i].y + t.objBB.tl().y);
                cv::Point sPOffset(t.stablePoints[i].x + t.objBB.tl().x, t.stablePoints[i].y + t.objBB.tl().y);
                cv::circle(visualizationMat, ePOffset, 1, cv::Scalar(0, 0, 255), -1);
                cv::circle(visualizationMat, sPOffset, 1, cv::Scalar(0, 255, 0), -1);
            }

            cv::imshow("TemplateMatcher::train Sobel", sobel);
            cv::imshow("TemplateMatcher::train Canny", canny);
            cv::imshow("TemplateMatcher::train Feature points", visualizationMat);
            cv::waitKey(0);
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

                // Create offset points to work with uncropped templates
                cv::Point stablePOff(t.stablePoints[i].x + t.objBB.x, t.stablePoints[i].y + t.objBB.y);
                cv::Point edgePOff(t.edgePoints[i].x + t.objBB.x, t.edgePoints[i].y + t.objBB.y);

                // Save features to template
                t.features.orientationGradients[i] = quantizeOrientationGradient(extractOrientationGradient(t.src, edgePOff));
                t.features.surfaceNormals[i] = Hasher::quantizeSurfaceNormals(Hasher::extractSurfaceNormal(t.srcDepth, stablePOff));
                t.features.depth[i] = t.srcDepth.at<float>(stablePOff);
                t.features.color[i] = t.srcHSV.at<cv::Vec3b>(stablePOff);
                depthArray.push_back(static_cast<int>(t.features.depth[i]));

                // Checks
                assert(t.features.orientationGradients[i] >= 0);
                assert(t.features.orientationGradients[i] < 5);
                assert(t.features.surfaceNormals[i] >= 0);
                assert(t.features.surfaceNormals[i] < 8);
            }

            // Save median value
            t.features.depthMedian = static_cast<uint>(median(depthArray));
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


int TemplateMatcher::testSurfaceNormalOrientation(const int tNormal, Window &w, const cv::Mat &srcDepth, const cv::Point &featurePoint, const cv::Range &n) {
    for (int offsetY = n.start; offsetY <= n.end; ++offsetY) {
        for (int offsetX = n.start; offsetX <= n.end; ++offsetX) {
            // Apply needed offsets to stable point
            cv::Point stablePoint(featurePoint.x + w.tl().x + offsetX, featurePoint.y + w.tl().y + offsetY);

            // Checks
            assert(stablePoint.x >= 0);
            assert(stablePoint.y >= 0);
            assert(stablePoint.x < srcDepth.cols);
            assert(stablePoint.y < srcDepth.rows);

            if (Hasher::quantizeSurfaceNormals(Hasher::extractSurfaceNormal(srcDepth, stablePoint)) == tNormal) {
                return 1;
            }
        }
    }

    return 0;
}

int TemplateMatcher::testIntensityGradients(const int tOrientation, Window &w, const cv::Mat &srcGrayscale, const cv::Point &featurePoint, const cv::Range &n) {
    for (int offsetY = n.start; offsetY <= n.end; ++offsetY) {
        for (int offsetX = n.start; offsetX <= n.end; ++offsetX) {
            // Apply needed offsets to edge point
            cv::Point edgePoint(featurePoint.x + w.tl().x + offsetX, featurePoint.y + w.tl().y + offsetY);

            // Checks
            assert(edgePoint.x >= 0);
            assert(edgePoint.y >= 0);
            assert(edgePoint.x < srcGrayscale.cols);
            assert(edgePoint.y < srcGrayscale.rows);

            if (quantizeOrientationGradient(extractOrientationGradient(srcGrayscale, edgePoint)) == tOrientation) {
                return 1;
            }
        }
    }

    return 0;
}

int TemplateMatcher::testDepth(int physicalDiameter, std::vector<int> &depths) {
    const float k = 1.0f;
    int dm = median(depths), score = 0;

    for (int i = 0; i < depths.size(); ++i) {
        std::cout << std::abs(depths[i] - dm) << std::endl;
        score += (std::abs(depths[i] - dm) < k * physicalDiameter) ? 1 : 0;
    }

    return score;
}

int TemplateMatcher::testColor() {
    return 0;
}

void TemplateMatcher::match(const cv::Mat &srcColor, const cv::Mat &srcGrayscale, const cv::Mat &srcDepth,
                            std::vector<Window> &windows, std::vector<TemplateMatch> &matches) {
    // Checks
    assert(!srcColor.empty());
    assert(!srcGrayscale.empty());
    assert(!srcDepth.empty());
    assert(srcColor.type() == 16); // CV_8UC3
    assert(srcGrayscale.type() == 5); // CV_32FC1
    assert(srcDepth.type() == 5); // CV_32FC1
    assert(windows.size() > 0);

    // Thresholds
    const int minThreshold = static_cast<int>(featurePointsCount * matchThreshold); // 60%
    std::vector<int> removeIndex;

    for (int l = 0; l < windows.size(); l++) {
        for (auto &candidate : windows[l].candidates) {
            // Checks
            assert(candidate != nullptr);

            // Do template matching, if any of the test fails in the cascade, matching is template is discarted
            int scoreII = 0, scoreIII = 0, scoreIV = 0, scoreV = 0;
            std::vector<int> ds;
            candidate->votes = 0; // TODO - remove, for testing only

            // Test I
            if (!testObjectSize(1.0f)) continue;

            // Test II
            for (int i = 0; i < candidate->stablePoints.size(); i++) {
                scoreII += testSurfaceNormalOrientation((*candidate).features.surfaceNormals[i], windows[l], srcDepth,
                                                        candidate->stablePoints[i], matchNeighbourhood);
            }

            if (scoreII < minThreshold) {
                candidate->votes = -1; // TODO - remove, for testing only
                continue;
            }

            // Test III
            for (int i = 0; i < candidate->edgePoints.size(); i++) {
                scoreIII += testIntensityGradients((*candidate).features.orientationGradients[i], windows[l],
                                                         srcDepth, candidate->edgePoints[i], matchNeighbourhood);
            }

            if (scoreIII < minThreshold) {
                candidate->votes = -1; // TODO - remove, for testing only
                continue;
            }

#ifndef NDEBUG
            std::cout
                << "id: " << candidate->id
                << ", window: " << l
                << ", score II: " << scoreII
                << ", score III: " << scoreIII
                << std::endl;
#endif
        }

        // TODO - remove, only for testing --->>>
        // Check if there are any more candidates with positive votes
        bool candidatesLeft = false;
        for (auto &candidate : windows[l].candidates) {
            if (candidate->votes >= 0) {
                candidatesLeft = true;
                break;
            };
        }

        if (!candidatesLeft) {
            removeIndex.push_back(l);
        }
        // TODO - remove, only for testing ^^^^
    }

    // TODO - remove, only for testing --->>>
    std::cout << windows.size() << " ----- ";
    // Remove empty windows
    utils::removeIndex<Window>(windows, removeIndex);
    std::cout << windows.size() << std::endl;
    // TODO - remove, only for testing ^^^^
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

float TemplateMatcher::getMatchThreshold() const {
    return matchThreshold;
}

const cv::Range &TemplateMatcher::getMatchNeighbourhood() const {
    return matchNeighbourhood;
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

void TemplateMatcher::setMatchThreshold(float matchThreshold) {
    assert(matchThreshold > 0);
    assert(matchThreshold <= 1.0f);
    this->matchThreshold = matchThreshold;
}

void TemplateMatcher::setMatchNeighbourhood(cv::Range matchNeighbourhood) {
    assert(matchNeighbourhood.start <= matchNeighbourhood.end);
    this->matchNeighbourhood = matchNeighbourhood;
}
