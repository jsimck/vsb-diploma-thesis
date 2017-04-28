#include <random>
#include <algorithm>
#include "matcher.h"
#include "../core/triplet.h"
#include "hasher.h"
#include "../core/template.h"
#include "../utils/utils.h"

float Matcher::orientationGradient(const cv::Mat &src, cv::Point &p) {
    assert(!src.empty());
    assert(src.type() == 5); // CV_32FC1

    float dx = (src.at<float>(p.y, p.x - 1) - src.at<float>(p.y, p.x + 1)) / 2.0f;
    float dy = (src.at<float>(p.y - 1, p.x) - src.at<float>(p.y + 1, p.x)) / 2.0f;

    return cv::fastAtan2(dy, dx);
}

int Matcher::median(std::vector<int> &values) {
    assert(values.size() > 0);

    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

int Matcher::quantizeOrientationGradient(float deg) {
    // Checks
    assert(deg >= 0);
    assert(deg <= 360);

    // We only work in first 2 quadrants (PI)
    int degPI = static_cast<int>(deg) % 180;

    // Quantize
    if (degPI >= 0 && degPI < 36) {
        return 0;
    } else if (degPI >= 36 && degPI < 72) {
        return 1;
    } else if (degPI >= 72 && degPI < 108) {
        return 2;
    } else if (degPI >= 108 && degPI < 144) {
        return 3;
    } else {
        return 4;
    }
}

cv::Vec3b Matcher::normalizeHSV(const cv::Vec3b &hsv) {
    const uchar tV = 22; // 0.12 of hue threshold
    const uchar tS = 31; // 0.12 of saturation threshold

    // Check for black
    if (hsv[2] <= tV) {
        return cv::Vec3b(120, hsv[1], hsv[2]); // Set to blue
    }

    // Check for white
    if (hsv[2] > tV && hsv[1] < tS) {
        return cv::Vec3b(30, hsv[1], hsv[2]); // Set to yellow
    }
}

void Matcher::generateFeaturePoints(std::vector<Group> &groups) {
    // Init engine
    typedef std::mt19937 engine;

    /* TODO - better generation of feature points
     * as in [19] S. Hinterstoisser, V. Lepetit, S. Ilic, S. Holzer, G. Bradski, K. Konolige,
     * and N. Navab, “Model based training, detection and pose estimation
     * of texture-less 3D objects in heavily cluttered scenes,” in ACCV, 2012.
     * - 3.1.2 Reducing Feature Redundancy:
     * - Color Gradient Features:
     * - Surface Normal Features:
     */
    for (auto &group : groups) {
        for (auto &t : group.templates) {
            std::vector<cv::Point> cannyPoints;
            std::vector<cv::Point> stablePoints;
            cv::Mat canny, sobelX, sobelY, sobel, src_8uc1;

            // Convert to uchar and apply canny to detect edges
            cv::convertScaleAbs(t.srcGray, src_8uc1, 255);

            // Apply canny to detect edges
            cv::blur(src_8uc1, src_8uc1, cv::Size(3, 3));
            cv::Canny(src_8uc1, canny, t1Canny, t2Canny, 3, false);

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

                    if (src_8uc1.at<uchar>(y, x) > tGray && sobel.at<uchar>(y, x) <= tSobel) {
                        stablePoints.push_back(cv::Point(x, y));
                    }
                }
            }

            // There should be more than MIN points for each template
            assert(stablePoints.size() > pointsCount);
            assert(cannyPoints.size() > pointsCount);

            // Shuffle
            std::shuffle(stablePoints.begin(), stablePoints.end(), engine(1));
            std::shuffle(cannyPoints.begin(), cannyPoints.end(), engine(1));

            // Save random points into the template arrays
            for (int i = 0; i < pointsCount; i++) {
                int ri;
                // If points extracted are on the part of depths image corrupted by noise (black spots)
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

            assert(t.stablePoints.size() == pointsCount);
            assert(t.edgePoints.size() == pointsCount);

#ifndef NDEBUG
//            // Visualize extracted features
//            cv::Mat visualizationMat;
//            cv::cvtColor(tpl.srcGray, visualizationMat, CV_GRAY2BGR);
//
//            for (int i = 0; i < pointsCount; ++i) {
//                cv::Point ePOffset(tpl.edgePoints[i].x + tpl.objBB.tl().x, tpl.edgePoints[i].y + tpl.objBB.tl().y);
//                cv::Point sPOffset(tpl.stablePoints[i].x + tpl.objBB.tl().x, tpl.stablePoints[i].y + tpl.objBB.tl().y);
//                cv::circle(visualizationMat, ePOffset, 1, cv::Scalar(0, 0, 255), -1);
//                cv::circle(visualizationMat, sPOffset, 1, cv::Scalar(0, 255, 0), -1);
//            }
//
//            cv::imshow("Matcher::train Sobel", sobel);
//            cv::imshow("Matcher::train Canny", canny);
//            cv::imshow("Matcher::train Feature points", visualizationMat);
//            cv::waitKey(0);
#endif
        }
    }
}

void Matcher::extractFeatures(std::vector<Group> &groups) {
    assert(groups.size() > 0);

    for (auto &group : groups) {
        for (auto &t : group.templates) {
            // Init tmp array to store depths values to compute median
            std::vector<int> depthArray;

            // Quantize surface normal and gradient orientations and extract other features
            for (int i = 0; i < pointsCount; i++) {
                assert(!t.srcGray.empty());
                assert(!t.srcHSV.empty());
                assert(!t.srcDepth.empty());

                // Create offset points to work with uncropped templates
                cv::Point stablePOff(t.stablePoints[i].x + t.objBB.x, t.stablePoints[i].y + t.objBB.y);
                cv::Point edgePOff(t.edgePoints[i].x + t.objBB.x, t.edgePoints[i].y + t.objBB.y);

                // Save features to template
                float depth = t.srcDepth.at<float>(stablePOff);
                t.features.gradients.push_back(quantizeOrientationGradient(orientationGradient(t.srcGray, edgePOff)));
                t.features.normals.push_back(
                    Hasher::quantizeSurfaceNormal(Hasher::surfaceNormal(t.srcDepth, stablePOff)));
                t.features.depths.push_back(depth);
                t.features.colors.push_back(normalizeHSV(t.srcHSV.at<cv::Vec3b>(stablePOff)));
                depthArray.push_back(static_cast<int>(t.features.depths[i]));

                assert(t.features.gradients[i] >= 0);
                assert(t.features.gradients[i] < 5);
                assert(t.features.normals[i] >= 0);
                assert(t.features.normals[i] < 8);
            }

            // Save median value
            t.features.median = static_cast<uint>(median(depthArray));
        }
    }
}

void Matcher::train(std::vector<Group> &groups) {
    // Generate edge and stable points for features extraction
    generateFeaturePoints(groups);

    // Extract features for all templates in template group
    extractFeatures(groups);
}

bool Matcher::testObjectSize(float scale) {
    return true; // TODO implement object size test
}

int Matcher::testSurfaceNormal(const uchar tNormal, Window &window, const cv::Mat &sceneDepth,
                               const cv::Point &featurePoint, const cv::Range &n) {
    // TODO Use bitwise operations using response maps
    for (int offsetY = n.start; offsetY <= n.end; ++offsetY) {
        for (int offsetX = n.start; offsetX <= n.end; ++offsetX) {
            // Apply needed offsets to stable point
            cv::Point stablePoint(featurePoint.x + window.tl().x + offsetX, featurePoint.y + window.tl().y + offsetY);

            // Checks
            assert(stablePoint.x >= 0);
            assert(stablePoint.y >= 0);
            assert(stablePoint.x < sceneDepth.cols);
            assert(stablePoint.y < sceneDepth.rows);

            if (Hasher::quantizeSurfaceNormal(Hasher::surfaceNormal(sceneDepth, stablePoint)) == tNormal) {
                return 1;
            }
        }
    }

    return 0;
}

int Matcher::testGradients(const uchar tOrientation, Window &window, const cv::Mat &sceneGray,
                           const cv::Point &featurePoint, const cv::Range &n) {
    // TODO Use bitwise operations using response maps
    for (int offsetY = n.start; offsetY <= n.end; ++offsetY) {
        for (int offsetX = n.start; offsetX <= n.end; ++offsetX) {
            // Apply needed offsets to edge point
            cv::Point edgePoint(featurePoint.x + window.tl().x + offsetX, featurePoint.y + window.tl().y + offsetY);

            // Checks
            assert(edgePoint.x >= 0);
            assert(edgePoint.y >= 0);
            assert(edgePoint.x < sceneGray.cols);
            assert(edgePoint.y < sceneGray.rows);

            if (quantizeOrientationGradient(orientationGradient(sceneGray, edgePoint)) == tOrientation) {
                return 1;
            }
        }
    }

    return 0;
}

int Matcher::testDepth(int physicalDiameter, std::vector<int> &depths) {
    const float k = 1.0f;
    int dm = median(depths), score = 0;

    for (int i = 0; i < depths.size(); ++i) {
        std::cout << std::abs(depths[i] - dm) << std::endl;
        score += (std::abs(depths[i] - dm) < k * physicalDiameter) ? 1 : 0;
    }

    return score;
}

/*
 * TODO - improvement
 * In practice we do not take into account the pixels that are too close to the
 * object projection boundaries, to be tolerant to the inaccuracy of the current
 * pose estimate. This can be done eciently
 * by eroding the object projection
 * beforehand.
 */
int Matcher::testColor(const cv::Vec3b tHSV, Window &window, const cv::Mat &sceneHSV, const cv::Point &edgePoint,
                       const cv::Range &n) {

    for (int offsetY = n.start; offsetY <= n.end; ++offsetY) {
        for (int offsetX = n.start; offsetX <= n.end; ++offsetX) {
            // Apply needed offsets to edge point
            cv::Point stablePoint(edgePoint.x + window.tl().x + offsetX, edgePoint.y + window.tl().y + offsetY);

            // Checks
            assert(stablePoint.x >= 0);
            assert(stablePoint.y >= 0);
            assert(stablePoint.x < sceneHSV.cols);
            assert(stablePoint.y < sceneHSV.rows);

            const uchar threshold = 5;
            if (std::abs(normalizeHSV(tHSV)[0] - normalizeHSV(sceneHSV.at<cv::Vec3b>(stablePoint))[0]) < threshold) {
                return 1;
            }
        }
    }

    return 0;
}

void Matcher::nonMaximaSuppression(std::vector<Match> &matches) {
    // Checks
    assert(matches.size() > 0);

    // Sort all matches by their highest score
    std::sort(matches.rbegin(), matches.rend());
    const float overlapThresh = 0.1f;

    std::vector<Match> pick;
    std::vector<int> suppress; // Indexes of matches to remove
    std::vector<int> idx(matches.size()); // Indexes of bounding boxes to check
    std::iota(idx.begin(), idx.end(), 0);

    while (!idx.empty()) {
        // Pick first element with highest score
        Match &firstMatch = matches[idx[0]];

        // Store this index into suppress array and push to final matches, we won'tpl check against this BB again
        suppress.push_back(idx[0]);
        pick.push_back(firstMatch);

        // Check overlaps with all other bounding boxes, skipping first one (since it is the one we're checking with)
        for (int i = 1; i < idx.size(); i++) {
            // Get overlap BB coordinates of each other bounding box and compare with the first one
            cv::Rect bb = matches[idx[i]].objBB;
            int x1 = std::max<int>(bb.tl().x, firstMatch.objBB.tl().x);
            int x2 = std::min<int>(bb.br().x, firstMatch.objBB.br().x);
            int y1 = std::max<int>(bb.tl().y, firstMatch.objBB.tl().y);
            int y2 = std::min<int>(bb.br().y, firstMatch.objBB.br().y);

            // Calculate overlap area
            int h = std::max<int>(0, y2 - y1);
            int w = std::max<int>(0, x2 - x1);
            float overlap = static_cast<float>(h * w) / static_cast<float>(firstMatch.objBB.area());

            // Push index of this window to suppression array, since it is overlapping over minimum threshold
            // with a window of higher score, we can safely ignore this window
            if (overlap > overlapThresh) {
                suppress.push_back(idx[i]);
            }
        }

        // Remove all suppress indexes from idx array
        idx.erase(std::remove_if(idx.begin(), idx.end(),
             [&suppress, &idx](int v) -> bool {
                 return std::find(suppress.begin(), suppress.end(), v) != suppress.end();
             }
        ), idx.end());

        suppress.clear();
    }

    matches.swap(pick);
}

void Matcher::match(const cv::Mat &sceneHSV, const cv::Mat &sceneGray, const cv::Mat &sceneDepth,
                    std::vector<Window> &windows, std::vector<Match> &matches) {
    // Checks
    assert(!sceneHSV.empty());
    assert(!sceneGray.empty());
    assert(!sceneDepth.empty());
    assert(sceneHSV.type() == 16); // CV_8UC3
    assert(sceneGray.type() == 5); // CV_32FC1
    assert(sceneDepth.type() == 5); // CV_32FC1
    assert(windows.size() > 0);

    // Thresholds
    const int minThreshold = static_cast<int>(pointsCount * tMatch); // 60%
    std::vector<int> removeIndex;

    for (int l = 0; l < windows.size(); l++) {
        for (auto &candidate : windows[l].candidates) {
            // Checks
            assert(candidate != nullptr);

            // Do template matching, if any of the test fails in the cascade, matching is template is discarted
            float scoreII = 0, scoreIII = 0, scoreIV = 0, scoreV = 0;
            std::vector<int> ds;
            candidate->votes = 0; // TODO - remove, for testing only

            // Test I
            if (!testObjectSize(1.0f)) continue;

            // Test II
            for (int i = 0; i < pointsCount; i++) {
                scoreII += testSurfaceNormal(candidate->features.normals[i], windows[l], sceneDepth,
                                             candidate->stablePoints[i], neighbourhood);
            }

            if (scoreII < minThreshold) continue;

            // Test III
            for (int i = 0; i < pointsCount; i++) {
                scoreIII += testGradients(candidate->features.gradients[i], windows[l],
                                          sceneDepth, candidate->edgePoints[i], neighbourhood);
            }

            if (scoreIII < minThreshold) continue;

            // Test V
            for (int i = 0; i < pointsCount; i++) {
                scoreV += testColor(candidate->features.colors[i], windows[l],
                                sceneHSV, candidate->stablePoints[i], neighbourhood);
            }

            if (scoreV < minThreshold) continue;

//#ifndef NDEBUG
//            std::cout
//                << "id: " << candidate->id
//                << ", window: " << l
//                << ", score II: " << scoreII
//                << ", score III: " << scoreIII
//                << ", score V: " << scoreV
//                << std::endl;
//#endif
            // Push template that passed all tests to matches array
            float score = (scoreII / pointsCount) + (scoreIII / pointsCount) + (scoreV / pointsCount);
            cv::Rect matchBB = cv::Rect(windows[l].tl().x, windows[l].tl().y, candidate->objBB.width, candidate->objBB.height);
            matches.push_back(Match(candidate, matchBB, score));
        }
    }

    // Run non maxima suppression on matches
    nonMaximaSuppression(matches);
}

uint Matcher::getPointsCount() const {
    return pointsCount;
}

uchar Matcher::getT1Canny() const {
    return t1Canny;
}

uchar Matcher::getT2Canny() const {
    return t2Canny;
}

uchar Matcher::getTSobel() const {
    return tSobel;
}

uchar Matcher::getTGray() const {
    return tGray;
}

float Matcher::getTMatch() const {
    return tMatch;
}

const cv::Range &Matcher::getNeighbourhood() const {
    return neighbourhood;
}

void Matcher::setPointsCount(uint count) {
    assert(count > 0);
    this->pointsCount = count;
}

void Matcher::setT1Canny(uchar t1) {
    assert(pointsCount > 0);
    assert(pointsCount < 256);
    this->t1Canny = t1;
}

void Matcher::setT2Canny(uchar t2) {
    assert(pointsCount > 0);
    assert(pointsCount < 256);
    this->t2Canny = t2;
}

void Matcher::setTSobel(uchar t) {
    assert(pointsCount > 0);
    assert(pointsCount < 256);
    this->tSobel = t;
}

void Matcher::setTGray(uchar t) {
    assert(pointsCount > 0);
    assert(pointsCount < 256);
    this->tGray = t;
}

void Matcher::setTMatch(float t) {
    assert(t > 0);
    assert(t <= 1.0f);
    this->tMatch = t;
}

void Matcher::setNeighbourhood(cv::Range matchNeighbourhood) {
    assert(matchNeighbourhood.start <= matchNeighbourhood.end);
    this->neighbourhood = matchNeighbourhood;
}