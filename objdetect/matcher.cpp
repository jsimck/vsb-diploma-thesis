#include <random>
#include <algorithm>
#include "matcher.h"
#include "../core/triplet.h"
#include "hasher.h"
#include "../utils/timer.h"
#include "objectness.h"
#include "../utils/visualizer.h"

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

uchar Matcher::quantizeOrientationGradient(float deg) {
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

    // Check for white (set to yellow)
    if (hsv[2] > tV && hsv[1] < tS) {
        return cv::Vec3b(30, hsv[1], hsv[2]); // Set to yellow
    }

    return hsv;
}

void Matcher::cherryPickFeaturePoints(std::vector<ValuePoint<float>> &points, double tMinDistance, uint pointsCount, std::vector<cv::Point> &out) {
    double minDst = tMinDistance;
    const size_t pointsSize = points.size();

    while (out.size() < pointsCount) {
        out.clear();
        minDst -= 0.5;

        for (size_t k = 0; k < pointsSize; ++k) {
            bool skip = false;
            const size_t edgePointsSize = out.size();

            for (size_t j = 0; j < edgePointsSize; ++j) {
                if (cv::norm(points[k].p - out[j]) < minDst) {
                    skip = true;
                    break;
                }
            }

            if (skip) {
                continue;
            }

            out.push_back(points[k].p);
        }
    }

    // Resize result to actual required size
    out.resize(pointsCount);
}

void Matcher::generateFeaturePoints(std::vector<Group> &groups) {
    for (auto &group : groups) {
        const size_t iSize = group.templates.size();

        for (size_t i = 0; i < iSize; i++) {
            // Get template by reference for better access
            Template &t = group.templates[i];
            std::vector<ValuePoint<float>> edgePoints;
            std::vector<ValuePoint<float>> stablePoints;
            cv::Mat sobel, visualization;

            // Apply sobel to get mask for edge areas
            Objectness::filterSobel(t.srcGray, sobel);

            for (int y = 0; y < sobel.rows; y++) {
                for (int x = 0; x < sobel.cols; x++) {
                    float sobelValue = sobel.at<float>(y, x);
                    float stableValue = t.srcGray.at<float>(y, x);

                    if (sobelValue > 0.2f) {
                        // Save point and offset to object BB
                        ValuePoint<float> sPoint(cv::Point(x + t.objBB.tl().x, y + t.objBB.tl().y), sobelValue);
                        edgePoints.push_back(sPoint);
                    } else if (sobelValue < 0.6f && stableValue > 0.15f) {
                        // Save point and offset to object BB
                        ValuePoint<float> sPoint(cv::Point(x + t.objBB.tl().x, y + t.objBB.tl().y), stableValue);
                        stablePoints.push_back(sPoint);
                    }
                }
            }

            // Check if there's enough points to extract
            assert(edgePoints.size() > pointsCount);
            assert(stablePoints.size() > pointsCount);

            // Sort point values descending & cherry pick feature points
            std::sort(edgePoints.rbegin(), edgePoints.rend());
            cherryPickFeaturePoints(edgePoints, edgePoints.size() / pointsCount, pointsCount, t.edgePoints);
            cherryPickFeaturePoints(stablePoints, stablePoints.size() / pointsCount, pointsCount, t.stablePoints);

            assert(edgePoints.size() > pointsCount);
            assert(stablePoints.size() > pointsCount);

#ifndef NDEBUG
//            Visualizer::visualizeSingleTemplateFeaturePoints(t);
#endif
        }
    }
}

void Matcher::extractFeatures(std::vector<Group> &groups) {
    assert(groups.size() > 0);

    for (auto &group : groups) {
        const size_t iSize = group.templates.size();

        #pragma omp parallel for
        for (size_t i = 0; i < iSize; i++) {
            // Get template by reference for better access
            Template &t = group.templates[i];
            assert(!t.srcGray.empty());
            assert(!t.srcHSV.empty());
            assert(!t.srcDepth.empty());

            for (uint j = 0; j < pointsCount; j++) {
                // Create offsets to object bounding box
                cv::Point stablePOff(t.stablePoints[j].x + t.objBB.x, t.stablePoints[j].y + t.objBB.y);
                cv::Point edgePOff(t.edgePoints[j].x + t.objBB.x, t.edgePoints[j].y + t.objBB.y);

                // Extract features
                float depth = t.srcDepth.at<float>(stablePOff);
                uchar gradient = quantizeOrientationGradient(orientationGradient(t.srcGray, edgePOff));
                uchar normal = Hasher::quantizeSurfaceNormal(Hasher::surfaceNormal(t.srcDepth, stablePOff));
                cv::Vec3b hsv = normalizeHSV(t.srcHSV.at<cv::Vec3b>(stablePOff));

                // Save features
                t.features.depths.push_back(depth);
                t.features.gradients.push_back(gradient);
                t.features.normals.push_back(normal);
                t.features.colors.push_back(hsv);

                assert(t.features.gradients[j] >= 0);
                assert(t.features.gradients[j] < 5);
                assert(t.features.normals[j] >= 0);
                assert(t.features.normals[j] < 8);
            }
        }
    }
}

void Matcher::train(std::vector<Group> &groups) {
    // Generate edge and stable points for features extraction
    generateFeaturePoints(groups);
    std::cout << "  |_ Feature points generated" << std::endl;

    // Extract features for all templates
    extractFeatures(groups);
    std::cout << "  |_ Features extracted" << std::endl;
}

// TODO implement object size test
bool Matcher::testObjectSize(float scale) {
    return true;
}

// TODO Use bitwise operations using response maps
int Matcher::testSurfaceNormal(const uchar normal, Window &window, const cv::Mat &sceneDepth, const cv::Point &stable) {
    for (int y = neighbourhood.start; y <= neighbourhood.end; ++y) {
        for (int x = neighbourhood.start; x <= neighbourhood.end; ++x) {
            // Apply needed offsets to feature point
            cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

            // Template points in larger templates can go beyond scene boundaries (don't count)
            if (offsetP.x >= sceneDepth.cols || offsetP.y >= sceneDepth.rows ||
                offsetP.x < 0 || offsetP.y < 0) continue;

            if (Hasher::quantizeSurfaceNormal(Hasher::surfaceNormal(sceneDepth, offsetP)) == normal) return 1;
        }
    }

    return 0;
}

// TODO Use bitwise operations using response maps
int Matcher::testGradients(const uchar gradient, Window &window, const cv::Mat &sceneGray, const cv::Point &edge) {
    for (int y = neighbourhood.start; y <= neighbourhood.end; ++y) {
        for (int x = neighbourhood.start; x <= neighbourhood.end; ++x) {
            // Apply needed offsets to feature point
            cv::Point offsetP(edge.x + window.tl().x + x, edge.y + window.tl().y + y);

            // Template points in larger templates can go beyond scene boundaries (don't count)
            if (offsetP.x >= sceneGray.cols || offsetP.y >= sceneGray.rows ||
                offsetP.x < 0 || offsetP.y < 0) continue;

            if (quantizeOrientationGradient(orientationGradient(sceneGray, offsetP)) == gradient) return 1;
        }
    }

    return 0;
}

// TODO use proper value of k constant (physical diameter)
int Matcher::testDepth(int physicalDiameter, std::vector<int> &depths) {
    const float k = 0.8f;
    int dm = median(depths), score = 0;

    #pragma omp parallel for
    for (size_t i = 0; i < depths.size(); ++i) {
        #pragma omp atomic
        score += (std::abs(depths[i] - dm) < k * physicalDiameter) ? 1 : 0;
    }

    return score;
}

// TODO consider eroding object in training stage to be more tollerant to inaccuracy on the edges
int Matcher::testColor(const cv::Vec3b HSV, Window &window, const cv::Mat &sceneHSV, const cv::Point &stable) {
    for (int y = neighbourhood.start; y <= neighbourhood.end; ++y) {
        for (int x = neighbourhood.start; x <= neighbourhood.end; ++x) {
            // Apply needed offsets to feature point
            cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

            // Template points in larger templates can go beyond scene boundaries (don't count)
            if (offsetP.x >= sceneHSV.cols || offsetP.y >= sceneHSV.rows ||
                offsetP.x < 0 || offsetP.y < 0) continue;

            // Normalize scene HSV value
            int hT = static_cast<int>(HSV[0]);
            int hS = static_cast<int>(normalizeHSV(sceneHSV.at<cv::Vec3b>(offsetP))[0]);

            if (std::abs(hT - hS) < tColorTest) return 1;
        }
    }

    return 0;
}

void Matcher::nonMaximaSuppression(std::vector<Match> &matches) {
    assert(matches.size() > 0);

    // Sort all matches by their highest score
    std::sort(matches.rbegin(), matches.rend());

    std::vector<Match> pick;
    std::vector<int> suppress(matches.size()); // Indexes of matches to remove
    std::vector<int> idx(matches.size()); // Indexes of bounding boxes to check
    std::iota(idx.begin(), idx.end(), 0);

    while (!idx.empty()) {
        // Pick first element with highest score
        Match &firstMatch = matches[idx[0]];

        // Store this index into suppress array and push to final matches, we won't check against this match again
        suppress.push_back(idx[0]);
        pick.push_back(firstMatch);

        // Check overlaps with all other bounding boxes, skipping first one (since it is the one we're checking with)
        #pragma omp parallel for
        for (size_t i = 1; i < idx.size(); i++) {
            // Get overlap BB coordinates of each other bounding box and compare with the first one
            cv::Rect bb = matches[idx[i]].objBB;
            int x1 = std::min<int>(bb.br().x, firstMatch.objBB.br().x);
            int x2 = std::max<int>(bb.tl().x, firstMatch.objBB.tl().x);
            int y1 = std::min<int>(bb.br().y, firstMatch.objBB.br().y);
            int y2 = std::max<int>(bb.tl().y, firstMatch.objBB.tl().y);

            // Calculate overlap area
            int h = std::max<int>(0, y1 - y2);
            int w = std::max<int>(0, x1 - x2);
            float overlap = static_cast<float>(h * w) / static_cast<float>(firstMatch.objBB.area());

            // If overlap is bigger than min threshold, remove the match
            if (overlap > tOverlap) {
                #pragma omp critical
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

void Matcher::match(const cv::Mat &sceneHSV, const cv::Mat &sceneGray, const cv::Mat &sceneDepth, std::vector<Window> &windows, std::vector<Match> &matches) {
    // Checks
    assert(!sceneHSV.empty());
    assert(!sceneGray.empty());
    assert(!sceneDepth.empty());
    assert(sceneHSV.type() == 16); // CV_8UC3
    assert(sceneGray.type() == 5); // CV_32FC1
    assert(sceneDepth.type() == 5); // CV_32FC1
    assert(windows.size() > 0);

    // Min threshold of matched feature points
    const auto minThreshold = static_cast<int>(pointsCount * tMatch); // 60%
    const size_t lSize = windows.size();

    // Stop template matching time
    Timer tMatching;

    #pragma omp parallel for
    for (size_t l = 0; l < lSize; l++) {
        for (auto &candidate : windows[l].candidates) {
            assert(candidate != nullptr);

            // Scores for each test
            float sII = 0, sIII = 0, sIV = 0, sV = 0;
            std::vector<int> depths;

            // Test I
            if (!testObjectSize(1.0f)) continue;

            // Test II
            #pragma omp parallel for
            for (uint i = 0; i < pointsCount; i++) {
                #pragma omp atomic
                sII += testSurfaceNormal(candidate->features.normals[i], windows[l], sceneDepth, candidate->stablePoints[i]);
            }

            if (sII < minThreshold) continue;

            // Test III
            #pragma omp parallel for
            for (uint i = 0; i < pointsCount; i++) {
                #pragma omp atomic
                sIII += testGradients(candidate->features.gradients[i], windows[l], sceneGray, candidate->edgePoints[i]);
            }

            if (sIII < minThreshold) continue;

            // Test IV
            for (uint i = 0; i < pointsCount; i++) {
                depths.push_back(static_cast<int>(sceneDepth.at<float>(candidate->stablePoints[i]) - candidate->srcDepth.at<float>(candidate->stablePoints[i])));
            }

            sIV = testDepth(candidate->objBB.width, depths);
            if (sIV < minThreshold) continue;

            // Test V
            #pragma omp parallel for
            for (uint i = 0; i < pointsCount; i++) {
                #pragma omp atomic
                sV += testColor(candidate->features.colors[i], windows[l], sceneHSV, candidate->stablePoints[i]);
            }

            if (sV < minThreshold) continue;

            // Push template that passed all tests to matches array
            float score = (sII / pointsCount) + (sIII / pointsCount) + (sV / pointsCount);
            cv::Rect matchBB = cv::Rect(windows[l].tl().x, windows[l].tl().y, candidate->objBB.width, candidate->objBB.height);

            #pragma omp critical
            matches.emplace_back(Match(candidate, matchBB, score));

#ifndef NDEBUG
//            std::cout
//                << "id: " << candidate->id
//                << ", window: " << l
//                << ", score: " << score
//                << ", score II: " << sII
//                << ", score III: " << sIII
//                << ", score IV: " << sIV
//                << ", score V: " << sV
//                << std::endl;
#endif
        }
    }

    std::cout << "  |_ Template matching took: " << tMatching.elapsed() << "s" << std::endl;

    // Stop non maxima time
    Timer tMaxima;

    // Run non maxima suppression on matches
    nonMaximaSuppression(matches);
    std::cout << "  |_ Non maxima suppression took: " << tMaxima.elapsed() << "s" << std::endl;
}

uint Matcher::getPointsCount() const {
    return pointsCount;
}

float Matcher::getTMatch() const {
    return tMatch;
}

const cv::Range &Matcher::getNeighbourhood() const {
    return neighbourhood;
}

uchar Matcher::getTColorTest() const {
    return tColorTest;
}

float Matcher::getTOverlap() const {
    return tOverlap;
}

void Matcher::setPointsCount(uint count) {
    assert(count > 0);
    this->pointsCount = count;
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

void Matcher::setTColorTest(uchar tColorTest) {
    assert(tColorTest < 180); // Hue values are in <0, 179>
    this->tColorTest = tColorTest;
}

void Matcher::setTOverlap(float tOverlap) {
    assert(tOverlap >= 0 && tOverlap <= 1.0f);
    Matcher::tOverlap = tOverlap;
}