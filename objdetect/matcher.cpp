#include <random>
#include <algorithm>
#include "matcher.h"
#include "../core/triplet.h"
#include "hasher.h"
#include "../utils/timer.h"
#include "objectness.h"
#include "../utils/visualizer.h"
#include "../processing/processing.h"
#include "../core/template.h"
#include "../core/classifier_criteria.h"

int Matcher::median(std::vector<int> &values) {
    assert(!values.empty());

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

void Matcher::cherryPickFeaturePoints(std::vector<ValuePoint<float>> &points, double tMinDistance, int pointsCount,
                                      std::vector<cv::Point> &out) {
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

            out.emplace_back(points[k].p);
        }
    }

    // Resize result to actual required size
    out.resize(pointsCount);
}

void Matcher::generateFeaturePoints(std::vector<Template> &templates) {
    const size_t iSize = templates.size();

    #pragma omp parallel for
    for (size_t i = 0; i < iSize; i++) {
        // Get template by reference for better access
        Template &t = templates[i];
        std::vector<ValuePoint<float>> edgePoints;
        std::vector<ValuePoint<float>> stablePoints;
        cv::Mat sobel, visualization;

        // Apply sobel to get mask for edge areas
        Processing::filterSobel(t.srcGray, sobel, true, true);

        for (int y = 0; y < sobel.rows; y++) {
            for (int x = 0; x < sobel.cols; x++) {
                float sobelValue = sobel.at<float>(y, x);
                float stableValue = t.srcGray.at<float>(y, x);

                if (sobelValue > 0.3f) {
                    edgePoints.emplace_back(ValuePoint<float>(cv::Point(x, y) - t.objBB.tl(), sobelValue));
                } else if (stableValue > 0.2f) {
                    stablePoints.emplace_back(ValuePoint<float>(cv::Point(x, y) - t.objBB.tl(), stableValue));
                }
            }
        }

        // Check if there's enough points to extract
        assert(edgePoints.size() > criteria->trainParams.matcher.pointsCount);
        assert(stablePoints.size() > criteria->trainParams.matcher.pointsCount);

        // Sort point values descending & cherry pick feature points
        std::sort(edgePoints.rbegin(), edgePoints.rend());
        std::shuffle(stablePoints.rbegin(), stablePoints.rend(), std::mt19937(std::random_device()())); // Randomize stable points
        cherryPickFeaturePoints(edgePoints, edgePoints.size() / criteria->trainParams.matcher.pointsCount, criteria->trainParams.matcher.pointsCount, t.edgePoints);
        cherryPickFeaturePoints(stablePoints, stablePoints.size() / criteria->trainParams.matcher.pointsCount, criteria->trainParams.matcher.pointsCount, t.stablePoints);

        assert(edgePoints.size() > criteria->trainParams.matcher.pointsCount);
        assert(stablePoints.size() > criteria->trainParams.matcher.pointsCount);
    }
}

void Matcher::extractFeatures(std::vector<Template> &templates) {
    const size_t iSize = templates.size();

    #pragma omp parallel for
    for (size_t i = 0; i < iSize; i++) {
        // Get template by reference for better access
        Template &t = templates[i];
        assert(!t.srcGray.empty());
        assert(!t.srcHSV.empty());
        assert(!t.srcDepth.empty());

        for (int j = 0; j < criteria->trainParams.matcher.pointsCount; j++) {
            // Create offsets to object bounding box
            cv::Point stablePOff(t.stablePoints[j].x + t.objBB.x, t.stablePoints[j].y + t.objBB.y);
            cv::Point edgePOff(t.edgePoints[j].x + t.objBB.x, t.edgePoints[j].y + t.objBB.y);

            // Save features
            t.features.depths.emplace_back(t.srcDepth.at<float>(stablePOff));
            t.features.gradients.emplace_back(quantizeOrientationGradient(t.angles.at<float>(edgePOff)));
            t.features.normals.emplace_back(Hasher::quantizeSurfaceNormal(Hasher::surfaceNormal(t.srcDepth, stablePOff)));
            t.features.colors.emplace_back(normalizeHSV(t.srcHSV.at<cv::Vec3b>(stablePOff)));

            assert(t.features.gradients[j] >= 0);
            assert(t.features.gradients[j] < 5);
            assert(t.features.normals[j] >= 0);
            assert(t.features.normals[j] < 8);
        }

#ifndef NDEBUG
//            Visualizer::visualizeTemplate(t, "data/", 0, "Template feature points");
#endif
    }
}

void Matcher::train(std::vector<Template> &templates) {
    assert(!templates.empty());

    // Generate edge and stable points for features extraction
    generateFeaturePoints(templates);

    // Extract features for all templates
    extractFeatures(templates);
}

// TODO implement object size test
bool Matcher::testObjectSize(float scale) {
    return true;
}

// TODO Use bitwise operations using response maps
int Matcher::testSurfaceNormal(const uchar normal, Window &window, const cv::Mat &sceneDepth, const cv::Point &stable) {
    for (int y = criteria->detectParams.matcher.neighbourhood.start; y <= criteria->detectParams.matcher.neighbourhood.end; ++y) {
        for (int x = criteria->detectParams.matcher.neighbourhood.start; x <= criteria->detectParams.matcher.neighbourhood.end; ++x) {
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
int Matcher::testGradients(const uchar gradient, Window &window, const cv::Mat &sceneAngles, const cv::Mat &sceneMagnitude, const cv::Point &edge) {
    for (int y = criteria->detectParams.matcher.neighbourhood.start; y <= criteria->detectParams.matcher.neighbourhood.end; ++y) {
        for (int x = criteria->detectParams.matcher.neighbourhood.start; x <= criteria->detectParams.matcher.neighbourhood.end; ++x) {
            // Apply needed offsets to feature point
            cv::Point offsetP(edge.x + window.tl().x + x, edge.y + window.tl().y + y);

            // Template points in larger templates can go beyond scene boundaries (don't count)
            if (offsetP.x >= sceneAngles.cols || offsetP.y >= sceneAngles.rows ||
                offsetP.x < 0 || offsetP.y < 0) continue;

            // TODO - make member threshold (detect automatically based on training values)
            if (quantizeOrientationGradient(sceneAngles.at<float>(offsetP)) == gradient && sceneMagnitude.at<float>(offsetP) > 0.1f) {
                return 1;
            }
        }
    }

    return 0;
}

// TODO use proper value of k constant (physical diameter)
int Matcher::testDepth(int physicalDiameter, std::vector<int> &depths) {
    const float k = 0.8f;
    int dm = median(depths), score = 0;

    #pragma omp parallel for reduction(+:score)
    for (size_t i = 0; i < depths.size(); ++i) {
        score += (std::abs(depths[i] - dm) < k * physicalDiameter) ? 1 : 0;
    }

    return score;
}

// TODO consider eroding object in training stage to be more tolerant to inaccuracy on the edges
int Matcher::testColor(const cv::Vec3b HSV, Window &window, const cv::Mat &sceneHSV, const cv::Point &stable) {
    for (int y = criteria->detectParams.matcher.neighbourhood.start; y <= criteria->detectParams.matcher.neighbourhood.end; ++y) {
        for (int x = criteria->detectParams.matcher.neighbourhood.start; x <= criteria->detectParams.matcher.neighbourhood.end; ++x) {
            // Apply needed offsets to feature point
            cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

            // Template points in larger templates can go beyond scene boundaries (don't count)
            if (offsetP.x >= sceneHSV.cols || offsetP.y >= sceneHSV.rows ||
                offsetP.x < 0 || offsetP.y < 0) continue;

            // Normalize scene HSV value
            auto hT = static_cast<int>(HSV[0]);
            auto hS = static_cast<int>(normalizeHSV(sceneHSV.at<cv::Vec3b>(offsetP))[0]);

            if (std::abs(hT - hS) < criteria->detectParams.matcher.tColorTest) return 1;
        }
    }

    return 0;
}

void Matcher::nonMaximaSuppression(std::vector<Match> &matches) {
    if (matches.empty()) {
        return;
    }

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
        suppress.emplace_back(idx[0]);
        pick.emplace_back(firstMatch);

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
            if (overlap > criteria->detectParams.matcher.tOverlap) {
                #pragma omp critical
                suppress.emplace_back(idx[i]);
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

// Enable/disable visualization for function below
#define VISUALIZE_MATCH
void Matcher::match(float scale, const cv::Mat &sceneHSV, const cv::Mat &sceneGray, const cv::Mat &sceneDepth, std::vector<Window> &windows, std::vector<Match> &matches) {
    // Checks
    assert(!sceneHSV.empty());
    assert(!sceneGray.empty());
    assert(!sceneDepth.empty());
    assert(sceneHSV.type() == CV_8UC3);
    assert(sceneGray.type() == CV_32FC1);
    assert(sceneDepth.type() == CV_32FC1);
    assert(!windows.empty());

    // Min threshold of matched feature points
    const auto N = criteria->trainParams.matcher.pointsCount;
    const auto minThreshold = static_cast<int>(criteria->trainParams.matcher.pointsCount * criteria->detectParams.matcher.tMatch); // 60%
    const long lSize = windows.size();

    // Stop template matching time
    Timer tMatching;

    // Calculate angels and magnitudes
    cv::Mat sceneAngle, sceneMagnitude;
    Processing::orientationGradients(sceneGray, sceneAngle, sceneMagnitude);

#if not defined NDEBUG and not defined VISUALIZE_MATCH
    #pragma omp parallel for
#endif
    for (long l = lSize - 1; l >= 0; l--) { // TODO - should run from start, not end
        for (auto &candidate : windows[l].candidates) {
            assert(candidate != nullptr);

#if not defined NDEBUG and defined VISUALIZE_MATCH
            bool continuous = false;
            std::vector<int> tIITrue, tIIITrue, tVTrue;
#endif

            // Scores for each test
            float sII = 0, sIII = 0, sIV = 0, sV = 0;
            std::vector<int> depths;

            // Test I
            if (!testObjectSize(scale)) continue;

            // Test II
#if not defined NDEBUG and not defined VISUALIZE_MATCH
            #pragma omp parallel for reduction(+:sII)
#endif
            for (int i = 0; i < N; i++) {
#if not defined NDEBUG and defined VISUALIZE_MATCH
                int tmpResult = testSurfaceNormal(candidate->features.normals[i], windows[l], sceneDepth, candidate->stablePoints[i]);
                sII += tmpResult;
                tIITrue.emplace_back(tmpResult);
#else
                sII += testSurfaceNormal(candidate->features.normals[i], windows[l], sceneDepth, candidate->stablePoints[i]);
#endif
            }

#if not defined NDEBUG and defined VISUALIZE_MATCH
            Visualizer::visualizeTests(*candidate, sceneHSV, windows[l], candidate->stablePoints, candidate->edgePoints,
                                       criteria->detectParams.matcher.neighbourhood, tIITrue, tIIITrue, sIV, tVTrue, N, minThreshold, continuous);
#endif

            if (sII < minThreshold) continue;

            // Test III
#if not defined NDEBUG and not defined VISUALIZE_MATCH
            #pragma omp parallel for reduction(+:sIII)
#endif
            for (int i = 0; i < N; i++) {
#if not defined NDEBUG and defined VISUALIZE_MATCH
                int tmpResult = testGradients(candidate->features.gradients[i], windows[l], sceneAngle, sceneMagnitude, candidate->edgePoints[i]);
                sIII += tmpResult;
                tIIITrue.emplace_back(tmpResult);
#else
                sIII += testGradients(candidate->features.gradients[i], windows[l], sceneAngle, sceneMagnitude, candidate->edgePoints[i]);
#endif
            }

#if not defined NDEBUG and defined VISUALIZE_MATCH
            Visualizer::visualizeTests(*candidate, sceneHSV, windows[l], candidate->stablePoints, candidate->edgePoints,
                                       criteria->detectParams.matcher.neighbourhood, tIITrue, tIIITrue, sIV, tVTrue, N, minThreshold, continuous);
#endif

            if (sIII < minThreshold) continue;

            // Test IV
            for (int i = 0; i < N; i++) {
                depths.emplace_back(static_cast<int>(sceneDepth.at<float>(candidate->stablePoints[i]) - candidate->features.depths[i]));
            }

            sIV = testDepth(candidate->objBB.width, depths);
            if (sIV < minThreshold) continue;

            // Test V
#if not defined NDEBUG and not defined VISUALIZE_MATCH
            #pragma omp parallel for reduction(+:sV)
#endif
            for (int i = 0; i < N; i++) {
#if not defined NDEBUG and defined VISUALIZE_MATCH
                int tmpResult = testColor(candidate->features.colors[i], windows[l], sceneHSV, candidate->stablePoints[i]);
                sV += tmpResult;
                tVTrue.emplace_back(tmpResult);
#else
                sV += testColor(candidate->features.colors[i], windows[l], sceneHSV, candidate->stablePoints[i]);
#endif
            }

#if not defined NDEBUG and defined VISUALIZE_MATCH
            Visualizer::visualizeTests(*candidate, sceneHSV, windows[l], candidate->stablePoints, candidate->edgePoints,
                                       criteria->detectParams.matcher.neighbourhood, tIITrue, tIIITrue, sIV, tVTrue, N, minThreshold, continuous);
#endif

            if (sV < minThreshold) continue;

            // Push template that passed all tests to matches array
            float score = (sII / N) + (sIII / N) + (sIV / N) + (sV / N);
            cv::Rect matchBB = cv::Rect(windows[l].tl().x, windows[l].tl().y, candidate->objBB.width, candidate->objBB.height);

            #pragma omp critical
            matches.emplace_back(Match(candidate, matchBB, score));

#ifndef NDEBUG
            std::cout
                << "  |_ id: " << candidate->id
                << ", window: " << l
                << ", score: " << score
                << ", score II: " << sII
                << ", score III: " << sIII
                << ", score IV: " << sIV
                << ", score V: " << sV
                << ", matches: " << matches.size()
                << std::endl;
#endif
        }
    }

    std::cout << "  |_ Template matching took: " << tMatching.elapsed() << "s" << std::endl;

    // Run non maxima suppression on matches
    Timer tMaxima;
    nonMaximaSuppression(matches);
    std::cout << "  |_ Non maxima suppression took: " << tMaxima.elapsed() << "s" << std::endl;
}

void Matcher::setCriteria(std::shared_ptr<ClassifierCriteria> criteria) {
    this->criteria = std::move(criteria);
}
