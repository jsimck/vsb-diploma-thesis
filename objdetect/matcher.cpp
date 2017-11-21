#include <random>
#include <algorithm>
#include <utility>
#include "matcher.h"
#include "../core/triplet.h"
#include "hasher.h"
#include "../utils/timer.h"
#include "objectness.h"
#include "../utils/visualizer.h"
#include "../processing/processing.h"
#include "../core/template.h"
#include "../core/classifier_criteria.h"
#include "../utils/utils.h"

cv::Vec3b Matcher::normalizeHSV(cv::Vec3b &hsv) {
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

void Matcher::cherryPickFeaturePoints(std::vector<ValuePoint<float>> &points, double tMinDistance, int pointsCount, std::vector<cv::Point> &out) {
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

void Matcher::generateFeaturePoints(std::vector<Template> &templates) {
    const size_t iSize = templates.size();

    #pragma omp parallel for shared(templates) firstprivate(criteria)
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
                    edgePoints.push_back(ValuePoint<float>(cv::Point(x, y) - t.objBB.tl(), sobelValue));
                } else if (stableValue > 0.2f) {
                    stablePoints.push_back(ValuePoint<float>(cv::Point(x, y) - t.objBB.tl(), stableValue));
                }
            }
        }

        // Check if there's enough points to extract
        assert(edgePoints.size() > criteria->train.matcher.pointsCount);
        assert(stablePoints.size() > criteria->train.matcher.pointsCount);

        // Sort point values descending & cherry pick feature points
        std::sort(edgePoints.rbegin(), edgePoints.rend());
        std::shuffle(stablePoints.rbegin(), stablePoints.rend(), std::mt19937(std::random_device()())); // Randomize stable points
        cherryPickFeaturePoints(edgePoints, edgePoints.size() / criteria->train.matcher.pointsCount, criteria->train.matcher.pointsCount, t.edgePoints);
        cherryPickFeaturePoints(stablePoints, stablePoints.size() / criteria->train.matcher.pointsCount, criteria->train.matcher.pointsCount, t.stablePoints);

        assert(edgePoints.size() > criteria->train.matcher.pointsCount);
        assert(stablePoints.size() > criteria->train.matcher.pointsCount);
    }
}

// TODO do something with invalid depth values
void Matcher::extractFeatures(std::vector<Template> &templates) {
    const size_t iSize = templates.size();

    #pragma omp parallel for shared(templates) firstprivate(criteria)
    for (size_t i = 0; i < iSize; i++) {
        // Get template by reference for better access
        Template &t = templates[i];
        assert(!t.srcGray.empty());
        assert(!t.srcHSV.empty());
        assert(!t.srcDepth.empty());
        std::vector<float> depths;

        for (int j = 0; j < criteria->train.matcher.pointsCount; j++) {
            // Create offsets to object bounding box
            cv::Point stablePOff(t.stablePoints[j].x + t.objBB.x, t.stablePoints[j].y + t.objBB.y);
            cv::Point edgePOff(t.edgePoints[j].x + t.objBB.x, t.edgePoints[j].y + t.objBB.y);

            // Save features
            float depth = t.srcDepth.at<ushort>(stablePOff);
            t.features.depths.push_back(depth);
            t.features.gradients.emplace_back(t.quantizedGradients.at<uchar>(edgePOff));
            t.features.normals.emplace_back(t.quantizedNormals.at<uchar>(stablePOff));
            t.features.colors.push_back(normalizeHSV(t.srcHSV.at<cv::Vec3b>(stablePOff)));

            // Save only valid depths (skip 0)
            if (depth != 0) {
                depths.push_back(depth);
            }

            assert(t.features.gradients[j] >= 0);
            assert(t.features.gradients[j] < 5);
            assert(t.features.normals[j] >= 0);
            assert(t.features.normals[j] < 255);
        }

        // Calculate median of depths
        t.features.depthMedian = Utils::median<float>(depths);
        depths.clear();

#ifndef NDEBUG
//        Visualizer::visualizeTemplate(t, "data/", 0, "Template feature points");
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
int Matcher::testObjectSize(float scale, float depth, Window &window, cv::Mat &sceneDepth, cv::Point &stable) {
    const unsigned long fSize = criteria->detect.matcher.depthDeviationFunction.size();

    for (int y = criteria->detect.matcher.neighbourhood.start; y <= criteria->detect.matcher.neighbourhood.end; ++y) {
        for (int x = criteria->detect.matcher.neighbourhood.start; x <= criteria->detect.matcher.neighbourhood.end; ++x) {
            // Apply needed offsets to feature point
            cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

            // Template points in larger templates can go beyond scene boundaries (don't count)
            if (offsetP.x >= sceneDepth.cols || offsetP.y >= sceneDepth.rows ||
                offsetP.x < 0 || offsetP.y < 0) continue;

            // Get depth value at point
            float ratio = 0;
            float sDepth = sceneDepth.at<ushort>(offsetP);

            // TODO better wrong depth handling
            if (sDepth == 0) continue;

            // Get correct deviation ratio
            for (size_t j = 0; j < fSize - 1; j++) {
                if (sDepth < criteria->detect.matcher.depthDeviationFunction[j + 1][0]) {
                    ratio = (1 - criteria->detect.matcher.depthDeviationFunction[j + 1][1]);
                    break;
                }
            }

            if (sDepth >= ((depth * scale) * ratio) && sDepth <= ((depth * scale) / ratio)) {
                return 1;
            }
        }
    }

    return 0;
}

// TODO Use bitwise operations using response maps
int Matcher::testSurfaceNormal(uchar normal, Window &window, cv::Mat &sceneSurfaceNormalsQuantized, cv::Point &stable) {
    for (int y = criteria->detect.matcher.neighbourhood.start; y <= criteria->detect.matcher.neighbourhood.end; ++y) {
        for (int x = criteria->detect.matcher.neighbourhood.start; x <= criteria->detect.matcher.neighbourhood.end; ++x) {
            // Apply needed offsets to feature point
            cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

            // Template points in larger templates can go beyond scene boundaries (don't count)
            if (offsetP.x >= sceneSurfaceNormalsQuantized.cols || offsetP.y >= sceneSurfaceNormalsQuantized.rows ||
                offsetP.x < 0 || offsetP.y < 0) continue;

            if (sceneSurfaceNormalsQuantized.at<uchar>(offsetP) == normal) {
                return 1;
            }
        }
    }

    return 0;
}

// TODO Use bitwise operations using response maps
int Matcher::testGradients(uchar gradient, Window &window, cv::Mat &sceneAnglesQuantized, cv::Mat &sceneMagnitudes, cv::Point &edge) {
    for (int y = criteria->detect.matcher.neighbourhood.start; y <= criteria->detect.matcher.neighbourhood.end; ++y) {
        for (int x = criteria->detect.matcher.neighbourhood.start; x <= criteria->detect.matcher.neighbourhood.end; ++x) {
            // Apply needed offsets to feature point
            cv::Point offsetP(edge.x + window.tl().x + x, edge.y + window.tl().y + y);

            // Template points in larger templates can go beyond scene boundaries (don't count)
            if (offsetP.x >= sceneAnglesQuantized.cols || offsetP.y >= sceneAnglesQuantized.rows ||
                offsetP.x < 0 || offsetP.y < 0) continue;

            // TODO - make member threshold (detect automatically based on training values)
            if (sceneAnglesQuantized.at<uchar>(offsetP) == gradient && sceneMagnitudes.at<float>(offsetP) > criteria->detect.matcher.tMinGradMag) {
                return 1;
            }
        }
    }

    return 0;
}

int Matcher::testDepth(float scale, float diameter, float depthMedian, Window &window, cv::Mat &sceneDepth, cv::Point &stable) {
    for (int y = criteria->detect.matcher.neighbourhood.start; y <= criteria->detect.matcher.neighbourhood.end; ++y) {
        for (int x = criteria->detect.matcher.neighbourhood.start; x <= criteria->detect.matcher.neighbourhood.end; ++x) {
            // Apply needed offsets to feature point
            cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

            // Template points in larger templates can go beyond scene boundaries (don't count)
            if (offsetP.x >= sceneDepth.cols || offsetP.y >= sceneDepth.rows ||
                offsetP.x < 0 || offsetP.y < 0)
                continue;

            if ((sceneDepth.at<ushort>(offsetP) - depthMedian * scale) < (criteria->detect.matcher.depthK * diameter * criteria->info.depthScaleFactor)) {
                return 1;
            }
        }
    }

    return 0;
}

// TODO consider eroding object in training stage to be more tolerant to inaccuracy on the edges
int Matcher::testColor(cv::Vec3b HSV, Window &window, cv::Mat &sceneHSV, cv::Point &stable) {
    for (int y = criteria->detect.matcher.neighbourhood.start; y <= criteria->detect.matcher.neighbourhood.end; ++y) {
        for (int x = criteria->detect.matcher.neighbourhood.start; x <= criteria->detect.matcher.neighbourhood.end; ++x) {
            // Apply needed offsets to feature point
            cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

            // Template points in larger templates can go beyond scene boundaries (don't count)
            if (offsetP.x >= sceneHSV.cols || offsetP.y >= sceneHSV.rows ||
                offsetP.x < 0 || offsetP.y < 0) continue;

            // Normalize scene HSV value
            auto hT = static_cast<int>(HSV[0]);
            auto hS = static_cast<int>(normalizeHSV(sceneHSV.at<cv::Vec3b>(offsetP))[0]);

            if (std::abs(hT - hS) < criteria->detect.matcher.tColorTest) {
                return 1;
            }
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
    float tOverlap = criteria->detect.matcher.tOverlap;

    while (!idx.empty()) {
        // Pick first element with highest score
        Match &firstMatch = matches[idx[0]];

        // Store this index into suppress array and push to final matches, we won't check against this match again
        suppress.push_back(idx[0]);
        pick.push_back(firstMatch);

        // Check overlaps with all other bounding boxes, skipping first one (since it is the one we're checking with)
        #pragma omp parallel for default(none) shared(firstMatch, matches, idx, suppress) firstprivate(tOverlap)
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

//#define VISUALIZE
void Matcher::match(float scale, cv::Mat &sceneHSV, cv::Mat &sceneDepth, cv::Mat &sceneMagnitudes, cv::Mat &sceneAnglesQuantized,
                    cv::Mat &sceneSurfaceNormalsQuantized, std::vector<Window> &windows, std::vector<Match> &matches) {
    // Checks
    assert(!sceneDepth.empty());
    assert(!sceneMagnitudes.empty());
    assert(!sceneAnglesQuantized.empty());
    assert(!sceneSurfaceNormalsQuantized.empty());
    assert(sceneHSV.type() == CV_8UC3);
    assert(sceneDepth.type() == CV_16U);
    assert(sceneAnglesQuantized.type() == CV_8UC1);
    assert(sceneSurfaceNormalsQuantized.type() == CV_8UC1);
    assert(!windows.empty());

    // Min threshold of matched feature points
    const auto N = criteria->train.matcher.pointsCount;
    const auto minThreshold = static_cast<int>(criteria->train.matcher.pointsCount * criteria->detect.matcher.tMatch); // 60%
    const long lSize = windows.size();

    // Stop template matching time
    Timer tMatching;

#ifndef VISUALIZE
    #pragma omp parallel for \
        shared(sceneHSV, sceneDepth, sceneMagnitudes, sceneAnglesQuantized, sceneSurfaceNormalsQuantized, windows, matches) \
        firstprivate(N, minThreshold, scale)
#endif
    for (long l = lSize - 1; l >= 0; l--) {
        for (auto &candidate : windows[l].candidates) {
            // Checks
            assert(candidate != nullptr);

#ifdef VISUALIZE
            bool continuous = false;
            std::vector<int> tITrue, tIITrue, tIIITrue, tIVTrue, tVTrue;
#endif

            // Scores for each test
            float sI = 0, sII = 0, sIII = 0, sIV = 0, sV = 0;

            // Test I
#ifndef VISUALIZE
            #pragma omp parallel for reduction(+:sI)
#endif
            for (int i = 0; i < N; i++) {
#ifdef VISUALIZE
                int tmpResult = testObjectSize(scale, candidate->features.depths[i], windows[l], sceneDepth, candidate->stablePoints[i]);
                sI += tmpResult;
                tITrue.push_back(tmpResult);
#else
                sI += testObjectSize(scale, candidate->features.depths[i], windows[l], sceneDepth, candidate->stablePoints[i]);
#endif
            }

#ifdef VISUALIZE
            if (Visualizer::visualizeTests(*candidate, sceneHSV, sceneDepth, windows[l], candidate->stablePoints, candidate->edgePoints,
                                       criteria->detect.matcher.neighbourhood, tITrue, tIITrue, tIIITrue, tIVTrue, tVTrue, N,
                                       minThreshold, 1, continuous, "data/", 0, nullptr)) {
                break;
            }
#endif

            if (sI < minThreshold) continue;

            // Test II
#ifndef VISUALIZE
            #pragma omp parallel for reduction(+:sII)
#endif
            for (int i = 0; i < N; i++) {
#ifdef VISUALIZE
                int tmpResult = testSurfaceNormal(candidate->features.normals[i], windows[l], sceneSurfaceNormalsQuantized, candidate->stablePoints[i]);
                sII += tmpResult;
                tIITrue.push_back(tmpResult);
#else
                sII += testSurfaceNormal(candidate->features.normals[i], windows[l], sceneSurfaceNormalsQuantized, candidate->stablePoints[i]);
#endif
            }

#ifdef VISUALIZE
            if (Visualizer::visualizeTests(*candidate, sceneHSV, sceneDepth, windows[l], candidate->stablePoints, candidate->edgePoints,
                                       criteria->detect.matcher.neighbourhood, tITrue, tIITrue, tIIITrue, tIVTrue, tVTrue, N,
                                       minThreshold, 2, continuous, "data/", 0, nullptr)) {
                break;
            }
#endif

            if (sII < minThreshold) continue;

            // Test III
#ifndef VISUALIZE
            #pragma omp parallel for reduction(+:sIII)
#endif
            for (int i = 0; i < N; i++) {
#ifdef VISUALIZE
                int tmpResult = testGradients(candidate->features.gradients[i], windows[l], sceneAnglesQuantized, sceneMagnitudes, candidate->edgePoints[i]);
                sIII += tmpResult;
                tIIITrue.push_back(tmpResult);
#else
                sIII += testGradients(candidate->features.gradients[i], windows[l], sceneAnglesQuantized, sceneMagnitudes, candidate->edgePoints[i]);
#endif
            }

#ifdef VISUALIZE
            if (Visualizer::visualizeTests(*candidate, sceneHSV, sceneDepth, windows[l], candidate->stablePoints, candidate->edgePoints,
                                       criteria->detect.matcher.neighbourhood, tITrue, tIITrue, tIIITrue, tIVTrue, tVTrue, N,
                                       minThreshold, 3, continuous, "data/", 0, nullptr)) {
                break;
            }
#endif

            if (sIII < minThreshold) continue;

            // Test IV
#ifndef VISUALIZE
            #pragma omp parallel for reduction(+:sIV)
#endif
            for (int i = 0; i < N; i++) {
#ifdef VISUALIZE
                int tmpResult = testDepth(scale, candidate->diameter, candidate->features.depthMedian, windows[l], sceneDepth, candidate->stablePoints[i]);
                sIV += tmpResult;
                tIVTrue.push_back(tmpResult);
#else
                sIV += testDepth(scale, candidate->diameter, candidate->features.depthMedian, windows[l], sceneDepth, candidate->stablePoints[i]);
#endif
            }

#ifdef VISUALIZE
            if (Visualizer::visualizeTests(*candidate, sceneHSV, sceneDepth, windows[l], candidate->stablePoints, candidate->edgePoints,
                                       criteria->detect.matcher.neighbourhood, tITrue, tIITrue, tIIITrue, tIVTrue, tVTrue, N,
                                       minThreshold, 4, continuous, "data/", 0, nullptr)) {
                break;
            }
#endif

            if (sIV < minThreshold) continue;

            // Test V
#ifndef VISUALIZE
            #pragma omp parallel for reduction(+:sV)
#endif
            for (int i = 0; i < N; i++) {
#ifdef VISUALIZE
                std::cout << "visualize" << std::endl;
                int tmpResult = testColor(candidate->features.colors[i], windows[l], sceneHSV, candidate->stablePoints[i]);
                sV += tmpResult;
                tVTrue.push_back(tmpResult);
#else
                sV += testColor(candidate->features.colors[i], windows[l], sceneHSV, candidate->stablePoints[i]);
#endif
            }

#ifdef VISUALIZE
            if (Visualizer::visualizeTests(*candidate, sceneHSV, sceneDepth, windows[l], candidate->stablePoints, candidate->edgePoints,
                                       criteria->detect.matcher.neighbourhood, tITrue, tIITrue, tIIITrue, tIVTrue, tVTrue, N,
                                       minThreshold, 5, continuous, "data/", 0, nullptr)) {
                break;
            }
#endif

            if (sV < minThreshold) continue;

            // Push template that passed all tests to matches array
            float score = (sII / N) + (sIII / N) + (sIV / N) + (sV / N);
            cv::Rect matchBB = cv::Rect(windows[l].tl().x, windows[l].tl().y, candidate->objBB.width, candidate->objBB.height);

//            #pragma omp critical
//            matches.emplace_back(candidate, matchBB, score, score * (candidate->objBB.area() / scale));
            #pragma omp critical
            matches.emplace_back(candidate, matchBB, score, score);

#ifndef NDEBUG
            std::cout
                << "  |_ id: " << candidate->id
                << ", window: " << l
                << ", score: " << score
                << ", score I: " << sI
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
    this->criteria = criteria;
}
