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
#include "../processing/computation.h"

namespace tless {
    void Matcher::selectScatteredFeaturePoints(std::vector<std::pair<cv::Point, uchar>> &points, uint count, std::vector<cv::Point> &scattered) {
        // Define initial distance
        float minDst = points.size() / criteria->featurePointsCount;

        // Continue decreasing min distance in each loop, till we find >= number of desired points
        while (scattered.size() < count) {
            scattered.clear();
            minDst -= 0.5f;

            // Calculate euqlidian distance between each point
            for (size_t k = 0; k < points.size(); ++k) {
                bool skip = false;
                const size_t edgePointsSize = scattered.size();

                // Skip calculation for point[k] if there exists distance lower then minDist
                for (size_t j = 0; j < edgePointsSize; ++j) {
                    if (cv::norm(points[k].first - scattered[j]) < minDst) {
                        skip = true;
                        break;
                    }
                }

                if (skip) {
                    continue;
                }

                scattered.push_back(points[k].first);
            }
        }

        // Resize result to actual required size
        scattered.resize(count);
    }

    void Matcher::train(std::vector<Template> &templates, uchar minStableVal, uchar minEdgeMag) {
        assert(!templates.empty());
        assert(minStableVal > 0);
        assert(minEdgeMag > 0);

        #pragma omp parallel for shared(templates) firstprivate(criteria, minStableVal, minEdgeMag)
        for (size_t i = 0; i < templates.size(); i++) {
            Template &t = templates[i];
            std::vector<std::pair<cv::Point, uchar>> edgeVPoints;
            std::vector<std::pair<cv::Point, uchar>> stableVPoints;

            cv::Mat grad;
            filterEdges(t.srcGray, grad);

            // Generate stable and edge feature points
            for (int y = 0; y < grad.rows; y++) {
                for (int x = 0; x < grad.cols; x++) {
                    uchar sobelValue = grad.at<uchar>(y, x);
                    uchar stableValue = t.srcGray.at<uchar>(y, x);

                    // Save only points that have valid depth and are in defined thresholds
                    if (sobelValue > minEdgeMag) {
                        edgeVPoints.emplace_back(cv::Point(x, y), sobelValue);
                    } else if (stableValue > minStableVal && t.srcDepth.at<ushort>(y, x) >= t.minDepth) { // skip wrong depth values
                        stableVPoints.emplace_back(cv::Point(x, y), stableValue);
                    }
                }
            }

            // Sort edge points descending (best edge magnitude at top), randomize stable points
            std::shuffle(stableVPoints.rbegin(), stableVPoints.rend(), std::mt19937(std::random_device()()));
            std::stable_sort(edgeVPoints.rbegin(), edgeVPoints.rend(), [](const std::pair<cv::Point, uchar> &left, const std::pair<cv::Point, uchar> &right) {
                return left.second < right.second;
            });

            // Select scattered feature points
            selectScatteredFeaturePoints(edgeVPoints, criteria->featurePointsCount, t.edgePoints);
            selectScatteredFeaturePoints(stableVPoints, criteria->featurePointsCount, t.stablePoints);

            // Validate that we extracted desired number of feature points
            CV_Assert(t.edgePoints.size() == criteria->featurePointsCount);
            CV_Assert(t.stablePoints.size() == criteria->featurePointsCount);

            // Extract features for generated feature points
            for (uint j = 0; j < criteria->featurePointsCount; j++) {
                // Create offsets to object bounding box
                cv::Point stable = t.stablePoints[j];
                cv::Point edge = t.edgePoints[j];

                // Extract features on generated feature points
                t.features.depths.push_back(t.srcDepth.at<ushort>(stable));
                t.features.gradients.emplace_back(t.srcGradients.at<uchar>(edge));
                t.features.normals.emplace_back(t.srcNormals.at<uchar>(stable));
                t.features.colors.push_back(remapBlackWhiteHSV(t.srcHSV.at<cv::Vec3b>(stable)));
            }
            // Calculate median of depths
            t.features.depthMedian = median<ushort>(t.features.depths);

#ifndef NDEBUG
//            Visualizer viz(criteria);
//            viz.tplFeaturePoints(t, 0, "Template feature points");
#endif
        }
    }

    // TODO implement object size test
    int Matcher::testObjectSize(float scale, ushort depth, Window &window, cv::Mat &sceneDepth, cv::Point &stable) {
        const unsigned long fSize = criteria->depthDeviationFun.size();

        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneDepth.cols || offsetP.y >= sceneDepth.rows ||
                    offsetP.x < 0 || offsetP.y < 0)
                    continue;

                // Get depth value at point
                float ratio = 0;
                float sDepth = sceneDepth.at<ushort>(offsetP);
                // TODO better wrong depth handling
                if (sDepth == 0) continue;

                // Get correct deviation ratio
                for (size_t j = 0; j < fSize - 1; j++) {
                    if (sDepth < criteria->depthDeviationFun[j + 1][0]) {
                        ratio = (1 - criteria->depthDeviationFun[j + 1][1]);
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
        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneSurfaceNormalsQuantized.cols || offsetP.y >= sceneSurfaceNormalsQuantized.rows ||
                    offsetP.x < 0 || offsetP.y < 0)
                    continue;

                if (sceneSurfaceNormalsQuantized.at<uchar>(offsetP) == normal) {
                    return 1;
                }
            }
        }

        return 0;
    }

// TODO Use bitwise operations using response maps
    int Matcher::testGradients(uchar gradient, Window &window, cv::Mat &sceneAnglesQuantized, cv::Mat &sceneMagnitudes,
                               cv::Point &edge) {
        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(edge.x + window.tl().x + x, edge.y + window.tl().y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneAnglesQuantized.cols || offsetP.y >= sceneAnglesQuantized.rows ||
                    offsetP.x < 0 || offsetP.y < 0)
                    continue;

                // TODO - make member threshold (detect automatically based on training values)
                if (sceneAnglesQuantized.at<uchar>(offsetP) == gradient &&
                    sceneMagnitudes.at<float>(offsetP) > criteria->minMagnitude) {
                    return 1;
                }
            }
        }

        return 0;
    }

    int Matcher::testDepth(float scale, float diameter, ushort depthMedian, Window &window, cv::Mat &sceneDepth,
                           cv::Point &stable) {
        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneDepth.cols || offsetP.y >= sceneDepth.rows ||
                    offsetP.x < 0 || offsetP.y < 0)
                    continue;

                if ((sceneDepth.at<ushort>(offsetP) - depthMedian * scale) <
                    (criteria->depthK * diameter * criteria->info.depthScaleFactor)) {
                    return 1;
                }
            }
        }

        return 0;
    }

// TODO consider eroding object in training stage to be more tolerant to inaccuracy on the edges
    int Matcher::testColor(cv::Vec3b HSV, Window &window, cv::Mat &sceneHSV, cv::Point &stable) {
        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(stable.x + window.tl().x + x, stable.y + window.tl().y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneHSV.cols || offsetP.y >= sceneHSV.rows ||
                    offsetP.x < 0 || offsetP.y < 0)
                    continue;

                // Normalize scene HSV value
                auto hT = static_cast<int>(HSV[0]);
                auto hS = static_cast<int>(remapBlackWhiteHSV(sceneHSV.at<cv::Vec3b>(offsetP))[0]);

                if (std::abs(hT - hS) < 3) {
                    return 1;
                }
            }
        }

        return 0;
    }

    void Matcher::match(float scale, Scene &scene, std::vector<Window> &windows, std::vector<Match> &matches) {
        // Checks
        assert(!scene.srcDepth.empty());
        assert(!scene.srcNormals.empty());
        assert(!scene.srcHSV.empty());
        assert(!scene.srcMagnitudes.empty());
        assert(scene.srcHSV.type() == CV_8UC3);
        assert(scene.srcDepth.type() == CV_16U);
        assert(scene.srcNormals.type() == CV_8UC1);
        assert(!windows.empty());

        // Init vizaulizer
        Visualizer viz(criteria);

        // Min threshold of matched feature points
        const auto N = criteria->featurePointsCount;
        const auto minThreshold = static_cast<int>(criteria->featurePointsCount * criteria->matchFactor);
        const long lSize = windows.size();

//        #pragma omp parallel for shared(scene, windows, matches) firstprivate(N, minThreshold, scale)
        for (long l = 0; l < lSize; l++) {
            for (auto &candidate : windows[l].candidates) {
                assert(candidate != nullptr);

                // Scores for each test
                float sI = 0, sII = 0, sIII = 0, sIV = 0, sV = 0;

                // Test I
                #pragma omp parallel for reduction(+:sI)
                for (uint i = 0; i < N; i++) {
                    sI += testObjectSize(scale, candidate->features.depths[i], windows[l], scene.srcDepth, candidate->stablePoints[i]);
                }

                if (sI < minThreshold) continue;

                // Test II
                #pragma omp parallel for reduction(+:sII)
                for (uint i = 0; i < N; i++) {
                    sII += testSurfaceNormal(candidate->features.normals[i], windows[l], scene.srcNormals, candidate->stablePoints[i]);
                }

                if (sII < minThreshold) continue;

                // Test III
                #pragma omp parallel for reduction(+:sIII)
                for (uint i = 0; i < N; i++) {
                    sIII += testGradients(candidate->features.gradients[i], windows[l], scene.srcGradients, scene.srcMagnitudes, candidate->edgePoints[i]);
                }

                if (sIII < minThreshold) continue;

                // Test IV
                #pragma omp parallel for reduction(+:sIV)
                for (uint i = 0; i < N; i++) {
                    sIV += testDepth(scale, candidate->diameter, candidate->features.depthMedian, windows[l], scene.srcDepth, candidate->stablePoints[i]);
                }

                if (sIV < minThreshold) continue;

                // Test V
                #pragma omp parallel for reduction(+:sV)
                for (uint i = 0; i < N; i++) {
                    sV += testColor(candidate->features.colors[i], windows[l], scene.srcHSV, candidate->stablePoints[i]);
                }

                if (sV < minThreshold) continue;

                // Push template that passed all tests to matches array
                float score = (sII / N) + (sIII / N) + (sIV / N) + (sV / N);
                cv::Rect matchBB = cv::Rect(windows[l].tl().x, windows[l].tl().y, candidate->objBB.width, candidate->objBB.height);

                #pragma omp critical
                matches.emplace_back(candidate, matchBB, scale, score, score * (candidate->objArea / scale), sI, sII, sIII, sIV, sV);




                // Vizualization
                std::vector<std::pair<cv::Point, int>> vsI, vsII, vsIII, vsIV, vsV;

                // Save validation for all points
                for (uint i = 0; i < N; i++) {
                    vsI.emplace_back(candidate->stablePoints[i], testObjectSize(scale, candidate->features.depths[i], windows[l], scene.srcDepth, candidate->stablePoints[i]));
                    vsII.emplace_back(candidate->stablePoints[i], testSurfaceNormal(candidate->features.normals[i], windows[l], scene.srcNormals, candidate->stablePoints[i]));
                    vsIII.emplace_back(candidate->edgePoints[i], testGradients(candidate->features.gradients[i], windows[l], scene.srcGradients, scene.srcMagnitudes, candidate->edgePoints[i]));
                    vsIV.emplace_back(candidate->stablePoints[i], testDepth(scale, candidate->diameter, candidate->features.depthMedian, windows[l], scene.srcDepth, candidate->stablePoints[i]));
                    vsV.emplace_back(candidate->stablePoints[i], testColor(candidate->features.colors[i], windows[l], scene.srcHSV, candidate->stablePoints[i]));
                }

                // Push each score to scores vector
                std::vector<std::vector<std::pair<cv::Point, int>>> scores = {vsI, vsII, vsIII, vsIV, vsV};

                // Visualize matching
                viz.matching(scene, *candidate, windows[l], scores, criteria->patchOffset, N, minThreshold);
            }
        }
    }
}