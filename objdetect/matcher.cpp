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
    void Matcher::selectScatteredFeaturePoints(const std::vector<std::pair<cv::Point, uchar>> &points, uint count, std::vector<cv::Point> &scattered) {
        // Define initial distance
        float minDst = points.size() / criteria->featurePointsCount;

        // Continue decreasing min distance in each loop, till we find >= number of desired points
        while (scattered.size() < count) {
            scattered.clear();
            minDst -= 0.5f;

            // Calculate euqlidian distance between each point
            for (size_t k = 0; k < points.size(); ++k) {
                bool skip = false;

                // Skip calculation for point[k] if there exists distance lower then minDist
                for (size_t j = 0; j < scattered.size(); ++j) {
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
                    uchar gradientValue = t.srcGradients.at<uchar>(y, x);
                    uchar stableValue = t.srcGray.at<uchar>(y, x);

                    // Save only points that have valid depth and are in defined thresholds
                    if (sobelValue > minEdgeMag && gradientValue != 0) {
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
                t.features.hue.push_back(t.srcHue.at<uchar>(stable));
            }

#ifndef NDEBUG
//            Visualizer viz(criteria);
//            viz.tplFeaturePoints(t, 0, "Template feature points");
#endif
        }
    }

    int Matcher::depthDiffMedian(const cv::Mat &sceneDepth, const cv::Point &windowTl, const std::vector<cv::Point> &stablePoints, const std::vector<ushort> &tplDepths) {
        std::vector<int> diffs(stablePoints.size());

        // Accumulate depth differences
        for (uint i = 0; i < stablePoints.size(); i++) {
            ushort d = sceneDepth.at<ushort>(windowTl + stablePoints[i]);

            // Skip invalid depth pixels
            if (d == 0) {
                continue;
            }

            diffs[i] = tplDepths[i] - d;
        }

        return median<int>(diffs);
    }

    int Matcher::testObjectSize(ushort depth, Window &window, cv::Mat &sceneDepth, cv::Point &stable) {
        auto tl = window.tl() + stable;

        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(tl.x + x, tl.y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneDepth.cols || offsetP.y >= sceneDepth.rows || offsetP.x < 0 || offsetP.y < 0) {
                    continue;
                }

                // Get depth value at point
                float sDepth = sceneDepth.at<ushort>(offsetP);

                // Validate depth
                if (sDepth == 0) continue;

                // Check if template depth is in defined bounds  of scene depth
                if (sDepth >= (depth * criteria->depthDeviation) && sDepth <= (depth / criteria->depthDeviation)) {
                    return 1;
                }
            }
        }

        return 0;
    }

    int Matcher::testSurfaceNormal(uchar normal, Window &window, cv::Mat &sceneSurfaceNormalsQuantized, cv::Point &stable) {
        auto tl = window.tl() + stable;

        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(tl.x + x, tl.y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneSurfaceNormalsQuantized.cols || offsetP.y >= sceneSurfaceNormalsQuantized.rows || offsetP.x < 0 || offsetP.y < 0) {
                    continue;
                }

                if (sceneSurfaceNormalsQuantized.at<uchar>(offsetP) == normal) {
                    return 1;
                }
            }
        }

        return 0;
    }

    int Matcher::testGradients(uchar gradient, Window &window, cv::Mat &sceneAnglesQuantized, cv::Point &edge) {
        auto tl = window.tl() + edge;

        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(tl.x + x, tl.y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneAnglesQuantized.cols || offsetP.y >= sceneAnglesQuantized.rows || offsetP.x < 0 || offsetP.y < 0) {
                    continue;
                }

                if (sceneAnglesQuantized.at<uchar>(offsetP) != 0 && sceneAnglesQuantized.at<uchar>(offsetP) == gradient) {
                    return 1;
                }
            }
        }

        return 0;
    }

    int Matcher::testDepth(cv::Mat &sceneDepth, const cv::Point &windowTl, float diameter, int depthMedian, ushort depth, const cv::Point &stable) {
        auto tl = windowTl + stable;

        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(tl.x + x, tl.y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneDepth.cols || offsetP.y >= sceneDepth.rows || offsetP.x < 0 || offsetP.y < 0) {
                    continue;
                }

                if (std::abs((depth - sceneDepth.at<ushort>(offsetP)) - depthMedian) < diameter) {
                    return 1;
                }
            }
        }

        return 0;
    }

    int Matcher::testColor(uchar hue, Window &window, cv::Mat &sceneHSV, cv::Point &stable) {
        auto tl = window.tl() + stable;

        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(tl.x + x, tl.y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneHSV.cols || offsetP.y >= sceneHSV.rows ||
                    offsetP.x < 0 || offsetP.y < 0)
                    continue;

                if (std::abs(hue - sceneHSV.at<uchar>(offsetP)) < criteria->maxHueDiff) {
                    return 1;
                }
            }
        }

        return 0;
    }

    void Matcher::match(ScenePyramid &scene, std::vector<Window> &windows, std::vector<Match> &matches) {
        // Checks
        assert(!scene.srcDepth.empty());
        assert(!scene.srcNormals.empty());
        assert(!scene.srcHue.empty());
        assert(scene.srcHue.type() == CV_8UC1);
        assert(scene.srcDepth.type() == CV_16U);
        assert(scene.srcNormals.type() == CV_8UC1);
        assert(!windows.empty());

        // Init vizaulizer
        Visualizer viz(criteria);

        // Min threshold of matched feature points
        const auto N = criteria->featurePointsCount;
        const auto minThreshold = static_cast<int>(criteria->featurePointsCount * criteria->matchFactor);

        #pragma omp parallel for shared(scene, windows, matches) firstprivate(N, minThreshold)
        for (int l = 0; l < windows.size(); l++) {
            const cv::Point windowTl = windows[l].tl();
            int depthMedian;
            float diameter;

            for (int c = 0; c < windows[l].candidates.size(); ++c) {
                Template *candidate = windows[l].candidates[c];
                assert(candidate != nullptr);

#ifndef NDEBUG
//                // Accumulate depth differences
//                depthMedian = depthDiffMedian(scene.srcDepth, windows[l].tl(), candidate->stablePoints, candidate->features.depths);
//                diameter = candidate->diameter * criteria->info.depthScaleFactor * criteria->depthK;
//
//                // Vizualization
//                std::vector<std::pair<cv::Point, int>> vsI, vsII, vsIII, vsIV, vsV;
//
//                // Save validation for all points
//                for (uint i = 0; i < N; i++) {
//                    vsI.emplace_back(candidate->stablePoints[i], testObjectSize(candidate->features.depths[i], windows[l], scene.srcDepth, candidate->stablePoints[i]));
//                    vsII.emplace_back(candidate->stablePoints[i], testSurfaceNormal(candidate->features.normals[i], windows[l], scene.srcNormals, candidate->stablePoints[i]));
//                    vsIII.emplace_back(candidate->edgePoints[i], testGradients(candidate->features.gradients[i], windows[l], scene.srcGradients, candidate->edgePoints[i]));
//                    vsIV.emplace_back(candidate->stablePoints[i], testDepth(testDepth(scene.srcDepth, windowTl, diameter, depthMedian, candidate->features.depths[i], candidate->stablePoints[i]));
//                    vsV.emplace_back(candidate->stablePoints[i], testColor(candidate->features.hue[i], windows[l], scene.srcHue, candidate->stablePoints[i]));
//                }
//
//                // Push each score to scores vector
//                std::vector<std::vector<std::pair<cv::Point, int>>> scores = {vsI, vsII, vsIII, vsIV, vsV};
//
//                // Visualize matching
//                if (viz.matching(scene, *candidate, windows, l, c, scores, criteria->patchOffset, N, minThreshold)) {
//                    break;
//                }
#endif
                // Scores for each test
                float sI = 0, sII = 0, sIII = 0, sIV = 0, sV = 0;

                // Test I
                for (uint i = 0; i < N; i++) {
                    sI += testObjectSize(candidate->features.depths[i], windows[l], scene.srcDepth, candidate->stablePoints[i]);
                }

                if (sI < minThreshold) continue;

                // Test II
                for (uint i = 0; i < N; i++) {
                    sII += testSurfaceNormal(candidate->features.normals[i], windows[l], scene.srcNormals, candidate->stablePoints[i]);
                }

                if (sII < minThreshold) continue;

                // Test III
                for (uint i = 0; i < N; i++) {
                    sIII += testGradients(candidate->features.gradients[i], windows[l], scene.srcGradients, candidate->edgePoints[i]);
                }

                if (sIII < minThreshold) continue;

                // Test IV
                // Calculate depth median accross differences
                depthMedian = depthDiffMedian(scene.srcDepth, windows[l].tl(), candidate->stablePoints, candidate->features.depths);
                diameter = candidate->diameter * criteria->info.depthScaleFactor * criteria->depthK;

                // Perform depth test over stable points
                for (uint i = 0; i < N; i++) {
                    sIV += testDepth(scene.srcDepth, windowTl, diameter, depthMedian, candidate->features.depths[i], candidate->stablePoints[i]);
                }

                if (sIV < minThreshold) continue;

                // Test V
                for (uint i = 0; i < N; i++) {
                    sV += testColor(candidate->features.hue[i], windows[l], scene.srcHue, candidate->stablePoints[i]);
                }

                if (sV < minThreshold) continue;

                // Push template that passed all tests to matches array
                float score = (sI / N) + (sII / N) + (sIII / N) + (sIV / N) + (sV / N);
                cv::Rect matchBB = cv::Rect(windows[l].tl().x, windows[l].tl().y, candidate->objBB.width, candidate->objBB.height);

                // This section is almost never executed at the same time, as the tests do have non-uniform results, also most of the windows never passes the fifth test
                #pragma omp critical
                matches.emplace_back(candidate, matchBB, scene.scale, score, score * (candidate->objArea / scene.scale), sI, sII, sIII, sIV, sV);
            }
        }
    }
}
