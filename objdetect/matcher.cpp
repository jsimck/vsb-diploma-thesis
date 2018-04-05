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

#ifndef VIZ_TPL_FEATURES
        #pragma omp parallel for shared(templates) firstprivate(criteria, minStableVal, minEdgeMag)
#endif
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
            unsigned long depthSum = 0;
            for (uint j = 0; j < criteria->featurePointsCount; j++) {
                // Create offsets to object bounding box
                cv::Point stable = t.stablePoints[j];
                cv::Point edge = t.edgePoints[j];

                // Extract features on generated feature points
                t.features.depths.push_back(t.srcDepth.at<ushort>(stable));
                t.features.gradients.emplace_back(t.srcGradients.at<uchar>(edge));
                t.features.normals.emplace_back(t.srcNormals.at<uchar>(stable));
                t.features.hue.push_back(t.srcHue.at<uchar>(stable));
                depthSum += t.srcDepth.at<ushort>(stable);
            }

            // Calculate avg depth
            t.features.avgDepth = static_cast<ushort>(depthSum / criteria->featurePointsCount);

#ifdef VIZ_TPL_FEATURES
            Visualizer viz(criteria);
            viz.tplFeaturePoints(t, 1, "Template feature points");
#endif
        }
    }

    int Matcher::depthDiffMedian(const cv::Mat &sceneDepth, const std::vector<cv::Point> &stablePoints, const std::vector<ushort> &tplDepths) {
        std::vector<int> diffs(stablePoints.size());

        // Accumulate depth differences
        for (uint i = 0; i < stablePoints.size(); i++) {
            ushort d = sceneDepth.at<ushort>(stablePoints[i]);

            // Skip invalid depth pixels
            if (d == 0) {
                continue;
            }

            diffs[i] = tplDepths[i] - d;
        }

        return median<int>(diffs);
    }

    bool Matcher::testObjectSize(const cv::Mat &sceneDepth, const cv::Point winCenter, ushort avgDepth) {
        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to win center (we check in defined patch
                cv::Point offsetP(winCenter.x + x, winCenter.y + y);
                auto sDepth = sceneDepth.at<ushort>(offsetP);

                if (sDepth >= (avgDepth * criteria->depthDeviation) && sDepth <= (avgDepth / criteria->depthDeviation)) {
                    return true;
                }
            }
        }

        return false;
    }

    int Matcher::testDepth(const cv::Mat &sceneDepth, const cv::Point &stable, ushort depth, int depthMedian, float diameter) {
        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(stable.x + x, stable.y + y);

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

    int Matcher::testColor(const cv::Mat &sceneHue, const cv::Point &stable, uchar hue) {
        for (int y = -criteria->patchOffset; y <= criteria->patchOffset; ++y) {
            for (int x = -criteria->patchOffset; x <= criteria->patchOffset; ++x) {
                // Apply needed offsets to feature point
                cv::Point offsetP(stable.x + x, stable.y + y);

                // Template points in larger templates can go beyond scene boundaries (don't count)
                if (offsetP.x >= sceneHue.cols || offsetP.y >= sceneHue.rows || offsetP.x < 0 || offsetP.y < 0) {
                    continue;
                }

                if (std::abs(hue - sceneHue.at<uchar>(offsetP)) < criteria->maxHueDiff) {
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
            std::vector<cv::Point> offsetStable(N), offsetEdge(N); // Array of feature points shifted to currently processed window
            const cv::Point winTl = windows[l].tl();
            const cv::Point winCenter(winTl.x + windows[l].width / 2, winTl.y + windows[l].height / 2);
            float diameter;
            int depthMedian;

            for (int c = 0; c < windows[l].candidates.size(); ++c) {
                Template *candidate = windows[l].candidates[c];
                assert(candidate != nullptr);

                // Offset all feature points to coordinates of current window
                for (int i = 0; i < N; ++i) {
                    offsetStable[i] = candidate->stablePoints[i] + winTl;
                    offsetEdge[i] = candidate->edgePoints[i] + winTl;
                }

#ifndef NDEBUG
//                // Accumulate depth differences
//                depthMedian = depthDiffMedian(scene.srcDepth, offsetStable, candidate->features.depths);
//                diameter = candidate->diameter * criteria->info.depthScaleFactor * criteria->depthK;
//
//                // Vizualization
//                std::vector<std::pair<cv::Point, int>> vsI, vsII, vsIII, vsIV, vsV;
//
//                // Object size test
//                std::cout << winCenter << std::endl;
//                vsI.emplace_back(cv::Point(candidate->objBB.x + candidate->objBB.width / 2, candidate->objBB.y + candidate->objBB.height / 2),
//                                 testObjectSize(scene.srcDepth, winCenter, candidate->features.avgDepth));
//
//                // Save validation for all points
//                for (uint i = 0; i < N; i++) {
//                    vsII.emplace_back(candidate->stablePoints[i], testNormals(scene.srcNormals, offsetStable[i], candidate->features.normals[i]));
//                    vsIII.emplace_back(candidate->edgePoints[i], testGradients(scene.srcGradients, offsetEdge[i], candidate->features.gradients[i]));
//                    vsIV.emplace_back(candidate->stablePoints[i], testDepth(scene.srcDepth, offsetStable[i], candidate->features.depths[i], depthMedian, diameter));
//                    vsV.emplace_back(candidate->stablePoints[i], testColor(scene.srcHue, offsetStable[i], candidate->features.hue[i]));
//                }
//
//                // Push each score to scores vector
//                std::vector<std::vector<std::pair<cv::Point, int>>> scores = {vsI, vsII, vsIII, vsIV, vsV};
//
//                // Visualize matching
//                if (viz.matching(scene, *candidate, windows, l, c, scores, criteria->patchOffset, minThreshold)) {
//                    break;
//                }
#endif
                // Scores for each test
                float sII = 0, sIII = 0, sIV = 0, sV = 0;

                // TEST I
                if (!testObjectSize(scene.srcDepth, winCenter, candidate->features.avgDepth)) continue;

                // Test II
                for (uint i = 0; i < N; i++) {
                    sII += (scene.spreadNormals.at<uchar>(offsetStable[i]) & candidate->features.normals[i]) > 0;
                }

                if (sII < minThreshold) continue;

                // Test III
                for (uint i = 0; i < N; i++) {
                    sIII += (scene.spreadGradients.at<uchar>(offsetEdge[i]) & candidate->features.gradients[i]) > 0;
                }

                if (sIII < minThreshold) continue;

                // Test IV
                // Calculate depth median accross differences
                depthMedian = depthDiffMedian(scene.srcDepth, offsetStable, candidate->features.depths);
                diameter = candidate->diameter * criteria->info.depthScaleFactor * criteria->depthK;

                // Perform depth test over stable points
                for (uint i = 0; i < N; i++) {
                    sIV += testDepth(scene.srcDepth, offsetStable[i], candidate->features.depths[i], depthMedian, diameter);
                }

                if (sIV < minThreshold) continue;

                // Test V
                for (uint i = 0; i < N; i++) {
                    sV += testColor(scene.srcHue, offsetStable[i], candidate->features.hue[i]);
                }

                if (sV < minThreshold) continue;

                // Push template that passed all tests to matches array
                float score = (sII / N) + (sIII / N) + (sIV / N) + (sV / N);
                cv::Rect matchBB = cv::Rect(windows[l].tl().x, windows[l].tl().y, candidate->objBB.width, candidate->objBB.height);

                // This section is almost never executed at the same time, as the tests do have non-uniform results, also most of the windows never passes the fifth test
                #pragma omp critical
                matches.emplace_back(candidate, matchBB, scene.scale, score * (candidate->objArea / scene.scale));
            }
        }
    }
}
