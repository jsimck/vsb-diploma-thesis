#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "visualizer.h"
#include "../objdetect/hasher.h"
#include "../processing/processing.h"
#include "../core/classifier_criteria.h"

namespace tless {
    cv::Mat Visualizer::loadTemplateSrc(const Template &tpl, int flags) {
        std::ostringstream oss;
        oss << templatesPath;
        oss << std::setw(2) << std::setfill('0') << static_cast<int>(std::floor(tpl.id / 2000));

        if (flags == CV_LOAD_IMAGE_UNCHANGED) {
            oss << "/depth/" << tpl.fileName << ".png";
        } else {
            oss << "/rgb/" << tpl.fileName << ".png";
        }

        return cv::imread(oss.str(), flags);
    }

    void Visualizer::label(cv::Mat &dst, const std::string &label, const cv::Point &origin, double scale,
                           int padding, int thickness, cv::Scalar fColor, cv::Scalar bColor, int fontFace) {
        cv::Size text = cv::getTextSize(label, fontFace, scale, thickness, nullptr);
        rectangle(dst, origin + cv::Point(-padding - 1, padding + 2),
                  origin + cv::Point(text.width + padding, -text.height - padding - 2), bColor, CV_FILLED);
        putText(dst, label, origin, fontFace, scale, fColor, thickness, CV_AA);
    }

    void Visualizer::visualizeMatches(cv::Mat &scene, float scale, std::vector<Match> &matches, const std::string &templatesPath,
                                      int wait, const char *title) {
        cv::Mat viz = scene.clone();
        std::ostringstream oss;
        int tplCounter = 0;

        for (auto &match : matches) {
            cv::Rect matchBB = match.scaledBB(scale);
            cv::rectangle(viz, matchBB.tl(), matchBB.br(), cv::Scalar(0, 255, 0));

            oss.str("");
            oss << "id: " << match.t->id;
            label(viz, oss.str(), matchBB.tl() + cv::Point(matchBB.width + 5, 10));
            oss.str("");
            oss.precision(2);
            oss << std::fixed << "score: " << match.areaScore << " (" << (match.areaScore * 100.0f) / 4.0f << "%)";
            label(viz, oss.str(), matchBB.tl() + cv::Point(matchBB.width + 5, 28));
            oss.str("");
            oss << "I: " << match.sI;
            label(viz, oss.str(), matchBB.tl() + cv::Point(matchBB.width + 5, 46));
            oss.str("");
            oss << "II: " << match.sII;
            label(viz, oss.str(), matchBB.tl() + cv::Point(matchBB.width + 5, 64));
            oss.str("");
            oss << "III: " << match.sIII;
            label(viz, oss.str(), matchBB.tl() + cv::Point(matchBB.width + 5, 82));
            oss.str("");
            oss << "IV: " << match.sIV;
            label(viz, oss.str(), matchBB.tl() + cv::Point(matchBB.width + 5, 100));
            oss.str("");
            oss << "V: " << match.sV;
            label(viz, oss.str(), matchBB.tl() + cv::Point(matchBB.width + 5, 118));
            oss.str("");
            oss << "scale: " << match.scale;
            label(viz, oss.str(), matchBB.tl() + cv::Point(matchBB.width + 5, 132));

//            // Load matched template
//            cv::Mat tplSrc;
//            Template::loadTemplateSrc(templatesPath, *match.t, tplSrc, CV_LOAD_IMAGE_COLOR);
//
//            // draw label and bounding box
//            cv::rectangle(tplSrc, match.t->objBB.tl(), match.t->objBB.br(), cv::Scalar(0, 255, 0));
//            oss.str("");
//            oss << "id: " << match.t->id;
//            label(tplSrc, oss.str(), cv::Point(match.t->objBB.br().x + 5, match.t->objBB.tl().y + 10));
//
//            oss.str("");
//            oss << "Template: " << tplCounter;
//            std::string winName = oss.str();
//
//            // Show in resizable window
//            cv::namedWindow(winName, 0);
//            cv::imshow(winName, tplSrc);
//            tplCounter++;
        }

        // Show in resizable window
        std::string winName = title != nullptr ? title : "Matched results";
        cv::namedWindow(winName, 0);
        cv::imshow(winName, viz);
        cv::waitKey(wait);
    }

    bool Visualizer::visualizeTests(Template &tpl, const cv::Mat &sceneHSV, const cv::Mat &sceneDepth, Window &window,
                                    std::vector<cv::Point> &stablePoints, std::vector<cv::Point> &edgePoints,
                                    int patchOffset, std::vector<int> &scoreI, std::vector<int> &scoreII,
                                    std::vector<int> &scoreIII, std::vector<int> &scoreIV, std::vector<int> &scoreV,
                                    int pointsCount, int minThreshold, int currentTest, bool continuous,
                                    const std::string &templatesPath, int wait, const char *title) {
        // Init common
        std::ostringstream oss;
        cv::Scalar colorRed(0, 0, 255), colorGreen(0, 255, 0), colorWhite(255, 255, 255), colorBlue(0, 255, 0);
        cv::Point offsetPStart(-patchOffset, -patchOffset), offsetPEnd(patchOffset, patchOffset);

        // Convert matrices
        cv::Mat result, resultScene, resultSceneDepth;
        cv::cvtColor(sceneHSV, resultScene, CV_HSV2BGR);
        int scoreVTrue = 0, scoreIVTrue = 0, scoreIIITrue = 0, scoreIITrue = 0, scoreITrue = 0;

        // Normalize depth scene
        sceneDepth.convertTo(resultSceneDepth, CV_32FC1, 1.0f / 65536.0f);
        cv::normalize(resultSceneDepth, resultSceneDepth, 0, 1.0f, CV_MINMAX);

        // Dynamically load template src
        if (tpl.srcHSV.empty()) {
//            Template::loadSrc(templatesPath, tpl, result, CV_LOAD_IMAGE_COLOR);
        } else {
            cv::cvtColor(tpl.srcHSV, result, CV_HSV2BGR);
        }

        // Draw results
        for (int i = 0; i < pointsCount; i++) {
            cv::Scalar color, bwColor;
            cv::Point point = (currentTest == 3) ? edgePoints[i] : stablePoints[i];

            // Count scores
            scoreITrue += (!scoreI.empty() && scoreI[i] == 0) ? 0 : 1;
            scoreIITrue += (!scoreII.empty() && scoreII[i] == 0) ? 0 : 1;
            scoreIIITrue += (!scoreIII.empty() && scoreIII[i] == 0) ? 0 : 1;
            scoreIVTrue += (!scoreIV.empty() && scoreIV[i] == 0) ? 0 : 1;
            scoreVTrue += (!scoreV.empty() && scoreV[i] == 0) ? 0 : 1;

            // Check if point has been matched
            if (currentTest == 1) {
                color = scoreI[i] == 0 ? colorRed : colorGreen;
                bwColor = scoreI[i] == 0 ? cv::Scalar(0) : cv::Scalar(1.0f);
            } else if (currentTest == 2) {
                color = scoreII[i] == 0 ? colorRed : colorGreen;
                bwColor = scoreII[i] == 0 ? cv::Scalar(0) : cv::Scalar(1.0f);
            } else if (currentTest == 3) {
                color = scoreIII[i] == 0 ? colorRed : colorGreen;
                bwColor = scoreIII[i] == 0 ? cv::Scalar(0) : cv::Scalar(1.0f);
            } else if (currentTest == 4) {
                color = scoreIV[i] == 0 ? colorRed : colorGreen;
                bwColor = scoreIV[i] == 0 ? cv::Scalar(0) : cv::Scalar(1.0f);
            } else if (currentTest == 5) {
                color = scoreV[i] == 0 ? colorRed : colorGreen;
                bwColor = scoreV[i] == 0 ? cv::Scalar(0) : cv::Scalar(1.0f);
            }

            cv::rectangle(resultScene, window.tl() + point + offsetPStart, window.tl() + point + offsetPEnd, color, 1);
            cv::rectangle(resultSceneDepth, window.tl() + point + offsetPStart, window.tl() + point + offsetPEnd,
                          bwColor, 1);
            cv::circle(result, tpl.objBB.tl() + point, 1, color, -1);
        }

        // Draw window rects
        cv::rectangle(resultScene, window.tl(), window.tl() + cv::Point(tpl.objBB.width, tpl.objBB.height), colorWhite,
                      1);
        cv::rectangle(resultSceneDepth, window.tl(), window.tl() + cv::Point(tpl.objBB.width, tpl.objBB.height),
                      cv::Scalar(1.0f), 1);

        // Draw info labels
        oss.str("");
        oss << "I: " << scoreITrue << "/" << pointsCount;
//        Visualizer::label(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 10), currentTest == 1 ? (scoreITrue > minThreshold ? colorGreen : colorRed) : colorWhite, cv::Scalar(), 0.4, 1, 0, 0);
        oss.str("");
        oss << "II: " << scoreIITrue << "/" << pointsCount;
//        Visualizer::label(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 28), currentTest == 2 ? (scoreIITrue > minThreshold ? colorGreen : colorRed) : colorWhite, cv::Scalar(), 0.4, 1, 0, 0);
        oss.str("");
        oss << "III: " << scoreIIITrue << "/" << pointsCount;
//        Visualizer::label(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 46), currentTest == 3 ? (scoreIIITrue > minThreshold ? colorGreen : colorRed) : colorWhite, cv::Scalar(), 0.4, 1, 0, 0);
        oss.str("");
        oss << "IV: " << scoreIVTrue << "/" << pointsCount;
//        Visualizer::label(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 64), currentTest == 4 ? (scoreVTrue > minThreshold ? colorGreen : colorRed) : colorWhite, cv::Scalar(), 0.4, 1, 0, 0);
        oss.str("");
        oss << "V: " << scoreVTrue << "/" << pointsCount;
//        Visualizer::label(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 82), currentTest == 5 ? (scoreVTrue > minThreshold ? colorGreen : colorRed) : colorWhite, cv::Scalar(), 0.4, 1, 0, 0);
        oss.str("");
        oss.precision(2);
        oss << "score: " << std::fixed
            << scoreITrue / static_cast<float>(pointsCount) + scoreIITrue / static_cast<float>(pointsCount) +
               scoreIIITrue / static_cast<float>(pointsCount) + scoreIVTrue / static_cast<float>(pointsCount) +
               scoreVTrue / static_cast<float>(pointsCount);
//        Visualizer::label(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 100), currentTest >= 5 ? (scoreVTrue > minThreshold ? colorGreen : colorRed) : colorWhite, cv::Scalar(), 0.4, 1, 0, 0);

        // Number of candidates on this window
        oss.str("");
        oss << "candidates: " << window.candidates.size();
//        Visualizer::label(resultScene, oss.str(), window.tl() + cv::Point(0, -15), colorWhite, cv::Scalar(), 0.4, 1, 0, 0);

        // Show results
        cv::imshow(title == nullptr ? "Hashing visualization" : oss.str(), result);
        oss.str("");
        oss << title;
        oss << " - scene";
        cv::imshow(title == nullptr ? "Hashing visualization scene" : title, resultScene);
        oss.str("");
        oss << title;
        oss << " - depth";
        cv::imshow(title == nullptr ? "Hashing visualization scene depth" : title, resultSceneDepth);

        // Continuous till match (sV > than min threshold)
        int keyPressed = 0;
        if (continuous) {
            keyPressed = cv::waitKey(scoreVTrue > minThreshold ? 0 : 1);
        } else {
            keyPressed = cv::waitKey(wait);
        }

        // Spacebar pressed
        return keyPressed == 32;
    }

    void Visualizer::windowCandidates(const cv::Mat &src, cv::Mat &dst, Window &window) {
        std::ostringstream oss;
        dst = src.clone();

        // Create template mosaic of found candidates
        if (!window.candidates.empty()) {
            // Define grid, offsets and initialize tpl mosaic matrix
            const int offset = 8, topOffset = 25;
            int x, y, width = window.candidates[0]->objBB.width;
            int sizeX = width + 2 * offset, sizeY = width + offset + topOffset;
            auto gridSize = static_cast<int>(std::ceil(std::sqrt(window.candidates.size())));
            cv::Mat tplMosaic = cv::Mat::zeros(gridSize * sizeY, gridSize * sizeX, CV_8UC3);

            for (int i = 0; i < window.candidates.size(); ++i) {
                Template *candidate = window.candidates[i];

                // Calculate x, y and rect inside defined mosaic grid
                x = (i % gridSize);
                y = (i / gridSize);
                cv::Rect rect(x * sizeX + offset, y * sizeY + offset, width, width);

                // Load template src image
                cv::Mat tplSrc = loadTemplateSrc(*candidate);
                tplSrc.copyTo(tplMosaic(rect));

                // Draw triplets
                for (auto &triplet : window.triplets[i]) {
                    cv::line(tplMosaic, rect.tl() + triplet.c, rect.tl() + triplet.p1, cv::Scalar(0, 180, 0), 1, CV_AA);
                    cv::line(tplMosaic, rect.tl() + triplet.c, rect.tl() + triplet.p2, cv::Scalar(0, 180, 0), 1, CV_AA);
                    cv::circle(tplMosaic, rect.tl() + triplet.c, 2, cv::Scalar(0, 255, 0), -1, CV_AA);
                    cv::circle(tplMosaic, rect.tl() + triplet.p1, 2, cv::Scalar(255, 0, 0), -1, CV_AA);
                    cv::circle(tplMosaic, rect.tl() + triplet.p2, 2, cv::Scalar(0, 0, 255), -1, CV_AA);
                }

                // Annotate templates in mosaic
                cv::rectangle(tplMosaic, rect, cv::Scalar(200, 200, 200), 1);
                oss << "Votes: " << window.votes[i];
                label(tplMosaic, oss.str(), cv::Point(rect.x, rect.y + rect.height + 15));
                oss.str("");
            }

            cv::imshow("Candidate templates", tplMosaic);
        }

        // Annotate scene
        cv::rectangle(dst, window.rect(), cv::Scalar(0, 255, 0), 1, CV_AA);

        // Set labels
        oss << "candidates: " << window.candidates.size();
        label(dst, oss.str(), window.tr() + cv::Point(5, 12));
        oss.str("");
        oss << "edgels: " << window.edgels;
        label(dst, oss.str(), window.tr() + cv::Point(5, 30));
    }

    void Visualizer::windowsCandidates(const Scene &scene, std::vector<Window> &windows, int wait, const char *title) {
        const auto winSize = static_cast<const int>(windows.size());
        std::ostringstream oss;
        cv::Mat result;

        for (int i = 0; i < winSize; ++i) {
            result = scene.srcRGB.clone();

            // Draw all other windows in gray
            for (auto &win : windows) {
                cv::rectangle(result, win.rect(), cv::Scalar(90, 90, 90), 1);
            }

            // Vizualize window candidates
            windowCandidates(result, result, windows[i]);

            // Title
            oss.str("");
            oss << "Locations: " << winSize;
            label(result, oss.str(), cv::Point(2, 14), 0.5, 2);
            oss.str("");
            oss << "Scene: " << scene.id;
            label(result, oss.str(), cv::Point(2, 34), 0.5, 2);

            // Show results
            cv::imshow(title == nullptr ? "Window candidates" : title, result);

            // Get key pressed
            int key = cv::waitKey(wait);

            // Navigation using arrow keys and spacebar
            if (key == KEY_UP) {
                i = (i + 10 < winSize) ? i + 9 : winSize - i - 1;
            } else if (key == KEY_DOWN) {
                i = (i - 10 > 0) ? i - 11 : -1;
            } else if (key == KEY_LEFT) {
                i = (i - 1 > 0) ? i - 2 : -1;
            } else if (key == KEY_ENTER) {
                i = (i + 100 < winSize) ? i + 99 : winSize - i - 1;
            } else if (key == KEY_SPACEBAR) {
                break;
            }
        }
    }

    void Visualizer::objectness(const Scene &scene, std::vector<Window> &windows, int wait, const char *title) {
        const auto winSize = static_cast<const int>(windows.size());
        auto minMag = static_cast<int>(criteria->objectnessDiameterThreshold * criteria->info.smallestDiameter * criteria->info.depthScaleFactor);
        cv::Mat result, depth, edgels, resultDepth, resultEdgels;
        std::ostringstream oss;

        // Normalize min and max depths to look for objectness in
        auto minDepth = static_cast<int>(criteria->info.minDepth * depthNormalizationFactor(criteria->info.minDepth, criteria->depthDeviationFun));
        auto maxDepth = static_cast<int>(criteria->info.maxDepth / depthNormalizationFactor(criteria->info.maxDepth, criteria->depthDeviationFun));

        // Convert depth
        scene.srcDepth.convertTo(depth, CV_8UC1, 255.0f / 65535.0f);
        cv::cvtColor(depth, depth, CV_GRAY2BGR);

        // Edgels computation
        depthEdgels(scene.srcDepth, edgels, minDepth, maxDepth, minMag);
        cv::normalize(edgels, edgels, 0, 255, CV_MINMAX);
        cv::cvtColor(edgels, edgels, CV_GRAY2BGR);

        for (int i = 0; i < windows.size(); ++i) {
            result = scene.srcRGB.clone();
            resultEdgels = edgels.clone();
            resultDepth = depth.clone();

            // Draw all other windows in gray
            for (auto &win : windows) {
                cv::rectangle(result, win.rect(), cv::Scalar(90, 90, 90), 1);
                cv::rectangle(resultEdgels, win.rect(), cv::Scalar(23, 35, 24), 1);
            }

            // Annotate scene
            cv::rectangle(result, windows[i].rect(), cv::Scalar(0, 255, 0), 1, CV_AA);
            cv::rectangle(resultDepth, windows[i].rect(), cv::Scalar(0, 255, 0), 1, CV_AA);
            cv::rectangle(resultEdgels, windows[i].rect(), cv::Scalar(0, 255, 0), 1, CV_AA);

            // Set labels
            oss.str("");
            oss << "edgels: " << windows[i].edgels;
            label(result, oss.str(), windows[i].tr() + cv::Point(5, 12));
            label(resultDepth, oss.str(), windows[i].tr() + cv::Point(5, 12));
            label(resultEdgels, oss.str(), windows[i].tr() + cv::Point(5, 12));

            // Locations title
            oss.str("");
            oss << "Locations: " << windows.size();
            label(result, oss.str(), cv::Point(2, 14), 0.5, 2);
            label(resultDepth, oss.str(), cv::Point(2, 14), 0.5, 2);
            label(resultEdgels, oss.str(), cv::Point(2, 14), 0.5, 2);
            oss.str("");
            oss << "Min edgels: " << (criteria->info.minEdgels * criteria->objectnessFactor);
            label(result, oss.str(), cv::Point(2, 34), 0.5, 2);
            label(resultDepth, oss.str(), cv::Point(2, 34), 0.5, 2);
            label(resultEdgels, oss.str(), cv::Point(2, 34), 0.5, 2);
            oss.str("");
            oss << "Scene: " << scene.id;
            label(result, oss.str(), cv::Point(2, 54), 0.5, 2);
            label(resultDepth, oss.str(), cv::Point(2, 54), 0.5, 2);
            label(resultEdgels, oss.str(), cv::Point(2, 54), 0.5, 2);

            // Show results
            cv::imshow(title == nullptr ? "Objectness locations - rgb" : title, result);
            cv::imshow("Objectness locations - depth", resultDepth);
            cv::imshow("Objectness locations - edgels", resultEdgels);

            // Get key pressed
            int key = cv::waitKey(wait);

            // Navigation using arrow keys and spacebar
            if (key == KEY_UP) {
                i = (i + 10 < winSize) ? i + 9 : winSize - i - 1;
            } else if (key == KEY_DOWN) {
                i = (i - 10 > 0) ? i - 11 : -1;
            } else if (key == KEY_LEFT) {
                i = (i - 1 > 0) ? i - 2 : -1;
            } else if (key == KEY_ENTER) {
                i = (i + 100 < winSize) ? i + 99 : winSize - i - 1;
            } else if (key == KEY_SPACEBAR) {
                break;
            }
        }
    }

    void Visualizer::tplFeaturePoints(const Template &t, int wait, const char *title) {
        // Dynamically load template
        cv::Mat tplRGB = loadTemplateSrc(t, CV_LOAD_IMAGE_COLOR);
        cv::Mat tplDepth = loadTemplateSrc(t, CV_LOAD_IMAGE_UNCHANGED);
        tplDepth.convertTo(tplDepth, CV_8UC1, 255.0f / 65535.0f);
        cv::cvtColor(tplDepth, tplDepth, CV_GRAY2BGR);

        // Compute edges
        cv::Mat gray, edges;
        cv::cvtColor(tplRGB, gray, CV_BGR2GRAY);
        filterEdges(t.srcGray, edges);
        cv::cvtColor(edges, edges, CV_GRAY2BGR);

        // ROIs
        const int offset = 10;
        cv::Rect rgbROI(offset, offset, tplRGB.cols, tplRGB.rows);
        cv::Rect depthROI(rgbROI.br().x + offset, offset, tplDepth.cols, tplDepth.rows);
        cv::Rect edgesROI(depthROI.br().x + offset, offset, tplDepth.cols, tplDepth.rows);

        // Result size
        cv::Size resultSize(rgbROI.width * 3 + offset * 4, rgbROI.height + offset * 2 + 130);
        cv::Mat result = cv::Mat::zeros(resultSize, CV_8UC3);
        
        // Copy to result roi
        tplRGB.copyTo(result(rgbROI));
        tplDepth.copyTo(result(depthROI));
        edges.copyTo(result(edgesROI));

        // Draw bounding box
        cv::rectangle(result, rgbROI.tl(), rgbROI.br(), cv::Scalar(90, 90, 90), 1);
        cv::rectangle(result, depthROI.tl(), depthROI.br(), cv::Scalar(90, 90, 90), 1);
        cv::rectangle(result, edgesROI.tl(), edgesROI.br(), cv::Scalar(90, 90, 90), 1);

        // Draw edge points
        if (!t.edgePoints.empty()) {
            for (auto &point : t.edgePoints) {
                cv::circle(result, point + rgbROI.tl(), 1, cv::Scalar(0, 0, 255), -1, CV_AA);
                cv::circle(result, point + edgesROI.tl(), 1, cv::Scalar(0, 0, 255), -1, CV_AA);
            }
        }

        // Draw stable points
        if (!t.stablePoints.empty()) {
            for (auto &point : t.stablePoints) {
                cv::circle(result, point + rgbROI.tl(), 1, cv::Scalar(255, 0, 0), -1, CV_AA);
                cv::circle(result, point + depthROI.tl(), 1, cv::Scalar(255, 0, 0), -1, CV_AA);
            }
        }

        // Put text data to template image
        std::ostringstream oss;
        cv::Point textTl(offset, rgbROI.height + offset);

        oss.str("");
        oss << "Template: " << t.fileName << " (" << (t.id / 2000) << ")";
        textTl.y += 18;
        label(result, oss.str(), textTl);

        oss.str("");
        oss << "mode: " << t.camera.mode;
        textTl.y += 18;
        label(result, oss.str(), textTl);

        oss.str("");
        oss << "elev: " << t.camera.elev;
        textTl.y += 18;
        label(result, oss.str(), textTl);

        oss.str("");
        oss << "srcGradients: " << t.features.gradients.size();
        textTl.y += 18;
        label(result, oss.str(), textTl);

        oss.str("");
        oss << "srcNormals: " << t.features.normals.size();
        textTl.y += 18;
        label(result, oss.str(), textTl);

        oss.str("");
        oss << "depths: " << t.features.depths.size();
        textTl.y += 18;
        label(result, oss.str(), textTl);

        oss.str("");
        oss << "colors: " << t.features.colors.size();
        textTl.y += 18;
        label(result, oss.str(), textTl);

        cv::imshow(title == nullptr ? "Template features" : title, result);
        cv::waitKey(wait);
    }

    void Visualizer::matching(const Scene &scene, const Template &candidate, Window &window,
                              std::vector<std::vector<std::pair<cv::Point, int>>> scores, int patchOffset,
                              int pointsCount, int minThreshold, int wait, const char *title) {
        std::ostringstream oss;
        cv::Mat result = scene.srcRGB.clone();
        cv::Scalar cGreen(0, 255, 0), cRed(0, 0, 255), cBlue(255, 0, 0);
        cv::Point offsetStart(-patchOffset, -patchOffset), offsetEnd(patchOffset, patchOffset);

        for (int i = 0; i < scores.size(); ++i) {
            // Draw points
            for (auto &score : scores[i]) {
                cv::Scalar color = (score.second == 1) ? cGreen : cRed;
                cv::Point tl = window.tl() + score.first + offsetStart;
                cv::Point br = window.tl() + score.first + offsetEnd;

                // Draw small rectangles around matched feature points
                cv::rectangle(result, tl, br, color, 1);
                cv::rectangle(result, tl, br, color, 1);
            }

            // Draw rectangle around object
            cv::rectangle(result, window.tl(), window.br(), cGreen, 1);

            // Annotate scene
            cv::Point textTl(window.br().x + 5, window.tl().y + 10);

            oss.str("");
            oss << "Test: " << i + 1;
            label(result, oss.str(), textTl);
            textTl.y += 18;

            oss.str("");
            oss << "Template: " << candidate.fileName << " (" << (candidate.id / 2000) << ")";
            label(result, oss.str(), textTl);
            textTl.y += 18;

            // Draw scores
            float finalScore = 0;
            for (int l = 0; l < scores.size(); ++l) {
                int score = 0;

                for (auto &point : scores[l]) {
                    score += point.second;
                }

                finalScore += score;
                oss.str("");

                oss << "s" << (l + 1) << ": " << score << "/" << pointsCount;
                label(result, oss.str(), textTl, 0.4, 1, 1, cv::Scalar(0, 0, 0), (score < minThreshold) ? cRed : cGreen);
                textTl.y += 18;
            }

            oss.str("");
            oss << "Score: " << (finalScore / 100.0f);
            label(result, oss.str(), textTl);
            textTl.y += 18;

            // Show results and save key press
            cv::imshow(title == nullptr ? "Matched feature points" : title, result);
            int key = cv::waitKey(wait);

            // Navigation using arrow keys and spacebar
             if (key == KEY_LEFT) {
                i = (i - 1 > 0) ? i - 2 : -1;
            } else if (key == KEY_SPACEBAR) {
                break;
            }
        }
    }
}