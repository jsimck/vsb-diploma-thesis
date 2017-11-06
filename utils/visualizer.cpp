#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "visualizer.h"
#include "../objdetect/hasher.h"

cv::Vec3b Visualizer::heatMapValue(int min, int max, int value) {
    float range = max - min;
    float percentage = 0;

    if (range) {
        percentage = static_cast<float>(value - min) / range;
    } else {
        return cv::Vec3b(90, 90, 90);
    }

    if (percentage >= 0 && percentage < 0.1f) {
        return cv::Vec3b(90, 90, 90);
    } if (percentage >= 0.1f && percentage < 0.2f) {
        return cv::Vec3b(180, 181, 201);
    } if (percentage >= 0.2f && percentage < 0.3f) {
        return cv::Vec3b(178, 179, 226);
    } if (percentage >= 0.3f && percentage < 0.4f) {
        return cv::Vec3b(174, 176, 244);
    } if (percentage >= 0.4f && percentage < 0.5f) {
        return cv::Vec3b(151, 153, 241);
    } if (percentage >= 0.5f && percentage < 0.6f) {
        return cv::Vec3b(126, 129, 239);
    } if (percentage >= 0.6f && percentage < 0.7f) {
        return cv::Vec3b(96, 101, 237);
    } if (percentage >= 0.7f && percentage < 0.8f) {
        return cv::Vec3b(74, 82, 236);
    } if (percentage >= 0.8f && percentage < 0.9f) {
        return cv::Vec3b(51, 62, 235);
    } else {
        return cv::Vec3b(35, 50, 235);
    }
}

void Visualizer::setLabel(cv::Mat &im, const std::string &label, const cv::Point &origin, int padding, int fontFace,
                          double scale, const cv::Scalar &fColor, const cv::Scalar &bColor, int thickness) {
    cv::Size text = cv::getTextSize(label, fontFace, scale, thickness, 0);
    rectangle(im, origin + cv::Point(-padding - 1, padding + 2),
              origin + cv::Point(text.width + padding, -text.height - padding - 2), bColor, CV_FILLED);
    putText(im, label, origin, fontFace, scale, fColor, thickness, CV_AA);
}

void Visualizer::visualizeWindow(cv::Mat &scene, Window &window) {
    std::ostringstream oss;
    cv::rectangle(scene, window.tl(), window.br(), cv::Scalar(0, 255, 0), 1);

    // Set labels
    oss << "candidates: " << window.candidates.size();
    setLabel(scene, oss.str(), window.tr() + cv::Point(5, 10));

    oss.str("");
    oss << "edges: " << window.edgels;
    setLabel(scene, oss.str(), window.tr() + cv::Point(5, 28));
}

void Visualizer::visualizeWindows(cv::Mat &scene, std::vector<Window> &windows, bool continuous, const char *title) {
    // Init common variables
    cv::Mat result = scene.clone();

    // Title
    std::ostringstream oss;
    oss << "Window result detected: " << windows.size();
    std::string defTitle = oss.str();

    // Scene label
    oss.str("");
    oss << "Locations: " << windows.size();
    std::string locTitle = oss.str();

    // Sort windows for heat map
    std::vector<Window> sortedWindows(windows.size());
    std::copy(windows.begin(), windows.end(), sortedWindows.begin());
    std::sort(sortedWindows.begin(), sortedWindows.end());

    // Get max for heat map
    unsigned long max = sortedWindows[sortedWindows.size() - 1].candidates.size();

    if (continuous) {
        // Visualize each window
        for (auto &window : windows) {
            result = scene.clone();

            // Draw all windows in gray
            for (auto win : sortedWindows) {
                cv::rectangle(result, win.tl(), win.br(), heatMapValue(0, static_cast<int>(max), static_cast<int>(win.candidates.size())));
            }

            // Show current window with labels
            visualizeWindow(result, window);
            setLabel(result, locTitle, cv::Point(0, 20), 4, 0, 0.5);

            // Show results
            cv::imshow(title == nullptr ? oss.str() : title, result);
            cv::waitKey(50);
        }
    } else {
        // Draw all windows in gray
        for (auto win : windows) {
            cv::rectangle(result, win.tl(), win.br(), heatMapValue(0, static_cast<int>(max), static_cast<int>(win.candidates.size())));
        }

        // Show current window with labels
        Visualizer::visualizeWindow(result, sortedWindows[0]);
        setLabel(result, locTitle, cv::Point(0, 20), 4, 0, 0.5);
    }

    // Show results
    cv::imshow(title == nullptr ? oss.str() : title, result);
    cv::waitKey(0);
}

void Visualizer::visualizeMatches(cv::Mat &scene, std::vector<Match> &matches) {
    cv::Mat viz = scene.clone();

    for (auto &match : matches) {
        cv::rectangle(viz, cv::Point(match.objBB.x, match.objBB.y), cv::Point(match.objBB.x + match.objBB.width, match.objBB.y + match.objBB.height), cv::Scalar(0, 255, 0));

        std::ostringstream oss;
        oss << "id: " << match.tpl->id;
        setLabel(viz, oss.str(), cv::Point(match.objBB.br().x + 5, match.objBB.tl().y + 10));
        oss.str("");
        oss.precision(2);
        oss << std::fixed << "score: " << match.score << " (" << (match.score * 100.0f) / 4.0f << "%)";
        setLabel(viz, oss.str(), cv::Point(match.objBB.br().x + 5, match.objBB.tl().y + 28));

//        for (auto &group : groups) {
//            for (auto &tpl : group.templates) {
//                if (tpl.id == match.tpl->id) {
//                    // Crop template src
//                    cv::Mat tplSrc = tpl.srcHSV(tpl.objBB).clone();
//                    cv::cvtColor(tplSrc, tplSrc, CV_HSV2BGR);
//
//                    oss.str("");
//                    oss << "Template id: " << tpl.id;
//                    std::string winName = oss.str();
//
//                    // Show in resizable window
//                    cv::namedWindow(winName, 0);
//                    cv::imshow(winName, tplSrc);
//                }
//            }
//        }
    }

    // Show in resizable window
    std::string winName = "Matched results";
    cv::namedWindow(winName, 0);
    cv::imshow(winName, viz);
    cv::waitKey(0);
}

void Visualizer::visualizeTemplate(Template &tpl, const char *title) {
    cv::Mat result = tpl.srcHSV.clone();
    cv::cvtColor(tpl.srcHSV, result, CV_HSV2BGR);

    // Draw edge points
    if (!tpl.edgePoints.empty()) {
        for (auto &point : tpl.edgePoints) {
            cv::circle(result, point + tpl.objBB.tl(), 1, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Draw stable points
    if (!tpl.stablePoints.empty()) {
        for (auto &point : tpl.stablePoints) {
            cv::circle(result, point + tpl.objBB.tl(), 1, cv::Scalar(255, 0, 0), -1);
        }
    }

    // Draw bounding box
    cv::rectangle(result, tpl.objBB.tl(), tpl.objBB.br(), cv::Scalar(255 ,255, 255), 1);

    // Put text data to template image
    std::ostringstream oss;
    oss << "votes: " << tpl.votes;
    setLabel(result, oss.str(), tpl.objBB.tl() + cv::Point(tpl.objBB.width + 5, 10));
    oss.str("");
    oss << "mode: " << tpl.mode;
    setLabel(result, oss.str(), tpl.objBB.tl() + cv::Point(tpl.objBB.width + 5, 28));
    oss.str("");
    oss << "elev: " << tpl.elev;
    setLabel(result, oss.str(), tpl.objBB.tl() + cv::Point(tpl.objBB.width + 5, 46));
    oss.str("");
    oss << "gradients: " << tpl.features.gradients.size();
    setLabel(result, oss.str(), tpl.objBB.tl() + cv::Point(tpl.objBB.width + 5, 64));
    oss.str("");
    oss << "normals: " << tpl.features.normals.size();
    setLabel(result, oss.str(), tpl.objBB.tl() + cv::Point(tpl.objBB.width + 5, 82));
    oss.str("");
    oss << "depths: " << tpl.features.depths.size();
    setLabel(result, oss.str(), tpl.objBB.tl() + cv::Point(tpl.objBB.width + 5, 100));
    oss.str("");
    oss << "colors: " << tpl.features.colors.size();
    setLabel(result, oss.str(), tpl.objBB.tl() + cv::Point(tpl.objBB.width + 5, 118));

    oss.str("");
    oss << "Template: " << tpl.id;
    cv::imshow(title == nullptr ? "Window locations detected:" : title, result);
    cv::waitKey(0);
}

void Visualizer::visualizeHashing(cv::Mat &scene, cv::Mat &sceneDepth, std::vector<HashTable> &tables, std::vector<Window> &windows,
                                  DataSetInfo &info, const cv::Size &grid, bool continuous, const char *title) {
    // Init common
    cv::Scalar colorRed(0, 0, 255);
    std::ostringstream oss;

    for (size_t i = 0, windowsSize = windows.size(); i < windowsSize; ++i) {
        cv::Mat result = scene.clone();
        int matched = 0;

        // Draw processed windows
        for (size_t j = 0; j < i; j++) {
            cv::rectangle(result, windows[j].tl(), windows[j].br(), cv::Scalar::all(90));
        }

        // Draw window and searched box rectangles
        cv::rectangle(result, windows[i].tl(), windows[i].tl() + cv::Point(info.maxTemplate), cv::Scalar::all(255));
        cv::rectangle(result, windows[i].tl(), windows[i].br(), cv::Scalar(0, 255, 0));

        for (auto &table : tables) {
            // Prepare params to load hash key
            TripletParams params(info.maxTemplate.width, info.maxTemplate.height, grid, windows[i].tl().x, windows[i].tl().y);
            cv::Point c = table.triplet.getCenter(params);
            cv::Point p1 = table.triplet.getP1(params);
            cv::Point p2 = table.triplet.getP2(params);

            // If any point of triplet is out of scene boundaries, ignore it to not get false data
            if ((c.x < 0 || c.x >= sceneDepth.cols || c.y < 0 || c.y >= sceneDepth.rows) ||
                (p1.x < 0 || p1.x >= sceneDepth.cols || p1.y < 0 || p1.y >= sceneDepth.rows) ||
                (p2.x < 0 || p2.x >= sceneDepth.cols || p2.y < 0 || p2.y >= sceneDepth.rows)) continue;

            // Relative depths
            cv::Vec2i d = Hasher::relativeDepths(sceneDepth, c, p1, p2);

            // Generate hash key
            HashKey key(
                Hasher::quantizeDepth(d[0], table.binRanges, static_cast<uint>(table.binRanges.size())),
                Hasher::quantizeDepth(d[1], table.binRanges, static_cast<uint>(table.binRanges.size())),
                Hasher::quantizeSurfaceNormal(Hasher::surfaceNormal(sceneDepth, c)),
                Hasher::quantizeSurfaceNormal(Hasher::surfaceNormal(sceneDepth, p1)),
                Hasher::quantizeSurfaceNormal(Hasher::surfaceNormal(sceneDepth, p2))
            );

            // Draw only triplets that are matched
            if (!table.templates[key].empty()) {
                cv::circle(result, c, 2, colorRed, -1);
                cv::circle(result, p1, 2, colorRed, -1);
                cv::circle(result, p2, 2, colorRed, -1);
                cv::line(result, c, p1, colorRed);
                cv::line(result, c, p2, colorRed);
                matched++;
            }
        }

        // Labels
        oss.str("");
        oss << "candidates: " << windows[i].candidates.size();
        Visualizer::setLabel(result, oss.str(), windows[i].tl() + cv::Point(info.maxTemplate.width + 5, 10));
        oss.str("");
        oss << "matched: " << matched << "/" << tables.size();
        Visualizer::setLabel(result, oss.str(), windows[i].tl() + cv::Point(info.maxTemplate.width + 5, 28));
        oss.str("");
        oss << "edgels: " << windows[i].edgels;
        Visualizer::setLabel(result, oss.str(), windows[i].tl() + cv::Point(info.maxTemplate.width + 5, 46));

        // Show results
        cv::imshow(title == nullptr ? "Hashing visualization" : title, result);
        if (continuous) {
            cv::waitKey(1);
        }
    }

    cv::waitKey(0);
}

void Visualizer::visualizeTests(Template &tpl, const cv::Mat &sceneHSV, Window &window, std::vector<cv::Point> &stablePoints, std::vector<cv::Point> &edgePoints,
                                cv::Range &neighbourhood, std::vector<int> &scoreII, std::vector<int> &scoreIII, float scoreIV, std::vector<int> &scoreV,
                                int pointsCount, int minThreshold, bool continuous, const char *title) {
    // Init common
    std::ostringstream oss;
    cv::Scalar colorRed(0, 0, 255), colorGreen(0, 255, 0), colorWhite(255, 255, 255), colorBlue(0, 255, 0);
    cv::Point offsetPStart(neighbourhood.start, neighbourhood.start), offsetPEnd(neighbourhood.end, neighbourhood.end);

    // Convert matrices
    cv::Mat result, resultScene;
    cv::cvtColor(tpl.srcHSV, result, CV_HSV2BGR);
    cv::cvtColor(sceneHSV, resultScene, CV_HSV2BGR);
    int scoreVTrue = 0, scoreIIITrue = 0, scoreIITrue = 0, currentTest = 0;

    // Draw from end to enable continuous drawing
    if (!scoreV.empty()) {
        currentTest = 5;
        for (int i = 0; i < pointsCount; i++) {
            const cv::Scalar color = scoreV[i] == 0 ? colorRed : colorGreen;
            cv::rectangle(resultScene, window.tl() + stablePoints[i] + offsetPStart, window.tl() + stablePoints[i] + offsetPEnd, color, 1);
            cv::circle(result, tpl.objBB.tl() + stablePoints[i], 1, color, -1);
        }
    } else if (!scoreIII.empty()) {
        currentTest = 3;
        for (int i = 0; i < pointsCount; i++) {
            const cv::Scalar color = scoreIII[i] == 0 ? colorRed : colorGreen;
            cv::rectangle(resultScene, window.tl() + edgePoints[i] + offsetPStart, window.tl() + edgePoints[i] + offsetPEnd, color, 1);
            cv::circle(result, tpl.objBB.tl() + edgePoints[i], 1, color, -1);
        }
    } else if (!scoreII.empty()) {
        currentTest = 2;
        for (int i = 0; i < pointsCount; i++) {
            const cv::Scalar color = scoreII[i] == 0 ? colorRed : colorGreen;
            cv::rectangle(resultScene, window.tl() + stablePoints[i] + offsetPStart, window.tl() + stablePoints[i] + offsetPEnd, color, 1);
            cv::circle(result, tpl.objBB.tl() + stablePoints[i], 1, color, -1);
        }
    }

    // Count points
    for (int i = 0; i < pointsCount; i++) {
        if (!scoreII.empty()) {
            scoreIITrue += scoreII[i] == 0 ? 0 : 1;
        }

        if (!scoreIII.empty()) {
            scoreIIITrue += scoreIII[i] == 0 ? 0 : 1;
        }

        if (!scoreV.empty()) {
            scoreVTrue += scoreV[i] == 0 ? 0 : 1;
        }
    }

    // Draw window rect and info labels
    cv::rectangle(resultScene, window.tl(), window.tl() + cv::Point(tpl.objBB.width, tpl.objBB.height), colorWhite, 1);
    oss.str("");
    oss << "II: " << scoreIITrue << "/" << pointsCount;
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 10), 1, 0, 0.4,
                         currentTest >= 2 ? (scoreIITrue > minThreshold ? colorGreen : colorRed) : colorWhite);
    oss.str("");
    oss << "III: " << scoreIIITrue << "/" << pointsCount;
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 28), 1, 0, 0.4,
                         currentTest >= 3 ? (scoreIIITrue > minThreshold ? colorGreen : colorRed) : colorWhite);
    oss.str("");
    oss << "IV: " << scoreIV;
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 46), 1, 0, 0.4,
                         currentTest >= 4 ? (scoreVTrue > minThreshold ? colorGreen : colorRed) : colorWhite);
    oss.str("");
    oss << "V: " << scoreVTrue << "/" << pointsCount;
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 64), 1, 0, 0.4,
                         currentTest >= 5 ? (scoreVTrue > minThreshold ? colorGreen : colorRed) : colorWhite);
    oss.str("");
    oss.precision(2);
    oss << "score: " << std::fixed << scoreIITrue / static_cast<float>(pointsCount) + scoreIIITrue / static_cast<float>(pointsCount) +
                         scoreVTrue / static_cast<float>(pointsCount) + scoreIV / pointsCount;
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 82), 1, 0, 0.4,
                         currentTest >= 5 ? (scoreVTrue > minThreshold ? colorGreen : colorRed) : colorWhite);

    // Form title
    oss.str("");
    oss << title;
    oss << " - scene";

    // Show results
    cv::imshow(title == nullptr ? "Hashing visualization" : oss.str(), result);
    cv::imshow(title == nullptr ? "Hashing visualization scene" : title, resultScene);

    // Continuous till match (scoreV > than min threshold)
    if (continuous) {
        cv::waitKey(scoreVTrue > minThreshold ? 0 : 1);
    } else {
        cv::waitKey(0);
    }
}
