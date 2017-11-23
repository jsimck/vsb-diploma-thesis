#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "visualizer.h"
#include "../objdetect/hasher.h"
#include "../processing/processing.h"
#include "../core/classifier_criteria.h"

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
    cv::Size text = cv::getTextSize(label, fontFace, scale, thickness, nullptr);
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

void Visualizer::visualizeWindows(cv::Mat &scene, std::vector<Window> &windows, bool continuous, int wait, const char *title) {
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
    cv::waitKey(wait);
}

void Visualizer::visualizeMatches(cv::Mat &scene, std::vector<Match> &matches, const std::string &templatesPath, int wait, const char *title) {
    cv::Mat viz = scene.clone();
    std::ostringstream oss;
    int tplCounter = 0;

    for (auto &match : matches) {
        cv::rectangle(viz, match.objBB.tl(), match.objBB.br(), cv::Scalar(0, 255, 0));

        oss.str("");
        oss << "id: " << match.tpl->id;
        setLabel(viz, oss.str(), cv::Point(match.objBB.br().x + 5, match.objBB.tl().y + 10));
        oss.str("");
        oss.precision(2);
        oss << std::fixed << "score: " << match.score << " (" << (match.score * 100.0f) / 4.0f << "%)";
        setLabel(viz, oss.str(), cv::Point(match.objBB.br().x + 5, match.objBB.tl().y + 28));

        // Load matched template
        cv::Mat tplSrc = Template::loadSrc(templatesPath, *match.tpl, CV_LOAD_IMAGE_COLOR);

        // draw label and bounding box
        cv::rectangle(tplSrc, match.tpl->objBB.tl(), match.tpl->objBB.br(), cv::Scalar(0, 255, 0));
        oss.str("");
        oss << "id: " << match.tpl->id;
        setLabel(tplSrc, oss.str(), cv::Point(match.tpl->objBB.br().x + 5, match.tpl->objBB.tl().y + 10));

        oss.str("");
        oss << "Template: " << tplCounter;
        std::string winName = oss.str();

        // Show in resizable window
        cv::namedWindow(winName, 0);
        cv::imshow(winName, tplSrc);
        tplCounter++;
    }

    // Show in resizable window
    std::string winName = title != nullptr ? title : "Matched results";
    cv::namedWindow(winName, 0);
    cv::imshow(winName, viz);
    cv::waitKey(wait);
}

void Visualizer::visualizeTemplate(Template &tpl, const std::string &templatesPath, int wait, const char *title) {
    cv::Mat result;

    // Dynamically load template
    if (tpl.srcHSV.empty()) {
        result = Template::loadSrc(templatesPath, tpl, CV_LOAD_IMAGE_COLOR);
    } else {
        result = tpl.srcHSV.clone();
        cv::cvtColor(tpl.srcHSV, result, CV_HSV2BGR);
    }

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
    oss << "mode: " << tpl.mode;
    setLabel(result, oss.str(), tpl.objBB.tl() + cv::Point(tpl.objBB.width + 5, 28));
    oss.str("");
    oss << "elev: " << tpl.elev;
    setLabel(result, oss.str(), tpl.objBB.tl() + cv::Point(tpl.objBB.width + 5, 46));
    oss.str("");
    oss << "srcGradients: " << tpl.features.gradients.size();
    setLabel(result, oss.str(), tpl.objBB.tl() + cv::Point(tpl.objBB.width + 5, 64));
    oss.str("");
    oss << "srcNormals: " << tpl.features.normals.size();
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
    cv::waitKey(wait);
}

void Visualizer::visualizeHashing(cv::Mat &scene, cv::Mat &sceneDepth, std::vector<HashTable> &tables, std::vector<Window> &windows,
                                  ClassifierCriteria &criteria, bool continuous, int wait, const char *title) {
    // Init common
    cv::Scalar colorRed(0, 0, 255);
    std::ostringstream oss;

    // TODO user proper fx and fy
    // Init surface srcNormals
    cv::Mat sceneSurfaceNormals;
    Processing::quantizedNormals(sceneDepth, sceneSurfaceNormals, 1150, 1150, criteria.info.maxDepth, criteria.maxDepthDiff);

    for (size_t i = 0, windowsSize = windows.size(); i < windowsSize; ++i) {
        cv::Mat result = scene.clone();
        int matched = 0;

        // Draw processed windows
        for (size_t j = 0; j < i; j++) {
            cv::rectangle(result, windows[j].tl(), windows[j].br(), cv::Scalar::all(90));
        }

        // Draw window and searched box rectangles
        cv::rectangle(result, windows[i].tl(), windows[i].tl() + cv::Point(criteria.info.largestTemplate), cv::Scalar::all(255));
        cv::rectangle(result, windows[i].tl(), windows[i].br(), cv::Scalar(0, 255, 0));

        for (auto &table : tables) {
            // Prepare train to load hash key
            TripletParams params(criteria.info.largestTemplate.width, criteria.info.largestTemplate.height,
                                 criteria.tripletGrid, windows[i].tl().x, windows[i].tl().y);
            cv::Point c = table.triplet.getCenter(params);
            cv::Point p1 = table.triplet.getP1(params);
            cv::Point p2 = table.triplet.getP2(params);

            // If any point of triplet is out of scene boundaries, ignore it to not get false data
            if ((c.x < 0 || c.x >= sceneSurfaceNormals.cols || c.y < 0 || c.y >= sceneSurfaceNormals.rows) ||
                (p1.x < 0 || p1.x >= sceneSurfaceNormals.cols || p1.y < 0 || p1.y >= sceneSurfaceNormals.rows) ||
                (p2.x < 0 || p2.x >= sceneSurfaceNormals.cols || p2.y < 0 || p2.y >= sceneSurfaceNormals.rows)) continue;

            // Relative depths
            int d[2];
            Processing::relativeDepths(sceneDepth, c, p1, p2, d);

            // Generate hash key
            HashKey key(
                Processing::quantizeDepth(d[0], table.binRanges),
                Processing::quantizeDepth(d[1], table.binRanges),
                sceneSurfaceNormals.at<uchar>(c),
                sceneSurfaceNormals.at<uchar>(p1),
                sceneSurfaceNormals.at<uchar>(p2)
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
        Visualizer::setLabel(result, oss.str(), windows[i].tl() + cv::Point(criteria.info.largestTemplate.width + 5, 10));
        oss.str("");
        oss << "matched: " << matched << "/" << tables.size();
        Visualizer::setLabel(result, oss.str(), windows[i].tl() + cv::Point(criteria.info.largestTemplate.width + 5, 28));
        oss.str("");
        oss << "edgels: " << windows[i].edgels;
        Visualizer::setLabel(result, oss.str(), windows[i].tl() + cv::Point(criteria.info.largestTemplate.width + 5, 46));

        // Show results
        cv::imshow(title == nullptr ? "Hashing visualization" : title, result);
        if (continuous) {
            cv::waitKey(1);
        }
    }

    cv::waitKey(0);
}

bool Visualizer::visualizeTests(Template &tpl, const cv::Mat &sceneHSV, const cv::Mat &sceneDepth, Window &window,
                               std::vector<cv::Point> &stablePoints, std::vector<cv::Point> &edgePoints,
                               cv::Range &neighbourhood, std::vector<int> &scoreI, std::vector<int> &scoreII,
                               std::vector<int> &scoreIII, std::vector<int> &scoreIV, std::vector<int> &scoreV,
                               int pointsCount, int minThreshold, int currentTest, bool continuous,
                               const std::string &templatesPath, int wait, const char *title) {
    // Init common
    std::ostringstream oss;
    cv::Scalar colorRed(0, 0, 255), colorGreen(0, 255, 0), colorWhite(255, 255, 255), colorBlue(0, 255, 0);
    cv::Point offsetPStart(neighbourhood.start, neighbourhood.start), offsetPEnd(neighbourhood.end, neighbourhood.end);

    // Convert matrices
    cv::Mat result, resultScene, resultSceneDepth;
    cv::cvtColor(sceneHSV, resultScene, CV_HSV2BGR);
    int scoreVTrue = 0, scoreIVTrue = 0, scoreIIITrue = 0, scoreIITrue = 0, scoreITrue = 0;

    // Normalize depth scene
    resultSceneDepth = sceneDepth.clone();
    cv::normalize(resultSceneDepth, resultSceneDepth, 0, 1.0f, CV_MINMAX);

    // Dynamically load template src
    if (tpl.srcHSV.empty()) {
        result = Template::loadSrc(templatesPath, tpl, CV_LOAD_IMAGE_COLOR);
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
        cv::rectangle(resultSceneDepth, window.tl() + point + offsetPStart, window.tl() + point + offsetPEnd, bwColor, 1);
        cv::circle(result, tpl.objBB.tl() + point, 1, color, -1);
    }

    // Draw window rects
    cv::rectangle(resultScene, window.tl(), window.tl() + cv::Point(tpl.objBB.width, tpl.objBB.height), colorWhite, 1);
    cv::rectangle(resultSceneDepth, window.tl(), window.tl() + cv::Point(tpl.objBB.width, tpl.objBB.height), cv::Scalar(1.0f), 1);

    // Draw info labels
    oss.str("");
    oss << "I: " << scoreITrue << "/" << pointsCount;
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 10), 1, 0, 0.4,
                         currentTest == 1 ? (scoreITrue > minThreshold ? colorGreen : colorRed) : colorWhite);
    oss.str("");
    oss << "II: " << scoreIITrue << "/" << pointsCount;
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 28), 1, 0, 0.4,
                         currentTest == 2 ? (scoreIITrue > minThreshold ? colorGreen : colorRed) : colorWhite);
    oss.str("");
    oss << "III: " << scoreIIITrue << "/" << pointsCount;
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 46), 1, 0, 0.4,
                         currentTest == 3 ? (scoreIIITrue > minThreshold ? colorGreen : colorRed) : colorWhite);
    oss.str("");
    oss << "IV: " << scoreIVTrue << "/" << pointsCount;
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 64), 1, 0, 0.4,
                         currentTest == 4 ? (scoreVTrue > minThreshold ? colorGreen : colorRed) : colorWhite);
    oss.str("");
    oss << "V: " << scoreVTrue << "/" << pointsCount;
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 82), 1, 0, 0.4,
                         currentTest == 5 ? (scoreVTrue > minThreshold ? colorGreen : colorRed) : colorWhite);
    oss.str("");
    oss.precision(2);
    oss << "score: " << std::fixed << scoreITrue / static_cast<float>(pointsCount) + scoreIITrue / static_cast<float>(pointsCount) +
            scoreIIITrue / static_cast<float>(pointsCount) + scoreIVTrue / static_cast<float>(pointsCount) + scoreVTrue / static_cast<float>(pointsCount);
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(tpl.objBB.width + 5, 100), 1, 0, 0.4,
                         currentTest >= 5 ? (scoreVTrue > minThreshold ? colorGreen : colorRed) : colorWhite);

    // Number of candidates on this window
    oss.str("");
    oss << "candidates: " << window.candidates.size();
    Visualizer::setLabel(resultScene, oss.str(), window.tl() + cv::Point(0, -15), 1, 0, 0.4, colorWhite);

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

    // Continuous till match (scoreV > than min threshold)
    int keyPressed = 0;
    if (continuous) {
        keyPressed = cv::waitKey(scoreVTrue > minThreshold ? 0 : 1);
    } else {
        keyPressed = cv::waitKey(wait);
    }

    // Spacebar pressed
    return keyPressed == 32;
}