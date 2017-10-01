#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "visualizer.h"
#include "../core/triplet.h"
#include "utils.h"

cv::Vec3b Visualizer::heatMapValue(int min, int max, int value) {
    float range = max - min;
    float percentage = 0;

    if (range) {
        percentage = static_cast<float>(value - min) / range;
    } else {
        return cv::Vec3b(120, 120, 120);
    }

    if (percentage >= 0 && percentage < 0.1f) {
        return cv::Vec3b(120, 120, 120);
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

void Visualizer::visualizeSingleTemplateFeaturePoints(Template &tpl, const char *title) {
    cv::Mat points;
    cv::cvtColor(tpl.srcGray, points, CV_GRAY2BGR);

    for (uint i = 0; i < tpl.edgePoints.size(); ++i) {
        cv::circle(points, tpl.edgePoints[i], 1, cv::Scalar(0, 0, 255), -1);
        cv::circle(points, tpl.stablePoints[i], 1, cv::Scalar(255, 0, 0), -1);
    }

    std::stringstream ss;
    ss << "Template [" << tpl.id << "] feature points";

    cv::imshow(title == nullptr ? ss.str() : title, points);
    cv::waitKey(0);
}

void Visualizer::visualizeTriplets(Template &tpl, HashTable &table, DataSetInfo &info, cv::Size &grid, const char *title) {
    cv::Mat triplets;
    cv::cvtColor(tpl.srcGray, triplets, CV_GRAY2BGR);

    // Grid offset
    cv::Point gridOffset(
            tpl.objBB.tl().x - (info.maxTemplate.width - tpl.objBB.width) / 2,
            tpl.objBB.tl().y - (info.maxTemplate.height - tpl.objBB.height) / 2
    );

    // Get triplet points
    TripletParams coordParams(info.maxTemplate.width, info.maxTemplate.height, grid, tpl.objBB.tl().x, tpl.objBB.tl().y);
    cv::Point c = table.triplet.getCenter(coordParams);
    cv::Point p1 = table.triplet.getP1(coordParams);
    cv::Point p2 = table.triplet.getP2(coordParams);

    // Visualize triplets
    cv::rectangle(triplets, gridOffset, cv::Point(gridOffset.x + info.maxTemplate.width, gridOffset.y + info.maxTemplate.height), cv::Scalar(0, 255, 0));
    cv::rectangle(triplets, tpl.objBB.tl(), tpl.objBB.br(), cv::Scalar(0, 0, 255));

    for (int x = 0; x < 12; x++) {
        for (int y = 0; y < 12; y += 3) {
            Triplet tripletVis(cv::Point(x, y), cv::Point(x, y + 1), cv::Point(x, y + 2));
            cv::circle(triplets, tripletVis.getPoint(0, coordParams), 1, cv::Scalar(0, 100, 0), -1);
            cv::circle(triplets, tripletVis.getPoint(1, coordParams), 1, cv::Scalar(0, 100, 0), -1);
            cv::circle(triplets, tripletVis.getPoint(2, coordParams), 1, cv::Scalar(0, 100, 0), -1);
        }

        cv::circle(triplets, c, 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(triplets, p1, 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(triplets, p2, 2, cv::Scalar(0, 0, 255), -1);
        cv::line(triplets, c, p1, cv::Scalar(0, 0, 255));
        cv::line(triplets, c, p2, cv::Scalar(0, 0, 255));
    }

    std::stringstream ss;
    ss << "Template [" << tpl.id << "] feature triplets";

    cv::imshow(title == nullptr ? ss.str() : title, triplets);
    cv::waitKey(0);
}

void Visualizer::visualizeMatches(cv::Mat &scene, std::vector<Match> &matches, std::vector<Group> &groups) {
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

        for (auto &group : groups) {
            for (auto &tpl : group.templates) {
                if (tpl.id == match.tpl->id) {
                    // Crop template src
                    cv::Mat tplSrc = tpl.srcHSV(tpl.objBB).clone();
                    cv::cvtColor(tplSrc, tplSrc, CV_HSV2BGR);

                    oss.str("");
                    oss << "Template id: " << tpl.id;
                    std::string winName = oss.str();

                    // Show in resizable window
                    cv::namedWindow(winName, 0);
                    cv::imshow(winName, tplSrc);
                }
            }
        }
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

void Visualizer::setLabel(cv::Mat &im, const std::__cxx11::string label, const cv::Point &origin, int padding, int fontFace, double scale,
                          cv::Scalar fColor, cv::Scalar bColor, int thickness) {
    cv::Size text = cv::getTextSize(label, fontFace, scale, thickness, 0);
    rectangle(im, origin + cv::Point(-padding - 1, padding + 2),
                  origin + cv::Point(text.width + padding, -text.height - padding - 2), bColor, CV_FILLED);
    putText(im, label, origin, fontFace, scale, fColor, thickness, CV_AA);
}
