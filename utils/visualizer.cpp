#include <opencv2/opencv.hpp>
#include "visualizer.h"
#include "../core/triplet.h"
#include "utils.h"

void Visualizer::visualizeWindows(cv::Mat &scene, std::vector<Window> &windows, const char *title) {
    cv::Mat locations = scene.clone();

    for (auto window : windows) {
        cv::rectangle(locations, window.tl(), window.br(), cv::Scalar(190, 190, 190));
    }

    std::stringstream ss;
    ss << "Locations: " << windows.size();
    std::cout << windows.size() << std::endl;

    cv::rectangle(locations, windows[0].tl(), windows[0].br(), cv::Scalar(0, 255, 0));
    cv::rectangle(locations, cv::Point(0, 0), cv::Point(160, 30), cv::Scalar(0, 0, 0), -1);
    cv::putText(locations, ss.str(), cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, CV_AA);

    cv::imshow(title == nullptr ? "Window locations detected:" : title, locations);
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
        utils::setLabel(viz, oss.str(), cv::Point(match.objBB.br().x + 5, match.objBB.tl().y + 10));
        oss.str("");
        oss << "score: " << match.score;
        utils::setLabel(viz, oss.str(), cv::Point(match.objBB.br().x + 5, match.objBB.tl().y + 28));

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
