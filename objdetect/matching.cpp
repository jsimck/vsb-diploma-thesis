#include "matching.h"
#include "../utils/utils.h"

std::vector<cv::Rect> matchTemplate(cv::Mat &input, cv::Rect roi, std::vector<Template> &templates) {
    std::vector<cv::Rect> matchBB;

    #pragma omp parallel for schedule(dynamic, 1) shared(input, templates, matchBB)
    for (int i = 0; i < templates.size(); i++) {
        double sum = 0, Ti = 0, Ii = 0;
        int step = 4;

        // Calculate final size of roi, if possible we extend it's height by the half height of match template
        // since objectness measure can ignore bottom edges that are not visible on depth maps
        cv::Rect roiNormalized = roi;
        int totalRoiHeight = roi.y + roi.height;

        if (totalRoiHeight + templates[0].bounds.height / 2 < input.rows) {
            roiNormalized.height += templates[0].bounds.height / 2;
        }

        // Makes black pixels that didn't pass through objdetect irrelevant
        cv::Mat croppedInput = input(roiNormalized);
        cv::Mat result = cv::Mat::ones(croppedInput.rows, croppedInput.cols, CV_64FC1) * 1000;
        cv::Mat tpl = templates[i].src;
        cv::Size wSize = templates[i].src.size();

        for (int y = 0; y < croppedInput.rows - wSize.height; y += step) {
            for (int x = 0; x < croppedInput.cols - wSize.width; x += step) {
                sum = 0;

                // Loop through template
                for (int ty = 0; ty < wSize.height; ty++) {
                    for (int tx = 0; tx < wSize.width; tx++) {
                        Ti = tpl.at<double>(ty, tx);

                        // Ignore black pixels
                        if (Ti == 0) continue;

                        // Calculate correlation CV_TM_SQDIFF
                        Ii = croppedInput.at<double>(y + ty, x + tx);
                        sum += SQR(Ii - Ti);
                    }
                }

                result.at<double>(y, x) = sum;
            }
        }

        // Locate min value
        double minVal, maxVal;
        cv::Point minLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc);

        // Return best match rectangle position and BB
        matchBB.push_back(cv::Rect(minLoc.x + roiNormalized.x, minLoc.y + roiNormalized.y, wSize.width, wSize.height));
    }

    return matchBB;
}