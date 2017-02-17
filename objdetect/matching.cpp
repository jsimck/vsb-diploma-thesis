#include "matching.h"
#include "../utils/utils.h"

void matchTemplate(cv::Mat &input, std::vector<Template> &templates, std::vector<cv::Rect> &matchBB) {
    #pragma omp parallel for schedule(dynamic, 1) shared(input, templates, matchBB)
    for (int i = 0; i < templates.size(); i++) {
        double sum = 0, Ti = 0, Ii = 0;
        int step = 4;

        // Makes black pixels that didn't pass through objdetect irrelevant
        cv::Mat result = cv::Mat::ones(input.rows, input.cols, CV_64FC1) * 1000;
        cv::Mat tpl = templates[i].src;
        cv::Size wSize = templates[i].src.size();

        for (int y = 0; y < input.rows - wSize.height; y += step) {
            for (int x = 0; x < input.cols - wSize.width; x += step) {
                sum = 0;

                // Loop through template
                for (int ty = 0; ty < wSize.height; ty++) {
                    for (int tx = 0; tx < wSize.width; tx++) {
                        Ti = tpl.at<double>(ty, tx);

                        // Ignore black pixels
                        if (Ti == 0) continue;

                        // Calculate correlation CV_TM_SQDIFF
                        Ii = input.at<double>(y + ty, x + tx);
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

        // Return best match rectange position and BB
        matchBB.push_back(cv::Rect(minLoc.x, minLoc.y, wSize.width, wSize.height));
    }
}