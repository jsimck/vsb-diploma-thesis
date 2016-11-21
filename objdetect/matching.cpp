#include "matching.h"

void matchTemplate(cv::Mat &input, std::vector<Template> &templates, std::vector<cv::Rect> &matchBB) {
    // Convert input to grayScale
    cv::Mat input_32fc1;
    cv::cvtColor(input, input_32fc1, CV_BGR2GRAY);
    input_32fc1.convertTo(input_32fc1, CV_32FC1, 1.0/255.0);

    // Makes black pixels that didn't pass through objdetect irrelevant
    cv::Mat result = cv::Mat::ones(input.rows, input.cols, CV_32FC1) * 1000;

    int step = 4;
    float sum = 0, Ti = 0, Ii = 0;
    for (int i = 0; i < templates.size(); i++) {
        cv::Mat tpl = templates[i].src;
        cv::Size wSize = templates[i].src.size();

        for (int y = 0; y < input_32fc1.rows - wSize.height; y += step) {
            for (int x = 0; x < input_32fc1.cols - wSize.width; x += step) {
                sum = 0;

                // Loop through template
                for (int ty = 0; ty < wSize.height; ty++) {
                    for (int tx = 0; tx < wSize.width; tx++) {
                        Ti = tpl.at<float>(ty, tx);

                        // Ignore black pixels
                        if (Ti == 0) continue;

                        // Calculate correlation CV_TM_SQDIFF
                        Ii = input_32fc1.at<float>(y + ty, x + tx);
                        sum += SQR(Ii - Ti);
                    }
                }

                result.at<float>(y, x) = sum;
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