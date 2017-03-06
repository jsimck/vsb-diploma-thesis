#include "matching.h"
#include "../utils/utils.h"

#define DEBUG
#define MATCH_NORMED_CROSS_CORRELATION
// #define MATCH_NORMED_CORRELATION_COEF

std::vector<cv::Rect> nonMaximaSuppression(std::vector<cv::Rect> &matchBB) {
    // r = m(a/s) ... s - scale, a - area
}

cv::Scalar matRoiMean(cv::Size maskSize, cv::Rect roi) {
    cv::Mat mask(maskSize, CV_64F, cv::Scalar(0));
    cv::Mat maskRoi(mask, roi);
    maskRoi = cv::Scalar(1.0);

    return cv::mean(maskRoi);
}

std::vector<cv::Rect> matchTemplate(cv::Mat &input, cv::Rect inputRoi, std::vector<Template> &templates) {
    std::vector<cv::Rect> matchBB;

    // Match configuration
    const int step = 4;
    const double minCorrelation = 0.5;

    #pragma omp parallel for schedule(dynamic, 1) shared(input, inputRoi, templates, matchBB)
    for (int i = 0; i < templates.size(); i++) {
        // Calculate final size of inputRoi, if possible we extend it's height by the half height of match template
        // since objectness measure can ignore bottom edges that are not visible on depth maps
        if (inputRoi.br().y + templates[0].bounds.height / 2 < input.rows) {
            inputRoi.height += templates[0].bounds.height / 2;
        }

        // Get area of interest from input image using input ROI rect
        cv::Mat croppedInput(input, inputRoi);

        // Get match template from templates
        cv::Mat tpl = templates[i].src;
        cv::Size wSize = tpl.size();

        // Set default helper variables for matching
        bool matchFound = false;
        double maxScore = minCorrelation;
        cv::Rect maxRect;

#ifdef MATCH_NORMED_CORRELATION_COEF
        // Get template mean
        cv::Scalar meanT = cv::mean(tpl);
#endif

        for (int y = 0; y < croppedInput.rows - wSize.height; y += step) {
            for (int x = 0; x < croppedInput.cols - wSize.width; x += step) {
                double sum = 0, sumNormT = 0, sumNormI = 0;

#ifdef MATCH_NORMED_CORRELATION_COEF
                // Get mean of input image processed area
                cv::Scalar meanI = matRoiMean(croppedInput.size(), cv::Rect(x, y, wSize.width, wSize.height));
#endif

                // Loop through template
                for (int ty = 0; ty < wSize.height; ty++) {
                    for (int tx = 0; tx < wSize.width; tx++) {
                        double Ti = tpl.at<double>(ty, tx);

                        // Ignore black pixels
                        if (Ti == 0) continue;

                        // Get input image value
                        double Ii = croppedInput.at<double>(y + ty, x + tx);

#ifdef MATCH_NORMED_CORRELATION_COEF
                        // Calculate sums for normalized correlation coefficient method
                        sum += ((Ii - meanI.val[0]) * (Ti - meanT.val[0]));
                        sumNormI += SQR(Ii - meanI.val[0]);
                        sumNormT += SQR(Ti - meanT.val[0]);
#endif

#ifdef MATCH_NORMED_CROSS_CORRELATION
                        // Calculate sums for normalized cross correlation method
                        sum += ((Ii) * (Ti));
                        sumNormI += SQR(Ii);
                        sumNormT += SQR(Ti);
#endif
                    }
                }

                // Calculate correlation using NORMED_CROSS_CORRELATION / NORMED_CORRELATION_COEF method
                double crossSum = sum / sqrt(sumNormI * sumNormT);

                // Check if we found new max score, if yes -> save roi location + score
                if (crossSum > maxScore) {
                    maxRect = cv::Rect(x, y, wSize.width, wSize.height);
                    maxScore = crossSum;
                    matchFound = true;
                }
            }
        }

        // If we found match over given threshold, push it into BB array
        if (matchFound) {
#ifdef DEBUG
            // Print score
            std::cout << "Match rect score: " << maxScore << ", area: " << maxRect.area() << ", finalScore: " << maxRect.area() * maxScore << std::endl;
#endif
            // Offset result rectangle by inputROI x and y values
            maxRect.x += inputRoi.x;
            maxRect.y += inputRoi.y;

            // Return best match rectangle position and BB
            matchBB.push_back(maxRect);
        }
    }

    return matchBB;
}