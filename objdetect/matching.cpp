#include "matching.h"
#include <numeric>
#include "../utils/utils.h"

#define DEBUG
#define MATCH_NORMED_CROSS_CORRELATION
// #define MATCH_NORMED_CORRELATION_COEF

void sortBBByScore(std::vector<cv::Rect> &matchBB, std::vector<double> &scoreBB) {
    // By score, DESC
    for (int i = 0; i < matchBB.size() - 1; i++) {
        for (int j = 0; j < matchBB.size() - i - 1; j++) {
            if (scoreBB[j] < scoreBB[j + 1]) {
                // Save BB to tmp
                double tmpScore = scoreBB[j];
                cv::Rect tmpBB = matchBB[j];

                // Switch bb and scoreBB
                scoreBB[j] = scoreBB[j + 1];
                scoreBB[j + 1] = tmpScore;
                matchBB[j] = matchBB[j + 1];
                matchBB[j + 1] = tmpBB;
            }
        }
    }
}

std::vector<cv::Rect> nonMaximaSuppression(std::vector<cv::Rect> &matchBB, std::vector<double> &scoreBB, double overlapThresh) {
    // Result vector of picked bounding boxes
    std::vector<cv::Rect> pick;

    // Sort BB by score
    sortBBByScore(matchBB, scoreBB);

    // Indexes of bounding boxes to check (length of BB at start) and fill it with index values
    std::vector<int> idx(matchBB.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<int> suppress; // Vector to store indexes which we want to remove at end of each iteration

    // Loop until we check all indexes
    while (!idx.empty()) {
        // Pick first element with highest score (sorted from highest->lowest in previous step)
        int firstIndex = idx.front();
        cv::Rect firstBB = matchBB[firstIndex];

        // Store this index into suppress array, we won't check against this BB again
        suppress.push_back(firstIndex);

        // Push this BB to final array of filtered bounding boxes
        pick.push_back(firstBB);

        // Check overlaps with all other bounding boxes, skipping first (since it is the one we're checking)
        for (int i = 1; i < idx.size(); i++) {
            // Get next index and bounding box in line
            int offsetIndex = *(&idx.front() + i);
            cv::Rect BB = matchBB[offsetIndex];

            // Get overlap BB coordinates
            int ox1 = std::max<int>(BB.tl().x, firstBB.tl().x);
            int ox2 = std::min<int>(BB.br().x, firstBB.br().x);
            int oy1 = std::max<int>(BB.tl().y, firstBB.tl().y);
            int oy2 = std::min<int>(BB.br().y, firstBB.br().y);

            // Calculate overlap area
            int h = std::max<int>(0, oy2 - oy1);
            int w = std::max<int>(0, ox2 - ox1);
            double overlap = static_cast<double>(h * w) / static_cast<double>(firstBB.area());

            // Push index of this window to suppression array, since it is overlapping over minimum threshold
            // with a window of higher score, we can safely ignore this window
            if (overlap > overlapThresh) {
                suppress.push_back(offsetIndex);
            }
        }

        // Remove all suppress indexes from idx array
        idx.erase(std::remove_if(idx.begin(), idx.end(),
            [&suppress, &idx](int v) -> bool {
                return std::find(suppress.begin(), suppress.end(), v) != suppress.end();
            }
        ), idx.end());

        // Clear suppress list
        suppress.clear();
    }

    return pick;
}

cv::Scalar matRoiMean(cv::Size maskSize, cv::Rect roi) {
    cv::Mat mask(maskSize, CV_64F, cv::Scalar(0));
    cv::Mat maskRoi(mask, roi);
    maskRoi = cv::Scalar(1.0);

    return cv::mean(maskRoi);
}

std::vector<cv::Rect> matchTemplate(cv::Mat &input, cv::Rect inputRoi, std::vector<Template> &templates) {
    std::vector<cv::Rect> matchBB;
    std::vector<double> scoreBB;

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

                // Check if we found new max scoreBB, if yes -> save roi location + scoreBB
                if (crossSum > maxScore) {
                    maxRect = cv::Rect(x, y, wSize.width, wSize.height);
                    maxScore = crossSum;
                    matchFound = true;
                }
            }
        }

        // If we found match over given threshold, push it into BB array
        if (matchFound) {
            // Offset result rectangle by inputROI x and y values
            maxRect.x += inputRoi.x;
            maxRect.y += inputRoi.y;

            // Return best match rectangle position and BB
            matchBB.push_back(maxRect);
            scoreBB.push_back(maxScore);

            // Correct way, but not working in my implementation due to use of different matching algorithm
            // scoreBB.push_back(maxScore * maxRect.area());
        }
    }

    // Call non maxima suppression on result BBoxes
    return nonMaximaSuppression(matchBB, scoreBB);
}