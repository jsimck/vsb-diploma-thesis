#include "matching.h"
#include <numeric>
#include "../utils/utils.h"

void sortBBByScore(std::vector<cv::Rect> &matchBB, std::vector<float> &scoreBB) {
    // Checks
    assert(matchBB.size() > 0);
    assert(scoreBB.size() > 0);

    // By score, DESC
    for (int i = 0; i < matchBB.size() - 1; i++) {
        for (int j = 0; j < matchBB.size() - i - 1; j++) {
            if (scoreBB[j] < scoreBB[j + 1]) {
                // Save BB to tmp
                float tmpScore = scoreBB[j];
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

std::vector<cv::Rect> nonMaximaSuppression(std::vector<cv::Rect> &matchBB, std::vector<float> &scoreBB, float overlapThresh) {
    // Checks
    assert(matchBB.size() > 0);
    assert(scoreBB.size() > 0);
    assert(overlapThresh > 0);

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
            float overlap = static_cast<float>(h * w) / static_cast<float>(firstBB.area());

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
    cv::Mat mask(maskSize, CV_32F, cv::Scalar(0));
    cv::Mat maskRoi(mask, roi);
    maskRoi = cv::Scalar(1.0);

    return cv::mean(maskRoi);
}

std::vector<cv::Rect> matchTemplate(const cv::Mat &input, std::vector<Window> &windows) {
    // Checks
    assert(!input.empty());

    // Vector of matched bounding boxes and their score
    std::vector<cv::Rect> matchBB;
    std::vector<float> scoreBB;
    const float minCorrelation = 0.5;

    // Loop through each window
    for (auto &&window : windows) {
        // Skip windows with no candidates
        if (!window.hasCandidates()) {
            continue;
        }

        // Go through all candidates and take one with biggest score ?
        for (int i = 0; i < window.candidatesSize(); i++) {
            // TODO - Asserts
            Template *t = window.candidates[i];
#ifndef NDEBUG
            cv::Mat inputClone = input.clone();
            cv::rectangle(inputClone, window.tl(), cv::Point(window.tl().x + t->src.cols, window.tl().y + t->src.rows), cv::Scalar(1.0f));
            cv::imshow("matching::Template matching with filtered windows and templates (scene)", inputClone);
            cv::imshow("matching::Template matching with filtered windows and templates (template)", t->src);
            cv::waitKey(0);
#endif

            // Set default helper variables for matching
            bool matchFound = false;
            float maxScore = minCorrelation;
            cv::Rect maxRect;

            float sum = 0, sumNormT = 0, sumNormI = 0;

            // Loop through template
            for (int ty = 0; ty < t->src.rows; ty++) {
                for (int tx = 0; tx < t->src.cols; tx++) {
                    float Ti = t->src.at<float>(ty, tx);

                    // Ignore black pixels
                    if (Ti == 0) continue;

                    // Get input image value
                    float Ii = input.at<float>(window.p.y + ty, window.p.x + tx);

                    // Calculate sums for normalized cross correlation method
                    sum += ((Ii) * (Ti));
                    sumNormI += SQR(Ii);
                    sumNormT += SQR(Ti);
                }
            }

            // Calculate correlation using NORMED_CROSS_CORRELATION / NORMED_CORRELATION_COEF method
            float crossSum = sum / sqrt(sumNormI * sumNormT);

            // Check if we found new max scoreBB, if yes -> save roi location + scoreBB
            if (crossSum > maxScore) {
                maxRect = cv::Rect(window.p.x, window.p.y, t->src.cols, t->src.rows);
                maxScore = crossSum;
                matchFound = true;
            }

            // If we found match over given threshold, push it into BB array
            if (matchFound) {
                // Return best match rectangle position and BB
                matchBB.push_back(maxRect);
                scoreBB.push_back(maxScore);

                // Correct way, but not working in my implementation due to use of different matching algorithm
                // scoreBB.push_back(maxScore * maxRect.area());
            }
        }
    }



    return nonMaximaSuppression(matchBB, scoreBB);

    /*
    // Match configuration
    const int step = 5;
    const float minCorrelation = 0.5;

    for (auto &&window : windows) {
        auto templates = window.candidates;

        // Skip windows with no candidates
        if (templates.size() <= 0) {
            continue;
        }

        #pragma omp parallel for schedule(dynamic) shared(input, inputRoi, matchBB)
        for (int i = 0; i < templates.size(); i++) {
            // Get match template from templateFolders
            cv::Mat tpl = templates[i]->src;
            cv::Size wSize = tpl.size();

            // Set default helper variables for matching
            bool matchFound = false;
            float maxScore = minCorrelation;
            cv::Rect maxRect;

            float sum = 0, sumNormT = 0, sumNormI = 0;

            // Loop through template
            for (int ty = 0; ty < wSize.height; ty++) {
                for (int tx = 0; tx < wSize.width; tx++) {
                    float Ti = tpl.at<float>(ty, tx);

                    // Ignore black pixels
                    if (Ti == 0) continue;

                    // Get input image value
                    float Ii = input.at<float>(window.p.y + ty, window.p.x + tx);

                    // Calculate sums for normalized cross correlation method
                    sum += ((Ii) * (Ti));
                    sumNormI += SQR(Ii);
                    sumNormT += SQR(Ti);
                }
            }

            // Calculate correlation using NORMED_CROSS_CORRELATION / NORMED_CORRELATION_COEF method
            float crossSum = sum / sqrt(sumNormI * sumNormT);

            // Check if we found new max scoreBB, if yes -> save roi location + scoreBB
            if (crossSum > maxScore) {
                maxRect = cv::Rect(window.p.x, window.p.y, wSize.width, wSize.height);
                maxScore = crossSum;
                matchFound = true;
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
    }

    // Call non maxima suppression on result BBoxes
    return nonMaximaSuppression(matchBB, scoreBB);
     */
}