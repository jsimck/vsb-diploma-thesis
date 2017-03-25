#include "hashing.h"

cv::Vec3d Hashing::extractSurfaceNormal(cv::Mat &src, cv::Point c) {
    float dzdx = (src.at<float>(c.y, c.x + 1) - src.at<float>(c.y, c.x - 1)) / 2.0f;
    float dzdy = (src.at<float>(c.y + 1, c.x) - src.at<float>(c.y - 1, c.x)) / 2.0f;
    cv::Vec3f d(-dzdy, -dzdx, 1.0f);

    return cv::normalize(d);
}

int Hashing::quantizeSurfaceNormals(cv::Vec3f normal) {
    // For quantization of surface normals into 8 bins
    // in our case z is always positive, that's why we're using
    // 8 octants in top half of sphere only
    cv::Vec3f octantsNormals[8] = {
        cv::Vec3f(1.0f, 0, 1.0f), // 0. octant
        cv::Vec3f(1.0f, 1.0f, 1.0f), // 1. octant
        cv::Vec3f(0, 1.0f, 1.0f), // 2. octant
        cv::Vec3f(-1.0f, 1.0f, 1.0f), // 3. octant
        cv::Vec3f(-1.0f, 0.0f, 1.0f), // 4. octant
        cv::Vec3f(-1.0f, -1.0f, 1.0f), // 5. octant
        cv::Vec3f(0, -1.0f, 1.0f), // 6. octant
        cv::Vec3f(1.0f, -1.0f, 1.0f), // 7. octant
    };

    int minIndex = 0;
    float maxDot = 0, dot;
    for (int i = 0; i < 8; i++) {
        // By doing dot product between octant normals and calculated normal
        // we can find maximum -> index of octant where the vector belongs to
        cv::Vec3f octaNormal = cv::normalize(octantsNormals[i]);
        dot = normal.dot(octaNormal);

        if (dot > maxDot) {
            maxDot = dot;
            minIndex = i;
        }
    }

    return minIndex;
}

void trainingSetGeneration(cv::Mat &train, cv::Mat &trainDepth) {
    cv::Mat normals = cv::Mat::zeros(trainDepth.size(), CV_32FC3);
    Hashing h;


    for (int y = 1; y < trainDepth.rows - 1; y++) {
        for (int x = 1; x < trainDepth.cols - 1; x++) {
            normals.at<cv::Vec3f>(y, x) = h.extractSurfaceNormal(trainDepth, cv::Point(x, y));
            std::cout << h.quantizeSurfaceNormals(normals.at<cv::Vec3f>(y, x)) << " ";
        }
        std::cout << std::endl;
    }

    cv::Mat norm_8uc3 = cv::Mat(normals.size(), CV_8UC3);
    for (int y = 1; y < normals.rows - 1; y++) {
        for (int x = 1; x < normals.cols - 1; x++) {
            uchar b, g, r;
            cv::Vec3f px = normals.at<cv::Vec3f>(y, x);
            b = static_cast<uchar>(px[0] * 255.0f);
            g = static_cast<uchar>(px[1] * 255.0f);
            r = static_cast<uchar>(px[2] * 255.0f);
            norm_8uc3.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
        }
    }

    cv::normalize(normals, normals, 0.0f, 1.0f, CV_MINMAX, CV_32FC3);

    // Show results
    cv::imshow("test train - norm_8uc3", norm_8uc3);
    cv::imshow("test train depth", trainDepth);
    cv::imshow("test train depth - normals", normals);
    cv::waitKey(0);
}
