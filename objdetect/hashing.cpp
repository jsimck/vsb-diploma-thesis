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

void Hashing::train(std::vector<TemplateGroup> &groups) {
    // Testing, get first template
    auto t = groups[0].templates[0].srcDepth;

    cv::imshow("Train hashing, test template", t);
    cv::waitKey(0);
}
