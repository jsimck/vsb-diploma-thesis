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

    // Generate triplet params
    float stepX = t.cols / static_cast<float>(this->featurePointsGrid.width);
    float stepY = t.rows / static_cast<float>(this->featurePointsGrid.height);
    float offsetX = stepX / 2.0f;
    float offsetY = stepY / 2.0f;

    // vizualization of triplets
    Triplet triplets[100];
    for (int i = 0; i < 100; ++i) {
        triplets[i] = Triplet::createRandomTriplet(this->featurePointsGrid);

        cv::Point c = triplets[i].getCenterCoords(offsetX, stepX, offsetY, stepY);
        cv::Point p1 = triplets[i].getP1Coords(offsetX, stepX, offsetY, stepY);
        cv::Point p2 = triplets[i].getP2Coords(offsetX, stepX, offsetY, stepY);
        std::cout << "Triplet coords (" << c.x << ", " << c.y << ") "
                                  << " (" << p1.x << ", " << p1.y << ") "
                                  << " (" << p2.x << ", " << p2.y << ") "<< std::endl;


        cv::circle(t, c, 1, cv::Scalar(1.0), -1);
        cv::circle(t, p1, 1, cv::Scalar(0.65), -1);
        cv::circle(t, p2, 1, cv::Scalar(0.4), -1);

        cv::line(t, c, p1, cv::Scalar(1.0));
        cv::line(t, c, p2, cv::Scalar(1.0));

        std::cout << "Triplet " << i << ": " << triplets[i] << std::endl;
    }

    cv::imshow("Train hashing, test template", t);
    cv::waitKey(0);
}

const cv::Size Hashing::getFeaturePointsGrid() {
    return this->featurePointsGrid;
}

const std::vector<HashTable> &Hashing::getHashTables() {
    return this->hashTables;
}

void Hashing::setFeaturePointsGrid(cv::Size featurePointsGrid) {
    this->featurePointsGrid = featurePointsGrid;
}

void Hashing::setHashTables(const std::vector<HashTable> &hashTables) {
    this->hashTables = hashTables;
}
