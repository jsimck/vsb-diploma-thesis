#include "particle.h"
#include "../processing/processing.h"
#include <glm/ext.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

namespace tless {
    float Particle::velocity(float w, float v, float x, float pBest, float gBest, float c1, float c2, float r1, float r2) {
        return w * v + (c1 * r1) * (pBest - x) + (c2 * r2) * (gBest - x);
    }

    Particle::Particle(float tx, float ty, float tz, float rx, float ry, float rz,
                       float v1, float v2, float v3, float v4, float v5, float v6) {
        this->pose[0] = tx;
        this->pose[1] = ty;
        this->pose[2] = tz;
        this->pose[3] = rx;
        this->pose[4] = ry;
        this->pose[5] = rz;
        this->v[0] = v1;
        this->v[1] = v2;
        this->v[2] = v3;
        this->v[3] = v4;
        this->v[4] = v5;
        this->v[5] = v6;

        // Init pBest
        updatePBest();
    };

    glm::mat4 Particle::model()const {
        glm::mat4 m;
        glm::vec3 t(pose[0], pose[1], pose[2]);

        // Rotate
        m = glm::rotate(m, pose[3], glm::vec3(1, 0, 0));
        m = glm::rotate(m, pose[4], glm::vec3(0, 1, 0));
        m = glm::rotate(m, pose[5], glm::vec3(0, 0, 1));

        // Translate
        return glm::translate(m, t);
    }

    void Particle::updatePBest() {
        std::memcpy(pBest.v, v, sizeof v);
        std::memcpy(pBest.pose, pose, sizeof pose);
        pBest.fitness = fitness;
    }

    void Particle::progress(float w1, float w2, float c1, float c2, const Particle &gBest) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> d(0, 1.0f);

        // Calculate new velocity for translations
        for (int i = 0; i < 3; i++) {
            v[i] = velocity(w1, v[i], pose[i], pBest.pose[i], gBest.pose[i], c1, c2, d(gen), d(gen));
        }

        // Use different w coeff for euler angles
        for (int i = 3; i < 6; i++) {
            v[i] = velocity(w2, v[i], pose[i], pBest.pose[i], gBest.pose[i], c1, c2, d(gen), d(gen));
        }

        // Update current possition with new velocity
        for (int i = 0; i < 6; i++) {
            pose[i] = v[i] + pose[i];
        }
    }

    // TODO cleanup, improve
    float Particle::objFun(const cv::Mat &srcDepth, const cv::Mat &srcNormals, const cv::Mat &srcEdges,
                           const cv::Mat &poseDepth, const cv::Mat &poseNormals) {
        float sumD = 0, sumU = 0, sumE = 0;
        const float tD = 500;
        const float inf = std::numeric_limits<float>::max();

        // Compute edges
        cv::Mat poseT, poseEdges;
        cv::Laplacian(poseDepth, poseEdges, -1);
        cv::threshold(poseEdges, poseEdges, 20, 255, CV_THRESH_BINARY_INV);
        poseEdges.convertTo(poseEdges, CV_8U);
        cv::distanceTransform(poseEdges, poseT, CV_DIST_L2, 3);

        cv::Mat matD = cv::Mat::zeros(srcDepth.size(), CV_32FC1);
        cv::Mat matE = cv::Mat::zeros(srcDepth.size(), CV_32FC1);
        cv::Mat matU = cv::Mat::zeros(srcDepth.size(), CV_32FC1);
        cv::normalize(poseT, poseT, 0, 1, CV_MINMAX);
        cv::imshow("poseEdges", poseEdges);
        cv::imshow("distance", poseT);
        cv::waitKey(1);

        for (int y = 0; y < srcDepth.rows; y++) {
            for (int x = 0; x < srcDepth.cols; x++) {
                // Compute distance transform
                if (srcEdges.at<uchar>(y, x) > 0) {
                    sumE += 1 / (poseT.at<float>(y, x) + 1);
                    matE.at<float>(y, x) = 1 / (poseT.at<float>(y, x) + 1);
                }

                // Skip invalid depth pixels for other tests pixels
                if (poseDepth.at<ushort>(y, x) <= 0) {
                    continue;
                }

                // Compute depth diff
                int dDiff = std::abs(srcDepth.at<ushort>(y, x) - poseDepth.at<ushort>(y, x));
                sumD += (dDiff > tD) ? (1 / (inf + 1)) : (1 / (dDiff + 1));
                matD.at<float>(y, x) = (dDiff > tD) ? (1 / (inf + 1)) : (1 / (dDiff + 1));

                // Compare normals
                float dot = std::abs(srcNormals.at<cv::Vec3f>(y, x).dot(poseNormals.at<cv::Vec3f>(y, x)));
                sumU += std::isnan(dot) ? (1 / (inf + 1)) : (1 / (dot + 1));
                matU.at<float>(y, x) = std::isnan(dot) ? (1 / (9999999 + 1)) : (1 / (dot + 1));
            }
        }

        cv::normalize(matU, matU, 0, 1, CV_MINMAX);
        cv::normalize(matE, matE, 0, 1, CV_MINMAX);
        cv::normalize(matD, matD, 0, 1, CV_MINMAX);

        cv::imshow("matU", matU);
        cv::imshow("matE", matE);
        cv::imshow("matD", matD);
        cv::waitKey(1);

        return -sumD * sumU * sumE;
    }

    std::ostream &operator<<(std::ostream &os, const Particle &particle) {
        os << "fitness: " << particle.fitness << ", pbest: " << particle.pBest.fitness << ", pose: ";
        for (int i = 0; i < 6; ++i) {
            os << i << ": " << particle.pose[i] << ",";
        }
        os << std::endl << "velocity: ";
        for (int i = 0; i < 6; ++i) {
            os << i << ": " << particle.v[i] << ",";
        }

        return os;
    }
}