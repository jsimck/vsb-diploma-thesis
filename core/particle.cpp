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
        m = glm::eulerAngleXYZ(pose[3], pose[4], pose[5]);

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

        // Save old pose for clamping
        float oldPose[6];
        std::memcpy(oldPose, pose, sizeof pose);

        // Update current possition with new velocity
        for (int i = 0; i < 6; i++) {
            pose[i] = v[i] + pose[i];

            // Clamped poses within defined bound
            if (i > 2 && (pose[i] > 0.6f || pose[i] < -0.6f)) {
                pose[i] = oldPose[i]; // Rotation
            } else if (i == 2 && (pose[i] < -150 || pose[i] > 250)) {
                pose[i] = oldPose[i]; // Z-translation
            } else if (i < 2 && (pose[i] < -40 || pose[i] > 40)) {
                pose[i] = oldPose[i]; // XY-translation
            }
        }
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