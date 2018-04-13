#ifndef VSB_SEMESTRAL_PROJECT_PARTICLE_H
#define VSB_SEMESTRAL_PROJECT_PARTICLE_H

#include <glm/glm.hpp>
#include <ostream>
#include <opencv2/core/mat.hpp>

namespace tless {
    class Particle {
    private:
        inline float velocity(float w, float v, float x, float pBest, float gBest, float c1, float c2, float r1, float r2);
        inline float nextR();

    public:
        float fitness = 0;

        // Velocity vector
        union {
            struct {
                float v1, v2, v3, v4, v5, v6;
            };
            float v[6];
        };

        // 6D pose
        union {
            struct {
                float tx, ty, tz, rx, ry, rz;
            };
            float pose[6];
        };

        struct {
            union {
                struct {
                    float v1, v2, v3, v4, v5, v6;
                };
                float v[6];
            };

            union {
                struct {
                    float tx, ty, tz, rx, ry, rz;
                };
                float pose[6];
            };

            float fitness = 0;
        } pBest;

        Particle() = default;
        Particle(float tx, float ty, float tz, float rx, float ry, float rz,
                 float v1, float v2, float v3, float v4, float v5, float v6);

        glm::mat4 model();

        void updatePBest();

        void progress(float w1, float w2, float c1, float c2, const Particle &gBest);

        static float objFun(const cv::Mat &gt, const cv::Mat &gtNormals, const cv::Mat &gtEdges, const cv::Mat &pose, const cv::Mat &poseNormals);

        friend std::ostream &operator<<(std::ostream &os, const Particle &particle);
    };
}

#endif
