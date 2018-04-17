#ifndef VSB_SEMESTRAL_PROJECT_PARTICLE_H
#define VSB_SEMESTRAL_PROJECT_PARTICLE_H

#include <glm/glm.hpp>
#include <ostream>
#include <opencv2/core/mat.hpp>

namespace tless {
    class Particle {
    private:
        /**
         * @brief Updates velocity vector according to equation defined in fine-pose estimation.
         *
         * @param[in] w     Tunable param (reduces velocity over time)
         * @param[in] v     Current velocity
         * @param[in] x     Current position (pose component)
         * @param[in] pBest Current personal best
         * @param[in] gBest Current global best
         * @param[in] c1    Learning factor [0 - 2] that affects how much is the particle leading towards personal best
         * @param[in] c2    Learning factor [0 - 2] that affects how much is the particle leading towards global best
         * @param[in] r1    Random number [0 - 1]
         * @param[in] r2    Random number [0 - 1]
         * @return
         */
        inline float velocity(float w, float v, float x, float pBest, float gBest, float c1, float c2, float r1, float r2);

    public:
        float fitness = 0;
        float pose[6];
        float v[6];

        // pBest velocity and 6D pose
        struct {
            float fitness = 0;
            float pose[6];
            float v[6];
        } pBest;

        Particle() = default;
        Particle(float tx, float ty, float tz, float rx, float ry, float rz,
                 float v1, float v2, float v3, float v4, float v5, float v6);

        /**
         * @brief Returns particle model matrix, created from 6D pose params.
         *
         * @return Model matrix consiting of tx, ty, tz (translations) and rx, ry, rz (rotations)
         */
        glm::mat4 model();

        /**
         * @brief Updates personal best by copying current 6D properties and fitness value.
         */
        void updatePBest();

        /**
         * @brief Computes new velocity vectors and updates all 6D pose properties by adding new velocity.
         *
         * @param[in] w1    Tunable param [0 - 1.2] amount which decreases final velocity at each new calculation (translation)
         * @param[in] w2    Tunable param [0 - 1.2] amount which decreases final velocity at each new calculation (rotation)
         * @param[in] c1    Learning factor [0 - 2] that affects how much is the particle leading towards personal best
         * @param[in] c2    Learning factor [0 - 2] that affects how much is the particle leading towards global best
         * @param[in] gBest Particle that has the best fitness value currently
         */
        void progress(float w1, float w2, float c1, float c2, const Particle &gBest);

        /**
         * @brief Objective function used to calculate fitness of each particle.
         *
         * @param[in] srcDepth    16-bit depth image of matched bounding box
         * @param[in] srcNormals  32-bit 3D normals of matched bounding box
         * @param[in] srcEdges    Binary image of edges, detected in mached bounding box
         * @param[in] poseDepth   32-bit  depth image of currently processed pose (from OpenGL)
         * @param[in] poseNormals 32-bit 3D normals of currently processed pose (from OpenGL)
         * @return                Fitness value describind amount of matched features between rendered pose and matched bounding box
         */
        static float objFun(const cv::Mat &srcDepth, const cv::Mat &srcNormals, const cv::Mat &srcEdges,
                            const cv::Mat &poseDepth, const cv::Mat &poseNormals);

        friend std::ostream &operator<<(std::ostream &os, const Particle &particle);
    };
}

#endif
