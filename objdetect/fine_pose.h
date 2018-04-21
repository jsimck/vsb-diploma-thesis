#ifndef VSB_SEMESTRAL_PROJECT_FINE_POSE_H
#define VSB_SEMESTRAL_PROJECT_FINE_POSE_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <unordered_map>
#include "../glcore/frame_buffer.h"
#include "../glcore/shader.h"
#include "../glcore/mesh.h"
#include "../core/match.h"
#include "../core/scene.h"
#include "../core/classifier_criteria.h"
#include "../core/particle.h"

namespace tless {
    class FinePose {
    private:
        GLFWwindow *window;
        cv::Ptr<ClassifierCriteria> criteria;

        void vizualize(const Particle &p, const ScenePyramid &pyr, const FrameBuffer &fbo, const cv::Rect &matchBB, const Mesh &mesh,
                       const glm::mat4 &view, const glm::mat4 &viewProjection, const cv::Mat &depth, const cv::Mat &normals,
                       const cv::Mat &edges, int wait = 0, const char *title = nullptr);

        /**
         * @brief Initializes OpenGL using GLFW library.
         */
        void initOpenGL();

        /**
         * @brief Loads used shaders from the given folder.
         *
         * @param[in] shadersFolder Folder containing all used shaders
         */
        void loadShaders(const std::string &shadersFolder);

        /**
         * @brief Loads modesl in .ply format for objects defined in objIds array.
         *
         * @param[in] modelsFolder     Path to folder containing all objects
         * @param[in] modelsFileFormat File format for filename of each .ply file
         * @param[in] objIds           Array of objIds to load
         */
        void loadMeshes(const std::string &modelsFolder, const std::string &modelsFileFormat, const std::vector<int> &objIds);

        /**
         * @brief Generates uniformly distributed population in 6D space using 6D sobol sequence.
         *
         * @param[out] particles Array of generated particles
         * @param[in]  N         Number of particles to generate
         */
        void generatePopulation(std::vector<Particle> &particles, int N);

        /**
         * @brief Renders normals and depth images for given mesth and copies them from gpu into depth and normals matrices.
         *
         * @param[in]  fbo                 FrameBuffer object initialized to desired size
         * @param[in]  mesh                3D model of the object to render
         * @param[out] depth               32-bit float depth images of rendered object
         * @param[out] normals             32-bit 3-channel normals of rendered object
         * @param[in]  modelView           Object modelView matrix
         * @param[in]  modelViewProjection Object modelViewProjection matrix
         */
        void renderPose(const FrameBuffer &fbo, const Mesh &mesh, cv::Mat &depth, cv::Mat &normals,
                        const glm::mat4 &modelView, const glm::mat4 &modelViewProjection, float scale);

        /**
         * @brief Inflates detected boungin box, rescales K matrix and generates depths, normals, edges from source pyramid.
         *
         * @param[in]     pyr           Input level of image scale pyramid
         * @param[in]     match         Current match detected in detection cascade
         * @param[out]    inflatedBB    Out inflated original boudning box
         * @param[in,out] K             Out camera matrix rescaled to new bounding box
         * @param[out]    depth         Cropped image pyramid depths
         * @param[out]    normals       Cropped image pyramid normals
         * @param[out]    edges         Cropped image pyramid edges
         * @param[in]     inflateOffset Value identifying how much should the rectangle inflate
         */
        void prepareMatch(const ScenePyramid &pyr, const Match &match, cv::Rect &inflatedBB, cv::Mat &K,
                          cv::Mat &depth, cv::Mat &normals, cv::Mat &edges, int inflateOffset = 15);
    public:
        std::unordered_map<int, Mesh> meshes;
        std::unordered_map<int, Shader> shaders;

        static const int SHADER_DEPTH, SHADER_NORMAL, SHADER_DEPTH_NORMAL;
        static const int SCR_WIDTH, SCR_HEIGHT;

        FinePose(cv::Ptr<ClassifierCriteria> criteria, const std::string &shadersFolder, const std::string &meshesListPath,
                         const std::string &modelsFileFormat, const std::vector<int> &objIds);
        ~FinePose();

        /**
         * @brief Applies fine-pose estimation algorithm to given matches. Final refined poses are then returned in matches {R, t} matrices.
         *
         * @param[in,out] matches Array of matched detected during detection process
         * @param[in]     pyr   Current scene that is being classified
         */
        void estimate(std::vector<Match> &matches, const ScenePyramid &pyr);
    };
}

#endif
