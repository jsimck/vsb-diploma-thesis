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

namespace tless {
    struct SceneGt {
    public:
        cv::Mat R;
        cv::Mat t;
        int objID;
    };

    class FinePose {
    private:
        GLFWwindow *window;
        std::unordered_map<int, Mesh> meshes;
        std::unordered_map<int, Shader> shaders;
        cv::Ptr<ClassifierCriteria> criteria;

        const int SHADER_DEPTH = 0, SHADER_NORMAL = 1;

        void initOpenGL();
        void loadShaders(const std::string &shadersBasePath);
        void loadMeshes(const std::string &meshesListPath);

        void renderPose(const FrameBuffer &fbo, const Mesh &mesh, cv::Mat &depth, cv::Mat &normals,
                        const glm::mat4 &model, const glm::mat4 &view, const glm::mat4 &projection);
    public:
        static const int SCR_WIDTH = 800, SCR_HEIGHT = 600;

        FinePose(cv::Ptr<ClassifierCriteria> criteria, const std::string &shadersBasePath, const std::string &meshesListPath);
        ~FinePose();

        void estimate(std::vector<Match> &matches, const Scene &scene);
    };
}

#endif
