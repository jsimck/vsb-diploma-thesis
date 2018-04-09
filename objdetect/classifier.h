#ifndef VSB_SEMESTRAL_PROJECT_CLASSIFIER_H
#define VSB_SEMESTRAL_PROJECT_CLASSIFIER_H

#include <memory>
#include <unordered_map>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../core/match.h"
#include "../core/hash_table.h"
#include "../utils/parser.h"
#include "hasher.h"
#include "objectness.h"
#include "../core/window.h"
#include "matcher.h"
#include "../core/classifier_criteria.h"
#include "../glcore/shader.h"
#include "../glcore/mesh.h"
#include "../glcore/frame_buffer.h"

namespace tless {
    /**
     * class Classifier
     *
     * Main class which runs all other classifiers and template parsers in the correct order.
     * In this class it's also possible to fine-tune the resulted parameters of each verification stage
     * which can in the end produce different results. These params can be adapted to processed templates
     * and scenes.
     */
    class Classifier {
    private:
        cv::Ptr<ClassifierCriteria> criteria;
        std::vector<Template> templates;
        std::vector<HashTable> tables;
        std::vector<Window> windows;
        std::vector<Match> matches;

        // GL Related stuff
        FrameBuffer fbo;
        std::unordered_map<int, Mesh> meshes;
        std::unordered_map<int, Shader> shaders;
        GLFWwindow *window;

        // Methods
        void load(const std::string &trainedTemplatesListPath, const std::string &trainedPath);

        void initGL();
        void initShaders();
        void initMeshes(const std::string &meshesListPath);

    public:
        const int SHADER_DEPTH = 0, SHADER_NORMAL = 1;

        // Constructors
        explicit Classifier(cv::Ptr<ClassifierCriteria> criteria);
        ~Classifier();

        // Methods
        void train(std::string templatesListPath, std::string resultPath, std::vector<uint> indices = {});
        void detect(std::string trainedTemplatesListPath, std::string trainedPath, std::string scenePath);
        void testPSO();

        float fitness(const cv::Mat &gt, cv::Mat &pose);
        float computeVelocity(float w, float vi, float xi, float pBest, float gBest, float c1, float c2, float r1, float r2);
    };
}

#endif
