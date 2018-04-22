#include <glm/ext.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/ximgproc.hpp>
#include <gsl/gsl_qrng.h>
#include "fine_pose.h"
#include "../utils/parser.h"
#include "../utils/glutils.h"
#include "../core/particle.h"
#include "../processing/processing.h"
#include "../utils/timer.h"
#include "../core/classifier_criteria.h"

namespace tless {
    const int FinePose::SHADER_DEPTH = 0;
    const int FinePose::SHADER_NORMAL = 1;
    const int FinePose::SHADER_DEPTH_NORMAL = 2;
    const int FinePose::SCR_WIDTH = 720;
    const int FinePose::SCR_HEIGHT = 540;

    void FinePose::initOpenGL() {
        // GLFW init and config
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

        // GLFW window creation
        window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "DrawDepth", NULL, NULL);

        if (window == NULL) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window);
        glfwHideWindow(window);
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

        // Init Glew after GLFW init
        if (glewInit()) {
            std::cerr << "Failed to initialize GLXW" << std::endl;
            return;
        }

        // Init Opengl global settings
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
    }

    void FinePose::loadShaders(const std::string &shadersFolder) {
        shaders[SHADER_DEPTH] = Shader(shadersFolder + "depth.vert", shadersFolder + "depth.frag");
        shaders[SHADER_NORMAL] = Shader(shadersFolder + "normal.vert", shadersFolder + "normal.frag");
        shaders[SHADER_DEPTH_NORMAL] = Shader(shadersFolder + "depth_normal.vert", shadersFolder + "depth_normal.frag");
    }

    void FinePose::loadMeshes(const std::string &modelsFolder, const std::string &modelsFileFormat, const std::vector<int> &objIds) {
        for (auto &objId : objIds) {
            std::string path = cv::format((modelsFolder + modelsFileFormat).c_str(), objId);
            meshes[objId] = Mesh(path);
        }
    }

    FinePose::FinePose(cv::Ptr<ClassifierCriteria> criteria, const std::string &shadersFolder, const std::string &meshesListPath,
                           const std::string &modelsFileFormat, const std::vector<int> &objIds) : criteria(criteria) {
        // First initialize OpenGL
        initOpenGL();

        // Load shaders and meshes
        loadShaders(shadersFolder);
        loadMeshes(meshesListPath, modelsFileFormat, objIds);
    }

    FinePose::~FinePose() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void FinePose::vizualize(const Particle &p, cv::Mat &dst, const ScenePyramid &pyr, const FrameBuffer &fbo, const cv::Rect &matchBB, const Mesh &mesh,
                             const glm::mat4 &view, const glm::mat4 &viewProjection, int wait, const char *title) {
        // Init common
        cv::Mat pDepth, pNormals;
        glm::mat4 model = p.model();

        if (dst.empty()) {
            dst = pyr.srcRGB.clone();
        }

        // Render pose and calculate fitness
        renderPose(fbo, mesh, pDepth, pNormals, mvMat(model, view), mvpMat(model, viewProjection), 0);

        // Render gbest to vizualization
        for (int y = 0; y < pNormals.rows; y++) {
            for (int x = 0; x < pNormals.cols; x++) {
                auto px = pNormals.at<cv::Vec3f>(y, x);

                // Draw normals over rgbd image
                if (px[0] > 0 || px[1] > 0 || px[2] > 0) {
                    dst.at<cv::Vec3b>(y + matchBB.tl().y, x + matchBB.tl().x) = cv::Vec3b(px[0] * 128 + 128, px[1] * 128 + 128, px[2] * 128 + 128);
                }
            }
        }

        // Show results
        cv::imshow(title == nullptr ? "Particle progress vizualization" : title, dst);
        cv::waitKey(wait);
    }

    void FinePose::renderPose(const FrameBuffer &fbo, const Mesh &mesh, cv::Mat &depth, cv::Mat &normals, const glm::mat4 &modelView,
                                  const glm::mat4 &modelViewProjection, float scale) {
        // Bind frame buffer
        fbo.bind();

        /// NORMALS
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate shader and set uniforms
        shaders[SHADER_DEPTH_NORMAL].use();
        shaders[SHADER_DEPTH_NORMAL].setMat4("NMatrix", nMat(modelView));
        shaders[SHADER_DEPTH_NORMAL].setMat4("MVMatrix", modelView);
        shaders[SHADER_DEPTH_NORMAL].setMat4("MVPMatrix", modelViewProjection);
        shaders[SHADER_DEPTH_NORMAL].setFloat("scale", scale);

        // Draw mesh
        mesh.draw();

        // Read data from frame buffer
        cv::Mat result = cv::Mat::zeros(fbo.height, fbo.width, CV_32FC4);
        glReadPixels(0, 0, fbo.width, fbo.height, GL_BGRA, GL_FLOAT, result.data);

        // Unbind frame buffer
        fbo.unbind();

        // Convert to normals and depth
        cv::Mat channels[4];
        cv::split(result, channels);

        // Copy separate channels into own matrices
        cv::merge(channels, 3, normals);
        cv::merge(channels + 3, 1, depth);
        depth.convertTo(depth, CV_16U);
    }

    void FinePose::estimate(std::vector<Match> &matches, const ScenePyramid &pyr) {
        // Init common
        Particle gBest;
        std::vector<Particle> particles;
        cv::Mat result, pDepth, pNormals;
        glm::mat4 MMatrix;

        // Loop through mateches
        for (auto &match : matches) {
            cv::Rect matchBB;
            cv::Mat normals, edges, depth;
            cv::Mat K = match.t->camera.K.clone();

            // Prepare structures for current match
            prepareMatch(pyr, match, matchBB, K, depth, normals, edges, 15);

            // Show cropped part of the scene
            cv::imshow("normals", normals);
            cv::imshow("edges", edges);
            cv::waitKey(1);

            // Create FBO and prepare matrices
            FrameBuffer fbo(matchBB.width, matchBB.height);
            glm::mat4 VMatrix = vMat(match.t->camera.R, match.t->camera.t);
            glm::mat4 PMatrix = pMat(K, 0, 0, matchBB.width, matchBB.height);
            glm::mat4 VPMatrix = vpMat(VMatrix, PMatrix);

            // Init particles
            generatePopulation(particles, criteria->popSize);

            // Init global best
            gBest.fitness = 0;

            // Generate initial particle positions and save gBest
            for (int i = 0; i < particles.size(); ++i) {
                // Render depth image
                MMatrix = particles[i].model();
                renderPose(fbo, meshes[match.t->objId], pDepth, pNormals, mvMat(MMatrix, VMatrix),
                           mvpMat(MMatrix, VPMatrix), match.scale);

//                cv::imshow("Initial pose normal", pNormals);
//                cv::waitKey(1);

                // Compute fitness for new particle
                particles[i].fitness = Particle::objFun(depth, normals, edges, pDepth, pNormals);

                // Save gBest
                if (particles[i].fitness < gBest.fitness) {
                    gBest = particles[i];
                    cv::Mat vizGBest;
                    vizualize(gBest, vizGBest, pyr, fbo, matchBB, meshes[match.t->objId], VMatrix, VPMatrix, 1);
                }
            }

            // Run PSO
            for (int i = 0; i < criteria->generations; i++) {
                std::cout << "Iteration: " << i << std::endl;

                for (auto &p : particles) {
                    // Progress (updates velocity and moves particle)
                    p.progress(criteria->w1, criteria->w2, criteria->c1, criteria->c2, gBest);

                    // Fitness
                    MMatrix = p.model();
                    renderPose(fbo, meshes[match.t->objId], pDepth, pNormals, mvMat(MMatrix, VMatrix),
                               mvpMat(MMatrix, VPMatrix), match.scale);
                    p.fitness = Particle::objFun(depth, normals, edges, pDepth, pNormals);

                    // Check for pBest
                    if (p.fitness < p.pBest.fitness) {
                        p.updatePBest();
                    }

                    // Check for gBest
                    if (p.fitness < gBest.fitness) {
                        std::cout << gBest.fitness << std::endl;
                        gBest = p;

                        // Vizualize gbest
                        cv::Mat vizGBest;
                        vizualize(gBest, vizGBest, pyr, fbo, matchBB, meshes[match.t->objId], VMatrix, VPMatrix, 1);
                        cv::imshow("Gbest pose", pNormals);
                    }

//                    cv::imshow("Pose normals", pNormals);
//                    cv::waitKey(1);
                }
            }

            // Vizualize results
            vizualize(gBest, result, pyr, fbo, matchBB, meshes[match.t->objId], VMatrix, VPMatrix, 1);
        }
        cv::waitKey(0);
    }

    void FinePose::generatePopulation(std::vector<Particle> &particles, int N) {
        // Cleanup
        particles.clear();

        // Init sobol sequence for 6 dimensions
        gsl_qrng *q = gsl_qrng_alloc(gsl_qrng_sobol, 6);
        particles.reserve(N);

        // Random for velocity vectors
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> d(0, 1);

        for (int i = 0; i < N; i++) {
            // Generate sobol sequence
            double v[6];
            gsl_qrng_get(q, v);

            // Generate particle
            particles.emplace_back(Particle(
                (v[0] - 0.5) * 50,
                (v[1] - 0.5) * 50,
                (v[2] - 0.8) * 200,
                (v[3] - 0.5),
                (v[4] - 0.5),
                (v[5] - 0.5),
                d(gen) * 20,
                d(gen) * 20,
                d(gen) * 40,
                d(gen) * 0.2f,
                d(gen) * 0.2f,
                d(gen) * 0.2f
            ));
        }

        gsl_qrng_free(q);
    }

    void FinePose::prepareMatch(const ScenePyramid &pyr, const Match &match, cv::Rect &inflatedBB, cv::Mat &K,
                                cv::Mat &depth, cv::Mat &normals, cv::Mat &edges, int inflateOffset) {
        // Inflate bounding box and validate
        inflatedBB = cv::Rect(
            match.normObjBB.x - inflateOffset,
            match.normObjBB.y - inflateOffset,
            match.normObjBB.width + (inflateOffset * 2),
            match.normObjBB.height + (inflateOffset * 2)
        );
        inflatedBB.x = inflatedBB.x < 0 ? 0 : inflatedBB.x;
        inflatedBB.y = inflatedBB.y < 0 ? 0 : inflatedBB.y;

        // Rescale K
        rescaleK(K, match.objBB.size(), inflatedBB.size());

        // TODO - fix to depth edgels
        cv::Canny(pyr.srcGray, edges, 80, 120);

        // Crop
        depth = pyr.srcDepth(inflatedBB);
        normals = pyr.srcNormals3D(inflatedBB);
        edges = edges(inflatedBB);
    }
}