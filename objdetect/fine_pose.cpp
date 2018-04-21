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

    void FinePose::renderPose(const FrameBuffer &fbo, const Mesh &mesh, cv::Mat &depth, cv::Mat &normals,
                                  const glm::mat4 &modelView, const glm::mat4 &modelViewProjection) {
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
    }

    void FinePose::estimate(std::vector<Match> &matches, const Scene &scene) {
        // Init scene
        auto &pyr = scene.pyramid[criteria->pyrLvlsDown]; // TODO better handling of scene loading
        cv::Mat sNormals, sEdge, sDepth;

        // Normalize min and max depths to look for objectness in
        auto minMag = static_cast<int>(criteria->objectnessDiameterThreshold * criteria->info.smallestDiameter * criteria->info.depthScaleFactor);
        depthEdgels(pyr.srcDepth, sEdge, criteria->info.minDepth, criteria->info.maxDepth, minMag);
        pyr.srcDepth.convertTo(sDepth, CV_32F);
        cv::Mat vizResult = pyr.srcRGB.clone();

        // Canny test
        cv::Mat canny;
        cv::Canny(pyr.srcGray, canny, 80, 120);

        // Thin edges
//        cv::normalize(sEdge, sEdge, 0, 255, CV_MINMAX);
//        cv::ximgproc::thinning(sEdge, sEdge, cv::ximgproc::THINNING_GUOHALL);

        // Loop through mateches
        for (auto &match : matches) {
            // Enlarge BB
            cv::Rect bb(match.objBB.x - 20, match.objBB.y - 20, match.objBB.width + 40, match.objBB.height + 40);
            cv::Mat edgeCLone = canny.clone();

            // Crop to current bounding box
            cv::Mat normals, edges, depth;
            normals = pyr.srcNormals3D(bb);
            edges = edgeCLone(bb);
            depth = sDepth(bb);

            // Show cropped part of the scene
            cv::imshow("normals", normals);
            cv::imshow("sEdge", edges);
            cv::waitKey(1);

            // Create FBO with given size and update viewport size
            FrameBuffer fbo(bb.width, bb.height);

            // Rescale K
            cv::Mat K = match.t->camera.K.clone();
            rescaleK(K, match.t->objBB.size(), bb.size());

            // Precompute matrices
            glm::mat4 VMatrix = vMat(match.t->camera.R, match.t->camera.t);
            glm::mat4 PMatrix = pMat(K, 0, 0, bb.width, bb.height);
            glm::mat4 VPMatrix = vpMat(VMatrix, PMatrix);

            // Init particles
            glm::mat4 m;
            cv::Mat pose, poseNormals;
            std::vector<Particle> particles;
            generatePopulation(particles, criteria->popSize);

            // Render match for reference
            cv::Mat org, orgNormals;
            renderPose(fbo, meshes[match.t->objId], org, orgNormals, VMatrix, VPMatrix);
            cv::imshow("Found match", orgNormals);

            // Init global best
            Particle gBest;
            gBest.fitness = 0;

            // Generate initial particle positions
            for (int i = 0; i < particles.size(); ++i) {
                // Render depth image
                m = particles[i].model();
                renderPose(fbo, meshes[match.t->objId], pose, poseNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));

//                cv::imshow("Pose", poseNormals);
//                cv::waitKey(0);

                // Compute fitness for new particle
                particles[i].fitness = Particle::objFun(depth, normals, edges, pose, poseNormals);

                // Save gBest
                if (particles[i].fitness < gBest.fitness) {
                    gBest = particles[i];
                }
            }

            // PSO
            cv::Mat imGBest, imGBestNormals;
            m = gBest.model();
            renderPose(fbo, meshes[match.t->objId], imGBest, imGBestNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));

            // Generations
            for (int i = 0; i < criteria->generations; i++) {
                std::cout << "Iteration: " << i << std::endl;

                for (auto &p : particles) {
                    // Progress (updates velocity and moves particle)
                    p.progress(criteria->w1, criteria->w2, criteria->c1, criteria->c2, gBest);

                    // Fitness
                    m = p.model();
                    renderPose(fbo, meshes[match.t->objId], pose, poseNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));
                    p.fitness = Particle::objFun(depth, normals, edges, pose, poseNormals);

                    // Check for pBest
                    if (p.fitness < p.pBest.fitness) {
                        p.updatePBest();
                    }

                    // Check for gBest
                    if (p.fitness < gBest.fitness) {
                        gBest = p;

                        // Vizualization
                        m = gBest.model();
                        renderPose(fbo, meshes[match.t->objId], imGBest, imGBestNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));
                        Particle::objFun(depth, normals, edges, imGBest, imGBestNormals);

                        // Render gbest to vizualization
                        cv::Mat curVizResult = pyr.srcRGB.clone();
                        for (int y = 0; y < imGBestNormals.rows; y++) {
                            for (int x = 0; x < imGBestNormals.cols; x++) {
                                auto px = imGBestNormals.at<cv::Vec3f>(y, x);

                                if (px[0] > 0 || px[1] > 0 || px[2] > 0) {
                                    curVizResult.at<cv::Vec3b>(y + bb.tl().y, x + bb.tl().x) = cv::Vec3b(px[0] * 128 + 128, px[1] * 128 + 128, px[2] * 128 + 128);
                                }
                            }
                        }

                        cv::imshow("vizResult", curVizResult);
                        cv::waitKey(1);
                    }

//                    cv::imshow("pose 2", poseNormals);
//                    cv::waitKey(1);
                }
            }

            // Show results
            m = gBest.model();
            renderPose(fbo, meshes[match.t->objId], imGBest, imGBestNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));

            // Render gbest to vizualization
            for (int y = 0; y < imGBestNormals.rows; y++) {
                for (int x = 0; x < imGBestNormals.cols; x++) {
                    auto px = imGBestNormals.at<cv::Vec3f>(y, x);

                    if (px[0] > 0 || px[1] > 0 || px[2] > 0) {
                        vizResult.at<cv::Vec3b>(y + bb.tl().y, x + bb.tl().x) = cv::Vec3b(px[0] * 128 + 128, px[1] * 128 + 128, px[2] * 128 + 128);
                    }
                }
            }

            cv::imshow("vizResultResult", vizResult);
            cv::imshow("imGBestNormals", imGBestNormals);
            cv::waitKey(1);
        }
        cv::waitKey(0);
    }

    void FinePose::generatePopulation(std::vector<Particle> &particles, int N) {
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
}