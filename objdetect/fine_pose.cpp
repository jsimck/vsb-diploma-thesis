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
        GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "DrawDepth", NULL, NULL);

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

    void FinePose::loadShaders(const std::string &shadersBasePath) {
        shaders[SHADER_DEPTH] = Shader(shadersBasePath + "depth.vert", shadersBasePath + "depth.frag");
        shaders[SHADER_NORMAL] = Shader(shadersBasePath + "normal.vert", shadersBasePath + "normal.frag");
    }

    void FinePose::loadMeshes(const std::string &meshesListPath) {
        std::ifstream ifs(meshesListPath);
        std::string path;
        int id;

        // File format in ifs [obj_id path]
        while (ifs >> id) {
            ifs >> path;
            meshes[id] = Mesh(path);
        }

        ifs.close();
    }

    FinePose::FinePose(cv::Ptr<ClassifierCriteria> criteria, const std::string &shadersBasePath,
                       const std::string &meshesListPath) : criteria(criteria) {
        // First initialize OpenGL
        initOpenGL();

        // Load shaders and meshes
        loadShaders(shadersBasePath);
        loadMeshes(meshesListPath);
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
        shaders[SHADER_NORMAL].use();
        shaders[SHADER_NORMAL].setMat4("NMatrix", nMat(modelView));
        shaders[SHADER_NORMAL].setMat4("MVPMatrix", modelViewProjection);

        // Draw mesh
        mesh.draw();

        // Read data from frame buffer
        normals = cv::Mat::zeros(fbo.height, fbo.width, CV_32FC3);
        glReadPixels(0, 0, fbo.width, fbo.height, GL_BGR, GL_FLOAT, normals.data);

        /// DEPTH
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Activate shader and set uniforms
        shaders[SHADER_DEPTH].use();
        shaders[SHADER_DEPTH].setMat4("MVMatrix", modelView);
        shaders[SHADER_DEPTH].setMat4("MVPMatrix", modelViewProjection);

        // Draw mesh
        mesh.draw();

        // Read data from frame buffer
        depth = cv::Mat::zeros(fbo.height, fbo.width, CV_32FC3);
        glReadPixels(0, 0, fbo.width, fbo.height, GL_BGR, GL_FLOAT, depth.data);

        // Unbind frame buffer
        fbo.unbind();

        // Convert to 1-channel and normalize depth
        cv::cvtColor(depth, depth, CV_BGR2GRAY);
    }

    void FinePose::estimate(std::vector<Match> &matches, const Scene &scene) {
        // Constants
        const int IT = 100, N = 100;
        const float C1 = 0.2f, C2 = 0.2f, W = 0.85f;

        // Init scene
        auto &pyr = scene.pyramid[criteria->pyrLvlsDown]; // TODO better handling of scene loading
        cv::Mat sNormals, sEdge, sDepth;

        // Normalize min and max depths to look for objectness in
        auto minMag = static_cast<int>(criteria->objectnessDiameterThreshold * criteria->info.smallestDiameter * criteria->info.depthScaleFactor);
        depthEdgels(pyr.srcDepth, sEdge, criteria->info.minDepth, criteria->info.maxDepth, minMag);
        pyr.srcDepth.convertTo(sDepth, CV_32F);

        // Canny test
        cv::Mat canny;
        cv::Canny(pyr.srcGray, canny, 80, 120);

        // Thin edges
//        cv::normalize(sEdge, sEdge, 0, 255, CV_MINMAX);
//        cv::ximgproc::thinning(sEdge, sEdge, cv::ximgproc::THINNING_GUOHALL);

        // Loop through mateches
        for (auto &match : matches) {
            // Enlarge BB
            cv::Rect bb(match.normObjBB.x - 20, match.normObjBB.y - 20, match.normObjBB.width + 40, match.normObjBB.height + 40);
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
            generatePopulation(particles, N);

            // Render match for reference
            cv::Mat org, orgNormals;
            renderPose(fbo, meshes[match.t->objId], org, orgNormals, VMatrix, VPMatrix);
            cv::imshow("Found match", orgNormals);

            // Init global best
            Particle gBest;
            gBest.fitness = 0;

            // Generate initial particle positions
            for (int i = 0; i < N; ++i) {
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
            for (int i = 0; i < IT; i++) {
                std::cout << "Iteration: " << i << std::endl;

                for (auto &p : particles) {
                    // Progress (updates velocity and moves particle)
                    p.progress(W, W, C1, C2, gBest);

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
                        cv::imshow("imGBestNormals", imGBestNormals);
                        cv::waitKey(1);
                    }

                    cv::imshow("pose 2", poseNormals);
                    cv::waitKey(1);
                }
            }

            // Show results
            m = gBest.model();
            renderPose(fbo, meshes[match.t->objId], imGBest, imGBestNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));
            cv::imshow("imGBestNormals", imGBestNormals);
            cv::waitKey(0);
        }
    }

    void FinePose::generatePopulation(std::vector<Particle> &particles, int N) {
        // init sobol sequence for 6 dimensions
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