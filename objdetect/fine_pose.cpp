#include <glm/ext.hpp>
#include <opencv2/rgbd.hpp>
#include "fine_pose.h"
#include "../utils/parser.h"
#include "../utils/glutils.h"
#include "../core/particle.h"

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

        // Init frame buffer
        fbo.init();
    }

    void FinePose::renderPose(const Mesh &mesh, cv::Mat &depth, cv::Mat &normals, const glm::mat4 &modelView, const glm::mat4 &modelViewProjection) {
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
        normals = cv::Mat::zeros(SCR_HEIGHT, SCR_WIDTH, CV_32FC3);
        glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_BGR, GL_FLOAT, normals.data);

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
        depth = cv::Mat::zeros(SCR_HEIGHT, SCR_WIDTH, CV_32FC3);
        glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_BGR, GL_FLOAT, depth.data);

        // Unbind frame buffer
        fbo.unbind();

        // Convert to 1-channel and normalize depth
        cv::cvtColor(depth, depth, CV_BGR2GRAY);
    }

    void FinePose::estimate(std::vector<Match> &matches, const Scene &scene) {
        // TODO better handling for current pyramid
        auto &pyr = scene.pyramid[criteria->pyrLvlsDown];

        // Load templates
        std::vector<Template> templates;
        Parser parser(criteria);
        parser.parseObject("data/108x108/kinectv2/07/", templates, {34, 34});

        // Load scene
        cv::Rect rectGT(278, 220, 270, 270);
        cv::Mat sRGB = std::move(pyr.srcRGB);
        cv::Mat sGray = std::move(pyr.srcGray);
        cv::Mat sDepth = std::move(pyr.srcDepth);
        cv::rectangle(sRGB, rectGT, cv::Scalar(0, 255, 0));

        // Compute normals, edges, depth
        cv::Mat sNormals, sEdge, snDepth;
        cv::rgbd::RgbdNormals rgbdNormals(sDepth.rows, sDepth.cols, CV_32F, pyr.camera.K, 5,
                                          cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);
        rgbdNormals(sDepth, sNormals);

        // Edges
        sDepth.convertTo(snDepth, CV_32F, 1.0f / 65365.0f);
        cv::Laplacian(snDepth, sEdge, CV_32F, 5);
        cv::threshold(sEdge, sEdge, 0.3f, 0, CV_THRESH_TRUNC);
        cv::threshold(sEdge, sEdge, 0.02f, 1, CV_THRESH_BINARY);

        // Crop
        sNormals = sNormals(rectGT);
        sEdge = sEdge(rectGT);
        snDepth = snDepth(rectGT);

        // Resize to 108
//        cv::resize(sNormals, sNormals, cv::Size(SCR_WIDTH, SCR_HEIGHT));
//        cv::resize(sEdge, sEdge, cv::Size(SCR_WIDTH, SCR_HEIGHT));
//        cv::resize(snDepth, snDepth, cv::Size(SCR_WIDTH, SCR_HEIGHT));
        snDepth *= 1550;

        for (int y = 0; y < sNormals.rows; y++) {
            for (int x = 0; x < sNormals.cols; x++) {
                auto px = sNormals.at<cv::Vec3f>(y, x);
                sNormals.at<cv::Vec3f>(y, x) = cv::Vec3f(-px[2], -px[1], px[0]);
            }
        }

        cv::imshow("sNormals", sNormals);
        cv::imshow("sEdge", sEdge);
        cv::imshow("snDepth", snDepth);

        // Generators
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dR(-0.5f, 0.5f);
        static std::uniform_real_distribution<float> dRZ(-1.0f, 0.5f);
        static std::uniform_real_distribution<float> dT(-50, 50);
        static std::uniform_real_distribution<float> dTZ(-50, 150);
        static std::uniform_real_distribution<float> dVT(0, 10);
        static std::uniform_real_distribution<float> dVTz(0, 10);
        static std::uniform_real_distribution<float> dVR(0, 0.4f);
        static std::uniform_real_distribution<float> dRand(0, 1.0f);

        // References to templates
        Template &tGt = templates[0], &tOrg = templates[1];
        const int IT = 100, N = 100;

        // Rescale K
        rescaleK(tGt.camera.K, cv::Size(108, 108), cv::Size(240, 240));
        rescaleK(tOrg.camera.K, cv::Size(108, 108), cv::Size(240, 240));

        // Precompute matrices
        glm::mat4 VMatrix = vMat(tGt.camera.R, tGt.camera.t);
        glm::mat4 PMatrix = pMat(tGt.camera.K, 0, 0, SCR_WIDTH, SCR_HEIGHT);
        glm::mat4 MVPMatrix = mvpMat(glm::mat4(), VMatrix, PMatrix);

        // Precompute src matrices
        glm::mat4 orgVMatrix = vMat(tOrg.camera.R, tOrg.camera.t);
        glm::mat4 orgPMatrix = pMat(tOrg.camera.K, 0, 0, SCR_WIDTH, SCR_HEIGHT);
        glm::mat4 orgMVPMatrix = mvpMat(glm::mat4(), orgVMatrix, orgPMatrix);

        // Init GT depth
        cv::Mat gt, org, gtNormals, orgNormals, gtEdges, gtT;
        renderPose(meshes[tGt.objId], gt, gtNormals, VMatrix, MVPMatrix);
        renderPose(meshes[tOrg.objId], org, orgNormals, orgVMatrix, orgMVPMatrix);

        // Compute edges
        cv::Laplacian(gt, gtEdges, -1);
        cv::threshold(gtEdges, gtEdges, 0.5f, 1, CV_THRESH_BINARY);

        // Show org and ground truth
//        cv::imshow("Ground truth - Normals", gtNormals);
        cv::imshow("Found match - Normals", orgNormals);

        // Init particles
        glm::mat4 m;
        cv::Mat pose, poseNormals;
        std::vector<Particle> particles;
        Particle gBest;
        gBest.fitness = 0;

        for (int i = 0; i < N; ++i) {
            // Generate new particle
            particles.emplace_back(dT(gen), dT(gen), dTZ(gen), dRZ(gen), dR(gen), dR(gen), dVT(gen), dVT(gen),
                                   dVTz(gen), dVR(gen), dVR(gen), dVR(gen));

            // Render depth image
            m = particles[i].model();
            renderPose(meshes[tOrg.objId], pose, poseNormals, mvMat(m, orgVMatrix), mvpMat(m, orgVMatrix, orgPMatrix));

            // Compute fitness for new particle
            particles[i].fitness = Particle::objFun(snDepth, sNormals, sEdge, pose, poseNormals);

            // Save gBest
            if (particles[i].fitness < gBest.fitness) {
                gBest = particles[i];
            }
        }

        // PSO
        cv::Mat imGBest, imGBestNormals;
        m = gBest.model();
        renderPose(meshes[tOrg.objId], imGBest, imGBestNormals, mvMat(m, orgVMatrix), mvpMat(m, orgVMatrix, orgPMatrix));
        const float C1 = 0.3f, C2 = 0.3f, W = 0.90f;

        // Generations
        for (int i = 0; i < IT; i++) {
            std::cout << "Iteration: " << i << std::endl;

            for (auto &p : particles) {
                // Progress (updates velocity and moves particle)
                p.progress(W, C1, C2, gBest);

                // Fitness
                m = p.model();
                renderPose(meshes[tOrg.objId], pose, poseNormals, mvMat(m, orgVMatrix), mvpMat(m, orgVMatrix, orgPMatrix));
                p.fitness = Particle::objFun(snDepth, sNormals, sEdge, pose, poseNormals);

                // Check for pBest
                if (p.fitness < p.pBest.fitness) {
                    p.updatePBest();
                }

                // Check for gBest
                if (p.fitness < gBest.fitness) {
                    std::cout << p.fitness << std::endl;
                    gBest = p;

                    // Vizualization
                    m = gBest.model();
                    renderPose(meshes[tOrg.objId], imGBest, imGBestNormals, mvMat(m, orgVMatrix), mvpMat(m, orgVMatrix, orgPMatrix));
                }

                cv::imshow("imGBestNormals", imGBestNormals);
                cv::imshow("pose 2", poseNormals);
                cv::waitKey(1);
            }
        }

        // Show results
        m = gBest.model();
        renderPose(meshes[tOrg.objId], imGBest, imGBestNormals, mvMat(m, orgVMatrix), mvpMat(m, orgVMatrix, orgPMatrix));
        cv::imshow("imGBestNormals", imGBestNormals);
        cv::waitKey(0);
    }

    FinePose::~FinePose() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}