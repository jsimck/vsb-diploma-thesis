#include <glm/ext.hpp>
#include <opencv2/rgbd.hpp>
#include "fine_pose.h"
#include "../utils/parser.h"
#include "../utils/glutils.h"
#include "../core/particle.h"
#include "../processing/processing.h"

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

    void FinePose::renderPose(const FrameBuffer &fbo, const Mesh &mesh, cv::Mat &depth, cv::Mat &normals,
                                  const glm::mat4 &model, const glm::mat4 &view, const glm::mat4 &projection) {
        // Bind frame buffer
//        fbo.bind();

//        // Activate shader and set uniforms
//        shaders[SHADER_NORMAL].use();
//        shaders[SHADER_NORMAL].setMat4("MMatrix", model);
//        shaders[SHADER_NORMAL].setMat4("VMatrix", view);
//        shaders[SHADER_NORMAL].setMat4("PMatrix", projection);
//
//        // Draw mesh
//        mesh.draw();
//
//        // Read data from frame buffer
//        normals = cv::Mat::zeros(fbo.height, fbo.width, CV_32FC3);
//        glReadPixels(0, 0, fbo.width, fbo.height, GL_BGR, GL_FLOAT, normals.data);


        // Activate shader and set uniforms
        shaders[SHADER_DEPTH].use();
        shaders[SHADER_DEPTH].setMat4("NMatrix", nMat(model, view));
        shaders[SHADER_DEPTH].setMat4("MMatrix", model);
        shaders[SHADER_DEPTH].setMat4("VMatrix", view);
        shaders[SHADER_DEPTH].setMat4("PMatrix", projection);

        // Draw mesh
        mesh.draw();

        // Read data from frame buffer
        depth = cv::Mat::zeros(fbo.height, fbo.width, CV_32FC3);
        glReadPixels(0, 0, fbo.width, fbo.height, GL_BGR, GL_FLOAT, depth.data);

        // Unbind frame buffer
//        fbo.unbind();

        // Convert to 1-channel and normalize depth
        cv::cvtColor(depth, depth, CV_BGR2GRAY);
    }

    void FinePose::estimate(std::vector<Match> &matches, const Scene &scene) {
        // Constants
        const int IT = 50, N = 50;
        const float C1 = 0.2f, C2 = 0.2f, W = 0.85f;

        // Parse data
        Parser parser(criteria);
        std::vector<std::vector<SceneGt>> sceneGts;
        cv::FileStorage fs("data/scenes/kinectv2/01/gt.yml", cv::FileStorage::READ);

        for (int i = 0; i < 504; i++) {
            // Load scene info
            std::string infoIndex = "scene_" + std::to_string(i);
            cv::FileNode infoNode = fs[infoIndex];

            std::vector<SceneGt> currentGts;

            const int ids[4] = {2, 25, 29, 30};
            for (int j = 0; j < 4; j++) {
                // Parse yml file
                SceneGt gt;
                std::vector<float> vCamRw2c, vCamTw2c;

                std::string key1 = "cam_R_m2c_" + std::to_string(ids[j]);
                std::string key2 = "cam_t_m2c_" + std::to_string(ids[j]);
                std::string key3 = "obj_id_" + std::to_string(ids[j]);

                infoNode[key1] >> vCamRw2c;
                infoNode[key2] >> vCamTw2c;
                infoNode[key3] >> gt.objID;

                gt.R = cv::Mat(3, 3, CV_32FC1, vCamRw2c.data()).clone();
                gt.t = cv::Mat(3, 1, CV_32FC1, vCamTw2c.data()).clone();
                currentGts.push_back(gt);
            }

            sceneGts.push_back(currentGts);
        }

        fs.release();

        // Generators
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dR(-0.3f, 0);
        static std::uniform_real_distribution<float> dRZ(-0.3f, 0);
        static std::uniform_real_distribution<float> dT(-50, 50);
        static std::uniform_real_distribution<float> dTZ(-50, 50);
        static std::uniform_real_distribution<float> dVT(0, 10);
        static std::uniform_real_distribution<float> dVTz(0, 10);
        static std::uniform_real_distribution<float> dVR(0, 0.2f);
        static std::uniform_real_distribution<float> dRand(0, 1.0f);

        // Init scene
        cv::Mat sNormals, sEdge, sDepth;
        float totalSum = 0;
        int totalCount = 0;

        // Loop through mateches
//        for (auto &match : matches) {
            // Enlarge BB
//            cv::Rect bb(match.normObjBB.x - 10, match.normObjBB.y - 10, match.normObjBB.width + 20, match.normObjBB.height + 20);

        cv::Rect bb(0, 0, 720, 540);
        for (int i = 0; i < 504; i++) {
            // Create FBO with given size and update viewport size
            FrameBuffer fbo(bb.width, bb.height);
            fbo.bind();
            Scene sc = parser.parseScene("data/scenes/kinectv2/01/", i, 1, 0, 0);
            auto &pyr = sc.pyramid[0];
            pyr.srcDepth.convertTo(sDepth, CV_32FC1);

            // Projection matrix
            cv::Mat K = pyr.camera.K;
            glm::mat4 PMatrix = pMat(K, 0, 0, bb.width, bb.height);
            cv::Mat depth = cv::Mat::zeros(bb.height, bb.width, CV_32FC1);

            /// DEPTH
            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            for (auto &sceneGt : sceneGts[i]) {
                cv::Mat gtD, gtN;
                glm::mat4 VMatrix = vMat(sceneGt.R, sceneGt.t);
                renderPose(fbo, meshes[sceneGt.objID], depth, gtN, glm::mat4(), VMatrix, PMatrix);
            }

            // Delete background
            for (int y = 0; y < depth.rows; y++) {
                for (int x = 0; x < depth.cols; x++) {
                    if (depth.at<float>(y, x) <= 0) {
                        sDepth.at<float>(y, x) = 0;
                    }
                }
            }

            // Compute average
            cv::Mat div = (sDepth / depth);
            float sum = 0;
            int count = 0;

            for (int y = 0; y < div.rows; y++) {
                for (int x = 0; x < div.cols; x++) {
                    float px = div.at<float>(y, x);
                    if (px > 0) {
                        sum += px;
                        count++;
                    }
                }
            }

            std::cout << (sum / static_cast<float>(count)) << std::endl;
            cv::imshow("Depth", depth);
            cv::waitKey(1);

            totalSum += (sum / static_cast<float>(count));
            totalCount++;

            // Precompute matrices
//            glm::mat4 VMatrix = vMat(pyr.camera.R, pyr.camera.t);
//
//            // Flip
//            glm::mat4 yzFlip; // Create flip matrix for coordinate system conversion
//            yzFlip[1][1] = -1;
//            yzFlip[2][2] = -1;
//
//            // Flip
//            M1 *= yzFlip;
//            M2 *= yzFlip;
//            M3 *= yzFlip;
//
//            // Transpose
//            M1 = glm::transpose(M1);
//            M2 = glm::transpose(M2);
//            M3 = glm::transpose(M3);
//
//            // Init particles
//            glm::mat4 m;
//            cv::Mat pose, poseNormals;
//            std::vector<Particle> particles;
//
//            cv::Mat d1, d2, d3, n1, n2, n3;
//            renderPose(fbo, meshes[5], d1, n1, glm::mat4(), M1, PMatrix);
//            renderPose(fbo, meshes[6], d2, n2, glm::mat4(), M2, PMatrix);
//            renderPose(fbo, meshes[7], d3, n3, glm::mat4(), M3, PMatrix);
//
//            // Render match for reference
//            cv::Mat org;
//            org = d1;
//
//            cv::Mat snDepth;
//            pyr.srcDepth.convertTo(snDepth, CV_32F);
//
//
////            std::cout << org << std::endl;
//            cv::Mat div = (snDepth / org);
//            float sum = 0;
//            int count = 0;
//
//            for (int y = 0; y < div.rows; y++) {
//                for (int x = 0; x < div.cols; x++) {
//                    float px = div.at<float>(y, x);
//                    if (px > 0) {
//                        sum += px;
//                        count++;
//                    }
//                }
//            }
//
//            std::cout << (sum / static_cast<float>(count)) << std::endl;
//
//            cv::normalize(org, org, 0, 1, CV_MINMAX);
//            cv::normalize(snDepth, snDepth, 0, 1, CV_MINMAX);
//            cv::imshow("snDepth", org);
//            cv::imshow("org", org);
//            cv::waitKey(0);
            fbo.unbind();
        }

        std::cout << "FINAL: " << (totalSum / static_cast<float>(totalCount)) << std::endl;

            // Init global best
//            Particle gBest;
//            gBest.fitness = 0;
//
//            // Generate initial particle positions
//            for (int i = 0; i < N; ++i) {
//                // Generate new particle
//                particles.emplace_back(dT(gen), dT(gen), dTZ(gen), dRZ(gen), dR(gen), dR(gen),
//                                       dVT(gen), dVT(gen), dVTz(gen), dVR(gen), dVR(gen), dVR(gen));
//
//                // Render depth image
//                m = particles[i].model();
//                renderPose(fbo, meshes[match.t->objId], pose, poseNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));
//
//                // Compute fitness for new particle
//                particles[i].fitness = Particle::objFun(depth, normals, edges, pose, poseNormals);
//
//                // Save gBest
//                if (particles[i].fitness < gBest.fitness) {
//                    gBest = particles[i];
//                }
//            }
//
//            // PSO
//            cv::Mat imGBest, imGBestNormals;
//            m = gBest.model();
//            renderPose(fbo, meshes[match.t->objId], imGBest, imGBestNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));
//
//            // Generations
//            for (int i = 0; i < IT; i++) {
//                std::cout << "Iteration: " << i << std::endl;
//
//                for (auto &p : particles) {
//                    // Progress (updates velocity and moves particle)
//                    p.progress(W, C1, C2, gBest);
//
//                    // Fitness
//                    m = p.model();
//                    renderPose(fbo, meshes[match.t->objId], pose, poseNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));
//                    p.fitness = Particle::objFun(depth, normals, edges, pose, poseNormals);
//
//                    // Check for pBest
//                    if (p.fitness < p.pBest.fitness) {
//                        p.updatePBest();
//                    }
//
//                    // Check for gBest
//                    if (p.fitness < gBest.fitness) {
//                        gBest = p;
//
//                        // Vizualization
//                        m = gBest.model();
//                        renderPose(fbo, meshes[match.t->objId], imGBest, imGBestNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));
//                    }
//
//                    cv::imshow("imGBestNormals", imGBestNormals);
//                    cv::imshow("pose 2", poseNormals);
//                    cv::waitKey(1);
//                }
//            }
//
//            // Show results
//            m = gBest.model();
//            renderPose(fbo, meshes[match.t->objId], imGBest, imGBestNormals, mvMat(m, VMatrix), mvpMat(m, VPMatrix));
//            cv::imshow("imGBestNormals", imGBestNormals);
//            cv::waitKey(0);
//        }
    }

    FinePose::~FinePose() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}