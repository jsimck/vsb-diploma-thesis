#include "classifier.h"
#include <boost/filesystem.hpp>
#include <gsl/gsl_qrng.h>
#include "../utils/timer.h"
#include "../utils/visualizer.h"
#include "../processing/processing.h"
#include "../utils/glutils.h"
#include "../core/particle.h"

namespace tless {
    Classifier::Classifier(cv::Ptr<ClassifierCriteria> criteria) : criteria(criteria) {
        // Init opengl
        initGL();
    }

    void Classifier::initGL() {
        // GLFW init and config
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

        // GLFW window creation
        cv::Size winSize(108, 108); // TODO do something with this
        GLFWwindow *window = glfwCreateWindow(winSize.width, winSize.height, "DrawDepth", NULL, NULL);

        if (window == NULL) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window);
        glfwHideWindow(window);
        glViewport(0, 0, winSize.width, winSize.height);

        // Init Glew after GLFW init
        if (glewInit()) {
            std::cerr << "Failed to initialize GLXW" << std::endl;
            return;
        }

        // Init Opengl global settings
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);

        // Init shaders, meshes and FBO
        initShaders();
        initMeshes("data/meshes.txt"); // TODO add into classifier param
        fbo.init();
    }

    void Classifier::initShaders() {
        shaders[SHADER_DEPTH] = Shader("data/shaders/depth.vert", "data/shaders/depth.frag");
        shaders[SHADER_NORMAL] = Shader("data/shaders/normal.vert", "data/shaders/normal.frag");
    }

    void Classifier::initMeshes(const std::string &meshesListPath) {
        std::ifstream ifs(meshesListPath);
        std::string path;
        int id;

        while (ifs >> id) {
            ifs >> path;
            meshes[id] = Mesh(path);
        }

        ifs.close();
    }

    void Classifier::train(std::string templatesListPath, std::string resultPath, std::vector<uint> indices) {
        std::ifstream ifs(templatesListPath);
        assert(ifs.is_open());

        // Init classifiers and parser
        Hasher hasher(criteria);
        Matcher matcher(criteria);
        Parser parser(criteria);

        // Init common
        std::ostringstream oss;
        std::vector<Template> templates, allTemplates;
        std::string path;

        // Create directories if they don't exist
        boost::filesystem::create_directories(resultPath);

        Timer tTraining;
        std::cout << "Training... " << std::endl;

        while (ifs >> path) {
            std::cout << "  |_ " << path;

            // Parse each object by one and save it
            parser.parseObject(path, templates, indices);

            // Train features for loaded templates
            matcher.train(templates);

            // Save templates for later hash table generation
            allTemplates.insert(allTemplates.end(), templates.begin(), templates.end());

            // Persist trained data
            oss << resultPath << std::setw(2) << std::setfill('0') << templates[0].objId << ".yml.gz";
            std::string trainedPath = oss.str();
            cv::FileStorage fsw(trainedPath, cv::FileStorage::WRITE);

            // Save templates data
            fsw << "templates" << "[";
            for (auto &t : templates) {
                fsw << t;
            }
            fsw << "]";

            // Cleanup
            oss.str("");
            fsw.release();
            templates.clear();
            std::cout << " -> " << trainedPath << std::endl;
        }

        ifs.close();

        // Save classifier info
        cv::FileStorage fsw(resultPath + "classifier.yml.gz", cv::FileStorage::WRITE);
        fsw << "criteria" << *criteria;
        std::cout << "  |_ info -> " << resultPath + "classifier.yml.gz" << std::endl;

        // Train hash tables
        std::cout << "  |_ Training hash tables... " << std::endl;
        hasher.train(allTemplates, tables);
        assert(!tables.empty());
        std::cout << "    |_ " << tables.size() << " hash tables generated" << std::endl;

        // Persist hashTables
        fsw << "tables" << "[";
        for (auto &table : tables) {
            fsw << table;
        }
        fsw << "]";
        fsw.release();

        std::cout << "  |_ tables -> " << resultPath + "classifier.yml.gz" << std::endl;
        std::cout << "DONE!, took: " << tTraining.elapsed() << " s" << std::endl << std::endl;
    }

    void Classifier::load(const std::string &trainedTemplatesListPath, const std::string &trainedPath) {
        std::ifstream ifs(trainedTemplatesListPath);
        assert(ifs.is_open());

        Timer tLoading;
        std::string path;
        std::cout << "Loading trained templates... " << std::endl;

        while (ifs >> path) {
            std::cout << "  |_ " << path;

            // Load trained data
            cv::FileStorage fsr(path, cv::FileStorage::READ);
            cv::FileNode tpls = fsr["templates"];

            // Loop through templates
            for (auto &&t : tpls) {
                Template nTpl;
                t >> nTpl;
                templates.push_back(nTpl);
            }

            fsr.release();
            std::cout << " -> LOADED" << std::endl;
        }

        // Load data set
        cv::FileStorage fsr(trainedPath + "classifier.yml.gz", cv::FileStorage::READ);
        fsr["criteria"] >> criteria;
        std::cout << "  |_ info -> LOADED" << std::endl;
        std::cout << "  |_ loading hashtables..." << std::endl;

        // Load hash tables
        cv::FileNode hashTables = fsr["tables"];
        for (auto &&table : hashTables) {
            tables.emplace_back(HashTable::load(table, templates));
        }

        fsr.release();
        std::cout << "  |_ hashTables -> LOADED (" << tables.size() << ")" << std::endl;
        std::cout << "DONE!, took: " << tLoading.elapsed() << " s" << std::endl << std::endl;
    }

    void Classifier::detect(std::string trainedTemplatesListPath, std::string trainedPath, std::string scenePath) {
        // Checks
        assert(criteria->info.smallestTemplate.area() > 0);
        assert(criteria->info.minEdgels > 0);

        // Init classifiers
        Parser parser(criteria);
        Objectness objectness(criteria);
        Hasher hasher(criteria);
        Matcher matcher(criteria);
        Visualizer viz(criteria);
        Scene scene;

        // Load trained template data
        load(trainedTemplatesListPath, trainedPath);

        // Image pyramid
        const int pyrLevels = criteria->pyrLvlsDown + criteria->pyrLvlsUp;

        // Timing
        Timer tTotal;
        double ttSceneLoading, ttObjectness, ttVerification, ttMatching, ttNMS;
        std::cout << "Matching started..." << std::endl << std::endl;

        for (int i = 0; i < 503; ++i) {
            // Reset timers
            ttObjectness = ttVerification = ttMatching = 0;
            tTotal.reset();

            // Load scene
            Timer tSceneLoading;
            scene = parser.parseScene(scenePath, i, criteria->pyrScaleFactor, criteria->pyrLvlsDown, criteria->pyrLvlsUp);
            ttSceneLoading = tSceneLoading.elapsed();

            // Verification for a pyramid
            for (int l = 0; l <= pyrLevels; ++l) {
                // Objectness detection
                Timer tObjectness;
                objectness.objectness(scene.pyramid[l].srcDepth, windows);
                ttObjectness += tObjectness.elapsed();
//                viz.objectness(scene.pyramid[l], windows);

                /// Verification and filtering of template candidates
                if (windows.empty()) {
                    continue;
                }

                Timer tVerification;
                hasher.verifyCandidates(scene.pyramid[l].srcDepth, scene.pyramid[l].srcNormals, tables, windows);
                ttVerification += tVerification.elapsed();
                viz.windowsCandidates(scene.pyramid[l], windows);

                /// Match templates
                Timer tMatching;
                matcher.match(scene.pyramid[l], windows, matches);
                ttMatching += tMatching.elapsed();
                windows.clear();
            }

            // Apply non-maxima suppression
//            viz.preNonMaxima(scene.pyramid[criteria->pyrLvlsDown], matches);
            Timer tNMS;
            nms(matches, criteria->overlapFactor);
            ttNMS = tNMS.elapsed();

            // Print results
            std::cout << std::endl << "Classification took: " << tTotal.elapsed() << "s" << std::endl;
            std::cout << "  |_ Scene loading took: " << ttSceneLoading << "s" << std::endl;
            std::cout << "  |_ Objectness detection took: " << ttObjectness << "s" << std::endl;
            std::cout << "  |_ Hashing verification took: " << ttVerification << "s" << std::endl;
            std::cout << "  |_ Template matching took: " << ttMatching << "s" << std::endl;
            std::cout << "  |_ NMS took: " << ttNMS << "s" << std::endl;

            // Vizualize results and clear current matches
            viz.matches(scene.pyramid[criteria->pyrLvlsDown], matches, 1);
            matches.clear();
        }
    }

    void Classifier::testPSO() {
        // Load templates
        std::vector<Template> templates;
        Parser parser(criteria);
        parser.parseObject("data/108x108/kinectv2/07/", templates, {28, 106});

        // Generators
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dR(-0.3f, 0.3f);
        static std::uniform_real_distribution<float> dT(-30, 30);
        static std::uniform_real_distribution<float> dTz(-50, 100);
        static std::uniform_real_distribution<float> dVT(0, 5);
        static std::uniform_real_distribution<float> dVTz(0, 10);
        static std::uniform_real_distribution<float> dVR(0, 0.4f);
        static std::uniform_real_distribution<float> dRand(0, 1.0f);

        // References to templates
        Template &tGt = templates[0], &tOrg = templates[1];
        const float C1 = 0.2f, C2 = 0.2f, W = 0.90f;
        const int IT = 50, N = 50;

        // Precompute matrices
        cv::Size winSize(108, 108);
        glm::mat4 VMatrix = vMat(tGt.camera.R, tGt.camera.t);
        glm::mat4 PMatrix = pMat(tGt.camera.K, 0, 0, winSize.width, winSize.height);
        glm::mat4 MVPMatrix = mvpMat(glm::mat4(), VMatrix, PMatrix);

        // Precompute src matrices
        glm::mat4 orgVMatrix = vMat(tOrg.camera.R, tOrg.camera.t);
        glm::mat4 orgPMatrix = pMat(tOrg.camera.K, 0, 0, winSize.width, winSize.height);
        glm::mat4 orgMVPMatrix = mvpMat(glm::mat4(), orgVMatrix, orgPMatrix);

        // Init GT depth
        cv::Mat gt, org, gtNormals, orgNormals, gtEdges, gtT;
        render(tGt, fbo, shaders[SHADER_DEPTH], shaders[SHADER_NORMAL], meshes[tGt.objId], gt, gtNormals, VMatrix, MVPMatrix);
        render(tOrg, fbo, shaders[SHADER_DEPTH], shaders[SHADER_NORMAL], meshes[tOrg.objId], org, orgNormals, orgVMatrix, orgMVPMatrix);

        // Compute edges

        cv::Laplacian(gt, gtEdges, -1);
        cv::threshold(gtEdges, gtEdges, 0.5f, 1, CV_THRESH_BINARY);

        // Show org and ground truth
        cv::imshow("Ground truth - Normals", gtNormals);
        cv::imshow("Found match - Normals", orgNormals);

        // Init particles
        glm::mat4 m;
        cv::Mat pose, poseNormals;
        std::vector<Particle> particles;
        Particle gBest;
        gBest.fitness = 0;

        // Generate population
        generatePopulation(particles, N);

        // Find best candidate
        for (int i = 0; i < N; ++i) {
            // Render depth image
            glm::mat4 m = particles[i].model();
            render(tOrg, fbo, shaders[SHADER_DEPTH], shaders[SHADER_NORMAL], meshes[tOrg.objId], pose, poseNormals, mvMat(m, orgVMatrix), mvpMat(m, orgVMatrix, orgPMatrix));

            // Compute fitness for new particle
            particles[i].fitness = fitness(gt, gtNormals, gtEdges, pose, poseNormals);

            // Save gBest
            if (particles[i].fitness < gBest.fitness) {
                gBest = particles[i];
            }
        }

        // PSO
        cv::Mat imGBest, imGBestNormals;
        m = gBest.model();
        render(tOrg, fbo, shaders[SHADER_DEPTH], shaders[SHADER_NORMAL], meshes[tOrg.objId], imGBest, imGBestNormals, mvMat(m, orgVMatrix), mvpMat(m, orgVMatrix, orgPMatrix));

        // Generations
        for (int i = 0; i < IT; i++) {
            std::cout << "Iteration: " << i << std::endl;

            for (auto &p : particles) {
                // Compute velocity
                p.v1 = computeVelocity(W, p.v1, p.tx, p.pBest.tx, gBest.tx, C1, C2, dRand(gen), dRand(gen));
                p.v2 = computeVelocity(W, p.v2, p.ty, p.pBest.ty, gBest.ty, C1, C2, dRand(gen), dRand(gen));
                p.v3 = computeVelocity(W, p.v3, p.tz, p.pBest.tz, gBest.tz, C1, C2, dRand(gen), dRand(gen));
                p.v4 = computeVelocity(W, p.v4, p.rx, p.pBest.rx, gBest.rx, C1, C2, dRand(gen), dRand(gen));
                p.v5 = computeVelocity(W, p.v5, p.ry, p.pBest.ry, gBest.ry, C1, C2, dRand(gen), dRand(gen));
                p.v6 = computeVelocity(W, p.v6, p.rz, p.pBest.rz, gBest.rz, C1, C2, dRand(gen), dRand(gen));

                // Update
                p.update();

                // Fitness
                m = p.model();
                render(tOrg, fbo, shaders[SHADER_DEPTH], shaders[SHADER_NORMAL], meshes[tOrg.objId], pose, poseNormals, mvMat(m, orgVMatrix), mvpMat(m, orgVMatrix, orgPMatrix));
                p.fitness = fitness(gt, gtNormals, gtEdges, pose, poseNormals);

                // Check for pBest
                if (p.fitness < p.pBest.fitness) {
                    p.updatePBest();
                }

                // Check for gBest
                if (p.fitness < gBest.fitness) {
                    gBest = p;

                    // Vizualization
                    m = gBest.model();
                    render(tOrg, fbo, shaders[SHADER_DEPTH], shaders[SHADER_NORMAL], meshes[tOrg.objId], imGBest, imGBestNormals, mvMat(m, orgVMatrix), mvpMat(m, orgVMatrix, orgPMatrix));
                }

                cv::imshow("imGBestNormals", imGBestNormals);
                cv::imshow("pose 2", poseNormals);
                cv::waitKey(1);
            }
        }

        // Show results
        m = gBest.model();
        render(tOrg, fbo, shaders[SHADER_DEPTH], shaders[SHADER_NORMAL], meshes[tOrg.objId], imGBest, imGBestNormals, mvMat(m, orgVMatrix), mvpMat(m, orgVMatrix, orgPMatrix));
        cv::imshow("imGBestNormals", imGBestNormals);
        cv::waitKey(0);
    }

    Classifier::~Classifier() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    float Classifier::computeVelocity(float w, float vi, float xi, float pBest, float gBest, float c1, float c2, float r1, float r2) {
        return w * vi + (c1 * r1) * (pBest - xi) + (c2 * r2) * (gBest - xi);
    }

    float Classifier::fitness(const cv::Mat &gt, const cv::Mat &gtNormals, const cv::Mat &gtEdges, const cv::Mat &pose, const cv::Mat &poseNormals) {
        float sumD = 0, sumU = 0, sumE = 0;
        const float tD = 20;
        const float inf = std::numeric_limits<float>::max();

        // Compute edges
        cv::Mat poseT, poseEdges;
        cv::Laplacian(pose, poseEdges, -1);
        cv::threshold(poseEdges, poseEdges, 0.5f, 255, CV_THRESH_BINARY_INV);
        poseEdges.convertTo(poseEdges, CV_8U);
        cv::distanceTransform(poseEdges, poseT, CV_DIST_L2, 3);

        for (int y = 0; y < gt.rows; y++) {
            for (int x = 0; x < gt.cols; x++) {
                // Compute distance transform
                if (gtEdges.at<float>(y, x) > 0) {
                    sumE += 1 / (poseT.at<float>(y, x) + 1);
                }

                // Skip invalid depth pixels for other tests pixels
                if (pose.at<float>(y, x) <= 0) {
                    continue;
                }

                // Compute depth diff
                float dDiff = std::abs(gt.at<float>(y, x) - pose.at<float>(y, x));
                if (dDiff > tD) {
                    sumD += 1 / inf;
                } else {
                    sumD += 1 / (dDiff + 1);
                }

                // Compare normals
                float dot = std::abs(gtNormals.at<cv::Vec3f>(y, x).dot(poseNormals.at<cv::Vec3f>(y, x)));
                sumU += 1 / (dot + 1);
            }
        }

        return -sumD * sumU * sumE;
    }

    void Classifier::generatePopulation(std::vector<Particle> &particles, int N) {
        // init sobol sequence for 6 dimensions
        Timer t;
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
            particles.emplace_back(
                (v[0] - 0.5) * 50,
                (v[1] - 0.5) * 50,
                (v[2] - 0.5) * 100,
                (v[3] - 0.5) * 0.75,
                (v[4] - 0.5) * 0.75,
                (v[5] - 0.5) * 0.75,
                d(gen) * 10,
                d(gen) * 10,
                d(gen) * 20,
                d(gen) * 0.25,
                d(gen) * 0.25,
                d(gen) * 0.25);
        }

        gsl_qrng_free(q);
        std::cout << t.elapsed() << std::endl;
    }
}