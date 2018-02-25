#include "classifier.h"
#include "../utils/timer.h"
#include "../utils/visualizer.h"
#include "../core/classifier_criteria.h"
#include "../processing/processing.h"

namespace tless {
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

        Timer t;
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
            oss << resultPath << "trained_" << std::setw(2) << std::setfill('0') << templates[0].id / 2000 << ".yml.gz";
            std::string trainedPath = oss.str();
            cv::FileStorage fsw(trainedPath, cv::FileStorage::WRITE);

            // Save templates data
            fsw << "templates" << "[";
            for (auto &tpl : templates) {
                fsw << tpl;
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
        std::cout << "DONE!, took: " << t.elapsed() << " s" << std::endl << std::endl;
    }

    void Classifier::load(const std::string &trainedTemplatesListPath, const std::string &trainedPath) {
        std::ifstream ifs(trainedTemplatesListPath);
        assert(ifs.is_open());

        Timer t;
        std::string path;
        std::cout << "Loading trained templates... " << std::endl;

        while (ifs >> path) {
            std::cout << "  |_ " << path;

            // Load trained data
            cv::FileStorage fsr(path, cv::FileStorage::READ);
            cv::FileNode tpls = fsr["templates"];

            // Loop through templates
            for (auto &&tpl : tpls) {
                Template nTpl;
                tpl >> nTpl;
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
        std::cout << "DONE!, took: " << t.elapsed() << " s" << std::endl << std::endl;
    }

    void Classifier::detect(std::string trainedTemplatesListPath, std::string trainedPath, std::string scenePath) {
        // Init classifiers
        Parser parser(criteria);
        Objectness objectness(criteria);
        Hasher hasher(criteria);
        Matcher matcher(criteria);

        // Load trained template data
        load(trainedTemplatesListPath, trainedPath);

        cv::Mat result;
        const float initialScale = .4f;
        const float scaleFactor = 1.25f;
        const int finalScaleLevel = 9;
        float scale = initialScale;

        // Test windows
        // scale 1 = .722222222f at cv::Rect(122, 126, 108, 108) [520]
        // scale 2 = .736111111f at cv::Rect(211, 67, 108, 108) [530]
        // scale 3 = .402777778f at cv::Rect(116, 85, 108, 108) [290]

//        criteria->info.maxDepth = 20000;
        Scene scene = parser.parseScene(scenePath, 0, 1.0f);
        Scene scene1 = parser.parseScene(scenePath, 0, .722222222f);
        Scene scene2 = parser.parseScene(scenePath, 0, .736111111f);
        Scene scene3 = parser.parseScene(scenePath, 0, .402777778f);

        Visualizer viz(criteria);
        std::cout << *criteria << std::endl;

        objectness.objectness(scene.srcDepth, windows);
        Visualizer::visualizeWindows(scene.srcRGB, windows, false, 0, "Locations detected");
        windows.clear();

        objectness.objectness(scene1.srcDepth, windows);
        Visualizer::visualizeWindows(scene1.srcRGB, windows, false, 0, "Locations detected");
        hasher.verifyCandidates(scene1.srcDepth, scene1.srcNormals, tables, windows);
        for (auto &window : windows) {
            viz.windowCandidates(scene1, window);
        }

        windows.clear();

        objectness.objectness(scene2.srcDepth, windows);
        Visualizer::visualizeWindows(scene2.srcRGB, windows, false, 0, "Locations detected");
        hasher.verifyCandidates(scene2.srcDepth, scene2.srcNormals, tables, windows);
        for (auto &window : windows) {
            viz.windowCandidates(scene2, window);
        }

        windows.clear();

        objectness.objectness(scene3.srcDepth, windows);
        Visualizer::visualizeWindows(scene3.srcRGB, windows, false, 0, "Locations detected");
        hasher.verifyCandidates(scene3.srcDepth, scene3.srcNormals, tables, windows);
        for (auto &window : windows) {
            viz.windowCandidates(scene3, window);
        }

        windows.clear();


//        Window window1(cv::Rect(122, 126, 108, 108), 100);
//        Window window2(cv::Rect(211, 67, 108, 108), 100);
//        Window window3(cv::Rect(116, 85, 108, 108), 100);
//
//        std::vector<Window> windows1, windows2, windows3;
//        windows1.push_back(window1);
//        windows2.push_back(window2);
//        windows3.push_back(window3);
//
//        hasher.verifyCandidates(scene1.srcDepth, scene1.srcNormals, tables, windows1);
//        hasher.verifyCandidates(scene2.srcDepth, scene2.srcNormals, tables, windows2);
//        hasher.verifyCandidates(scene3.srcDepth, scene3.srcNormals, tables, windows3);

//        cv::Mat tpl, tplNormals, tplSrc = cv::imread("data/108x108/05/rgb/0190.png", CV_LOAD_IMAGE_COLOR);
//        cv::Mat tplDepth = cv::imread("data/108x108/05/depth/0190.png", CV_LOAD_IMAGE_UNCHANGED);
//
//        for (auto &table : tables) {
//            quantizedNormals(tplDepth, tplNormals, 640.f, 640.f, 30000, static_cast<int>(criteria->maxDepthDiff / 0.61f));
//
//            tpl = tplSrc.clone();
//            cv::line(tpl, table.triplet.c, table.triplet.p1, cv::Scalar(0, 155, 0));
//            cv::line(tpl, table.triplet.c, table.triplet.p2, cv::Scalar(0, 155, 0));
//            cv::circle(tpl, table.triplet.c, 3, cv::Scalar(0, 0, 255), -1);
//            cv::circle(tpl, table.triplet.p1, 3, cv::Scalar(0, 255, 0), -1);
//            cv::circle(tpl, table.triplet.p2, 3, cv::Scalar(255, 0, 0), -1);
//
//            cv::Mat test = scene3.srcRGB.clone();
//            cv::line(test, table.triplet.c + window3.tl(), table.triplet.p1 + window3.tl(), cv::Scalar(0, 155, 0));
//            cv::line(test, table.triplet.c + window3.tl(), table.triplet.p2 + window3.tl(), cv::Scalar(0, 155, 0));
//            cv::circle(test, table.triplet.c + window3.tl(), 3, cv::Scalar(0, 0, 255), -1);
//            cv::circle(test, table.triplet.p1 + window3.tl(), 3, cv::Scalar(0, 255, 0), -1);
//            cv::circle(test, table.triplet.p2 + window3.tl(), 3, cv::Scalar(255, 0, 0), -1);
//
//            cv::Mat testNormals = scene3.srcNormals.clone();
//            cv::line(testNormals, table.triplet.c + window3.tl(), table.triplet.p1 + window3.tl(), cv::Scalar(255));
//            cv::line(testNormals, table.triplet.c + window3.tl(), table.triplet.p2 + window3.tl(), cv::Scalar(255));
//            cv::circle(testNormals, table.triplet.c + window3.tl(), 3, cv::Scalar(255), -1);
//            cv::circle(testNormals, table.triplet.p1 + window3.tl(), 3, cv::Scalar(255), -1);
//            cv::circle(testNormals, table.triplet.p2 + window3.tl(), 3, cv::Scalar(255), -1);
//
//            std::cout << "SCENE: " << std::endl;
//            std::cout << "  c: " << static_cast<int>(scene3.srcDepth.at<ushort>(table.triplet.c + window3.tl()));
//            std::cout << "  p1: " << static_cast<int>(scene3.srcDepth.at<ushort>(table.triplet.p1 + window3.tl()));
//            std::cout << "  p2: " << static_cast<int>(scene3.srcDepth.at<ushort>(table.triplet.p2 + window3.tl())) << std::endl;
//            std::cout << "  norm c: " << static_cast<int>(scene3.srcNormals.at<uchar>(table.triplet.c + window3.tl()));
//            std::cout << "  norm p1: " << static_cast<int>(scene3.srcNormals.at<uchar>(table.triplet.p1 + window3.tl()));
//            std::cout << "  norm p2: " << static_cast<int>(scene3.srcNormals.at<uchar>(table.triplet.p2 + window3.tl())) << std::endl << std::endl;
//
//            std::cout << "TEMPLATE: " << std::endl;
//            std::cout << "  c: " << static_cast<int>(tplDepth.at<ushort>(table.triplet.c));
//            std::cout << "  p1: " << static_cast<int>(tplDepth.at<ushort>(table.triplet.p1));
//            std::cout << "  p2: " << static_cast<int>(tplDepth.at<ushort>(table.triplet.p2)) << std::endl;
//            std::cout << "  norm c: " << static_cast<int>(tplNormals.at<uchar>(table.triplet.c));
//            std::cout << "  norm p1: " << static_cast<int>(tplNormals.at<uchar>(table.triplet.p1));
//            std::cout << "  norm p2: " << static_cast<int>(tplNormals.at<uchar>(table.triplet.p2)) << std::endl << std::endl;
//
//            cv::line(tplNormals, table.triplet.c, table.triplet.p1, cv::Scalar(255));
//            cv::line(tplNormals, table.triplet.c, table.triplet.p2, cv::Scalar(255));
//            cv::circle(tplNormals, table.triplet.c, 3, cv::Scalar(255), -1);
//            cv::circle(tplNormals, table.triplet.p1, 3, cv::Scalar(255), -1);
//            cv::circle(tplNormals, table.triplet.p2, 3, cv::Scalar(255), -1);
//
//            cv::imshow("tpl - depth", tplDepth);
//            cv::imshow("tpl - srcNormals", tplNormals);
//            cv::imshow("tpl", tpl);
//
//            cv::imshow("scene3 - depth", scene3.srcDepth);
//            cv::imshow("scene3 - srcNormals", testNormals);
//            cv::imshow("scene", test);
//            cv::waitKey(0);
//        }

//        Visualizer viz(criteria);
//        viz.windowCandidates(scene1, windows1[0]);
//        viz.windowCandidates(scene2, windows2[0]);
//        viz.windowCandidates(scene3, windows3[0]);

//        std::cout << "windows1 candidates: " << windows1[0].candidates.size() << std::endl;
//        std::cout << "windows2 candidates: " << windows2[0].candidates.size() << std::endl;
//        std::cout << "windows3 candidates: " << windows3[0].candidates.size() << std::endl;
//
//        cv::rectangle(scene.srcRGB, match1.scaledBB(1.0f).tl(), match1.scaledBB(1.0f).br(), cv::Scalar(0, 255, 0));
//        cv::rectangle(scene.srcRGB, match2.scaledBB(1.0f).tl(), match2.scaledBB(1.0f).br(), cv::Scalar(0, 255, 0));
//        cv::rectangle(scene.srcRGB, match3.scaledBB(1.0f).tl(), match3.scaledBB(1.0f).br(), cv::Scalar(0, 255, 0));
//
//        cv::rectangle(scene1.srcRGB, match1.objBB.tl(), match1.objBB.br(), cv::Scalar(0, 255, 0));
//        cv::rectangle(scene2.srcRGB, match2.objBB.tl(), match2.objBB.br(), cv::Scalar(0, 255, 0));
//        cv::rectangle(scene3.srcRGB, match3.objBB.tl(), match3.objBB.br(), cv::Scalar(0, 255, 0));
//
//        cv::imshow("scene", scene.srcRGB);
//        cv::imshow("scene1", scene1.srcRGB);
//        cv::imshow("scene2", scene2.srcRGB);
//        cv::imshow("scene3", scene3.srcRGB);
//        cv::waitKey(0);

//        Visualizer viz(criteria);
//
//        for (int i = 0; i < 503; ++i) {
//            for (int pyramidLevel = 0; pyramidLevel < finalScaleLevel; ++pyramidLevel) {
//                Timer tTotal;
//
//                // Load scene
//                Timer tSceneLoading;
//                Scene scene = parser.parseScene(scenePath, i, scale);
//                std::cout << "  |_ Scene loaded in: " << tSceneLoading.elapsed() << "s" << std::endl;
//
//                // Save scene at scale 1.0f for visualization
//                if (scale <= 1.1f && scale >= 0.9f) {
//                    result = scene.srcRGB.clone();
//                }
//
//                /// Objectness detection
//                assert(criteria->info.smallestTemplate.area() > 0);
//                assert(criteria->info.minEdgels > 0);
//
//                Timer tObjectness;
//                objectness.objectness(scene.srcDepth, windows, scale);
//                std::cout << "  |_ Objectness detection took: " << tObjectness.elapsed() << "s" << std::endl;
//
//                Visualizer::visualizeWindows(scene.srcRGB, windows, false, 0, "Locations detected");
//
//                /// Verification and filtering of template candidates
//                if (windows.empty()) {
//                    continue;
//                }
//
//                Timer tVerification;
//                hasher.verifyCandidates(scene.srcDepth, scene.srcNormals, tables, windows);
//                std::cout << "  |_ Hashing verification took: " << tVerification.elapsed() << "s" << std::endl;
//
//                for (auto &window : windows) {
//                    viz.windowCandidates(scene, window);
//                }
//
////                Visualizer::visualizeHashing(scene.srcRGB, scene.srcDepth, tables, windows, criteria, false);
////                Visualizer::visualizeWindows(scene.srcRGB, windows, false, 1, "Filtered locations");
//
//                /// Match templates
//                assert(!windows.empty());
//                matcher.match(scale, scene, windows, matches);
//
//                /// Show matched template results
//                std::cout << std::endl << "Matches size: " << matches.size() << std::endl;
////                Visualizer::visualizeMatches(scene.srcRGB, scale, matches, "data/400x400/", 0);
//
//                std::cout << "Classification took: " << tTotal.elapsed() << "s" << std::endl;
//                scale *= scaleFactor;
//                windows.clear();
//            }
//
//            // Apply non-maxima suppression
//            nonMaximaSuppression(matches, criteria->overlapFactor);
//
//            // Visualize results
//            Visualizer::visualizeMatches(result, 1.0f, matches, "data/400x400/", 1, "Pyramid results");
//            scale = initialScale;
//            matches.clear();
//        }
    }
}