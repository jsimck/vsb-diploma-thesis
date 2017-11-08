#include "classifier.h"
#include "../utils/timer.h"
#include "../utils/visualizer.h"
#include "../core/classifier_criteria.h"

Classifier::Classifier() {
    criteria = std::make_shared<ClassifierCriteria>();

    // ------- Train params -------
    // Objectness
    criteria->trainParams.objectness.tEdgesMin = 0.01f;
    criteria->trainParams.objectness.tEdgesMax = 0.1f;

    // Hasher
    criteria->trainParams.hasher.grid = cv::Size(12, 12);
    criteria->trainParams.hasher.tablesCount = 100;
    criteria->trainParams.hasher.binCount = 5;
    criteria->trainParams.hasher.maxDistance = 3;

    // Matcher
    criteria->trainParams.matcher.pointsCount = 100;


    // ------- Detect params -------
    // Objectness
    criteria->detectParams.objectness.step = 5;
    criteria->detectParams.objectness.tMatch = 0.3f;

    // Hasher
    criteria->detectParams.hasher.minVotes = 3;

    // Matcher
    criteria->detectParams.matcher.tMatch = 0.6f;
    criteria->detectParams.matcher.tOverlap = 0.1f;
    criteria->detectParams.matcher.neighbourhood = cv::Range(-2, 2);
    criteria->detectParams.matcher.tColorTest = 3;
    criteria->detectParams.matcher.depthDeviationFunction = {{10000, 0.14f}, {15000, 0.12f}, {20000, 0.1f}, {65600, 0.08f}};


    // Init classifiers
    objectness.setCriteria(criteria);
    hasher.setCriteria(criteria);
    matcher.setCriteria(criteria);
}

void Classifier::train(std::string templatesListPath, std::string resultPath, std::vector<uint> indices) {
    std::ifstream ifs(templatesListPath);
    assert(ifs.is_open());

    // Init parser and common
    Parser parser(criteria);
    parser.indices.swap(indices);

    std::ostringstream oss;
    std::vector<Template> tpls, allTemplates;
    std::string path;

    Timer t;
    std::cout << "Training... " << std::endl;

    while (ifs >> path) {
        std::cout << "  |_ " << path;

        // Parse each object by one and save it
        parser.parse(path, tpls);

        // Train features for loaded templates
        matcher.train(tpls);

        // Save templates for later hash table generation
        allTemplates.insert(allTemplates.end(), tpls.begin(), tpls.end());

        // Extract min edgels for objectness detection
        objectness.extractMinEdgels(tpls);

        // Persist trained data
        oss.str("");
        oss << resultPath << "trained_" << std::setw(2) << std::setfill('0') << tpls[0].id / 2000 << ".yml.gz";
        std::string trainedPath = oss.str();
        cv::FileStorage fsw(trainedPath, cv::FileStorage::WRITE);

        fsw << "templates" << "[";
        for (auto &tpl : tpls) {
            tpl.save(fsw);
        }
        fsw << "]";

        fsw.release();
        tpls.clear();
        std::cout << " -> " << trainedPath << std::endl;
    }

    // Save data set
    cv::FileStorage fsw(resultPath + "classifier.yml.gz", cv::FileStorage::WRITE);
    criteria->save(fsw);
    std::cout << "  |_ info -> " << resultPath + "classifier.yml.gz" << std::endl;

    // Train hash tables
    std::cout << "  |_ Training hash tables... " << std::endl;
    hasher.train(allTemplates, tables);
    assert(!tables.empty());
    std::cout << "    |_ " << tables.size() << " hash tables generated" <<std::endl;

    // Persist hashTables
    fsw << "tables" << "[";
    for (auto &table : tables) {
        table.save(fsw);
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
            templates.emplace_back(Template::load(tpl));
        }

        fsr.release();
        std::cout << " -> LOADED" << std::endl;
    }

    // Load data set
    cv::FileStorage fsr(trainedPath + "classifier.yml.gz", cv::FileStorage::READ);
    ClassifierCriteria::load(fsr, criteria);
    std::cout << "  |_ info -> LOADED" << std::endl;

    // Load hash tables
    cv::FileNode hashTables = fsr["tables"];
    for (auto &&table : hashTables) {
        tables.emplace_back(HashTable::load(table, templates));
    }

    fsr.release();
    std::cout << "  |_ hashTables -> LOADED (" << tables.size() << ")" << std::endl;

    std::cout << "DONE!, took: " << t.elapsed() << " s" << std::endl << std::endl;
}

void Classifier::loadScene(const std::string &scenePath, const std::string &sceneName) {
    // Checks
    assert(scenePath.length() > 0);
    assert(sceneName.length() > 0);

    // Load scenes
    cv::Mat srcScene = cv::imread(scenePath + "rgb/" + sceneName + ".png", CV_LOAD_IMAGE_COLOR);
    cv::Mat srcSceneDepth = cv::imread(scenePath + "depth/" + sceneName + ".png", CV_LOAD_IMAGE_UNCHANGED);

    // Resize - TODO remove and add scale pyramid functionality
    cv::resize(srcScene, scene, cv::Size(), 1.2, 1.2);
    cv::resize(srcSceneDepth, sceneDepth, cv::Size(), 1.2, 1.2);

    // Convert and normalize
    cv::cvtColor(scene, sceneHSV, CV_BGR2HSV);
    cv::cvtColor(scene, sceneGray, CV_BGR2GRAY);
    sceneGray.convertTo(sceneGray, CV_32F, 1.0f / 255.0f);
    sceneDepth.convertTo(sceneDepth, CV_32F); // TODO work with 16U (int) rather than floats
    sceneDepth.convertTo(sceneDepthNorm, CV_32F, 1.0f / 65536.0f);

    // Check if conversion went ok
    assert(!sceneHSV.empty());
    assert(!sceneGray.empty());
    assert(!sceneDepthNorm.empty());
    assert(scene.type() == CV_8UC3);
    assert(sceneHSV.type() == CV_8UC3);
    assert(sceneGray.type() == CV_32FC1);
    assert(sceneDepth.type() == CV_32FC1);
    assert(sceneDepthNorm.type() == CV_32FC1);
}

void Classifier::detect(std::string trainedTemplatesListPath, std::string trainedPath, std::string scenePath) {
    // Load trained template data and scene
    load(trainedTemplatesListPath, trainedPath);
    std::ostringstream oss;

    for (int i = 0; i < 503; ++i) {
        // Load scene
        oss << std::setfill('0') << std::setw(4) << i;
        loadScene(scenePath, oss.str());

        Timer tTotal;

        /// Objectness detection
        assert(criteria->info.smallestTemplate.area() > 0);
        assert(criteria->info.minEdgels > 0);

//        std::cout << "Objectness detection started... " << std::endl;
        Timer tObjectness;
        objectness.objectness(sceneDepthNorm, windows);
//        std::cout << "  |_ Windows classified as containing object extracted: " << windows.size() << std::endl;
//        std::cout << "DONE! took: " << tObjectness.elapsed() << "s" << std::endl << std::endl;

//        Visualizer::visualizeWindows(this->scene, windows, false, 1, "Locations detected");

        /// Verification and filtering of template candidates
        assert(!tables.empty());

//        std::cout << "Verification of template candidates, using trained HashTables started... " << std::endl;
        Timer tVerification;
        hasher.verifyCandidates(sceneDepth, tables, windows);
//        std::cout << "DONE! took: " << tVerification.elapsed() << "s" << std::endl << std::endl;

//        Visualizer::visualizeHashing(scene, sceneDepth, tables, windows, info, hasher.getGrid(), false);
//        Visualizer::visualizeWindows(this->scene, windows, false, 1, "Filtered locations");

        /// Match templates
        assert(!windows.empty());

//        std::cout << "Template matching started... " << std::endl;
        Timer tMatching;
        matcher.match(1.2f, sceneHSV, sceneGray, sceneDepth, windows, matches);
//        std::cout << "DONE! " << matches.size() << " matches found, took: " << tMatching.elapsed() << "s" << std::endl << std::endl;
        std::cout << "Classification took: " << tTotal.elapsed() << "s" << std::endl;

        /// Show matched template results
        Visualizer::visualizeMatches(scene, matches, "data/", 1);

        // Cleanup
        windows.clear();
        matches.clear();
        oss.str("");
    }
}