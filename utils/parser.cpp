#include "parser.h"
#include "../processing/processing.h"

uint Parser::idCounter = 0;

void Parser::parse(std::vector<Group> &groups, DataSetInfo &info) {
    // Checks
    assert(!folders.empty());
    int parsedCount = 0;

    // Reset data set
    info.reset();

    // Parse
    for (auto &tplName : folders) {
        std::vector<Template> templates;
        parseTemplate(templates, info, tplName);
        groups.emplace_back(tplName, templates);
        parsedCount += templates.size();
        std::cout << "  |_ Parsed: " << tplName << ", templates size: " << templates.size() << std::endl;
    }

    std::cout << "  |_ Parsed total: " << parsedCount << " templates" << std::endl;
}

void Parser::parseTemplate(std::vector<Template> &templates, DataSetInfo &info, const std::string &tplName) {
    // Load obj_gt
    cv::FileStorage fs;
    fs.open(basePath + tplName + "/gt.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    const uint size = static_cast<uint>((!indices.empty()) ? indices.size() : tplCount);
    for (uint i = 0; i < size; i++) {
        uint tplIndex = static_cast<uint>((!indices.empty()) ? indices[i] : i);
        std::string index = "tpl_" + std::to_string(tplIndex);
        cv::FileNode objGt = fs[index];

        // Parse template gt file
        templates.push_back(parseGt(tplIndex, basePath + tplName, objGt, info));
        idCounter++;
    }

    // Load obj_info
    fs.open(basePath + tplName + "/info.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    for (uint i = 0; i < size; i++) {
        uint tplIndex = static_cast<uint>((!indices.empty()) ? indices[i] : i);
        std::string index = "tpl_" + std::to_string(tplIndex);
        cv::FileNode objGt = fs[index];

        // Parse template info file
        parseInfo(templates[i], objGt);
    }

    fs.release();
}

Template Parser::parseGt(uint index, const std::string &path, cv::FileNode &gtNode, DataSetInfo &info) {
    // Init template param matrices
    std::vector<float> vCamRm2c, vCamTm2c;
    std::vector<int> vObjBB;

    // Nodes containing matrices and vectors to parseTemplate
    gtNode["obj_bb"] >> vObjBB;
    gtNode["cam_R_m2c"] >> vCamRm2c;
    gtNode["cam_t_m2c"] >> vCamTm2c;

    // Parse objBB
    cv::Rect objBB(vObjBB[0], vObjBB[1], vObjBB[2], vObjBB[3]);

    // Create filename from index
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << index;
    std::string fileName = ss.str();

    // Load image
    cv::Mat srcHSV;
    cv::Mat src = cv::imread(path + "/rgb/" + fileName + ".png", CV_LOAD_IMAGE_COLOR);
    cv::Mat srcDepth = cv::imread(path + "/depth/" + fileName + ".png", CV_LOAD_IMAGE_UNCHANGED);

    // Convert to grayscale and HSV
    cv::cvtColor(src, srcHSV, CV_BGR2HSV);
    cv::cvtColor(src, src, CV_BGR2GRAY);

    // Convert to float
    src.convertTo(src, CV_32F, 1.0f / 255.0f);
    // TODO use CV_16S rather than floats
    srcDepth.convertTo(srcDepth, CV_32F); // because of surface normal calculation, don'tpl doo normalization

    // Find smallest object
    if (objBB.area() < info.smallestTemplate.area()) {
        info.smallestTemplate.width = objBB.width;
        info.smallestTemplate.height = objBB.height;
    }

    // Find largest object
    if (objBB.width >= info.maxTemplate.width) {
        info.maxTemplate.width = objBB.width;
    }
    if (objBB.height >= info.maxTemplate.height) {
        info.maxTemplate.height = objBB.height;
    }

    // Calculate angles
    cv::Mat angles, magnitude;
    Processing::orientationGradients(src, angles, magnitude);

    // Checks
    assert(!vObjBB.empty());
    assert(!vCamRm2c.empty());
    assert(!vCamTm2c.empty());
    assert(!src.empty());
    assert(!srcHSV.empty());
    assert(!srcDepth.empty());

    // Matrix type checks
    assert(src.type() == CV_32FC1);
    assert(srcDepth.type() == CV_32FC1);

    return {
        idCounter, fileName, src, srcHSV, srcDepth, angles, objBB,
        cv::Mat(3, 3, CV_32FC1, vCamRm2c.data()).clone(),
        cv::Vec3d(vCamTm2c[0], vCamTm2c[1], vCamTm2c[2])
    };
}

void Parser::parseInfo(Template &tpl, cv::FileNode &infoNode) {
    // Init template param matrices
    std::vector<float> vCamK;
    int elev, mode;

    // Parse params contained in info.yml
    infoNode["cam_K"] >> vCamK;
    infoNode["elev"] >> elev;
    infoNode["mode"] >> mode;

    // Checks
    assert(!vCamK.empty());

    // Assign new params to template
    tpl.elev = elev;
    tpl.mode = mode;
    tpl.camK = cv::Mat(3, 3, CV_32FC1, vCamK.data()).clone();
}

void Parser::clearIndices() {
    indices.clear();
}

int Parser::getIdCounter() {
    return idCounter;
}

std::string Parser::getBasePath() const {
    return this->basePath;
}

uint Parser::getTplCount() const {
    return this->tplCount;
}

const std::vector<std::string> &Parser::getTemplateFolders() const {
    return this->folders;
}

const std::vector<uint> &Parser::getIndices() const {
    return this->indices;
}

void Parser::setBasePath(std::string path) {
    assert(path.length() > 0);
    assert(path[path.length() - 1] == '/');
    this->basePath = path;
}

void Parser::setTplCount(uint tplCount) {
    assert(tplCount > 0);
    this->tplCount = tplCount;
}

void Parser::setFolders(const std::vector<std::string> &folders) {
    assert(!folders.empty());
    this->folders = folders;
}

void Parser::setIndices(const std::vector<uint> &indices) {
    assert(!indices.empty());
    this->indices = indices;
}