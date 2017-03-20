#include "template_parser.h"
#include <cassert>

int TemplateParser::idCounter = 0;

TemplateParser::TemplateParser(const std::string basePath, unsigned int tplCount) {
    setBasePath(basePath);
    setTplCount(tplCount);
}

void TemplateParser::parse(std::vector<TemplateGroup> &groups, std::vector<std::string> tplNames) {
    for (auto &&tplName : tplNames) {
        std::vector<Template> templates;
        parseTemplate(templates, tplName);

        // Push to groups vector
        groups.push_back(TemplateGroup(tplName, templates));
        std::cout << "Parsed: " << tplName << ", templates size: " << templates.size() << std::endl;
    }
}

void TemplateParser::parseTemplate(std::vector<Template> &templates, std::string tplName) {
    // Load obj_gt
    cv::FileStorage fs;
    fs.open(this->basePath + tplName + "/gt.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    for (int i = 0; i < this->tplCount; i++) {
        std::string index = "tpl_" + std::to_string(i);
        cv::FileNode objGt = fs[index];

        // Parse template gt file
        templates.push_back(parseGt(i, this->basePath + tplName, objGt));
        this->idCounter++;
    }

    // Load obj_info
    fs.open(this->basePath + tplName + "/info.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    for (int i = 0; i < this->tplCount; i++) {
        std::string index = "tpl_" + std::to_string(i);
        cv::FileNode objGt = fs[index];

        // Parse template info file
        parseInfo(templates[i], objGt);
    }

    fs.release();
}

void TemplateParser::parseTemplate(std::vector<Template> &templates, std::string tplName, std::vector<int> indices) {
    // Load obj_gt
    cv::FileStorage fs;
    fs.open(this->basePath + tplName + "/gt.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    for (int i = 0; i < indices.size(); i++) {
        int tplIndex = indices[i];
        std::string index = "tpl_" + std::to_string(tplIndex);
        cv::FileNode objGt = fs[index];

        // Parse template gt file
        templates.push_back(parseGt(tplIndex, this->basePath + tplName, objGt));
        this->idCounter++;
    }

    // Load obj_info
    fs.open(this->basePath + tplName + "/info.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    for (int i = 0; i < indices.size(); i++) {
        int tplIndex = indices[i];
        std::string index = "tpl_" + std::to_string(tplIndex);
        cv::FileNode objGt = fs[index];

        // Parse template info file
        parseInfo(templates[i], objGt);
    }

    fs.release();
}

Template TemplateParser::parseGt(int index, std::string path, cv::FileNode &gtNode) {
    // Init template param matrices
    std::vector<double> vCamRm2c, vCamTm2c;
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
    cv::Mat src = cv::imread(path + "/rgb/" + fileName + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat srcDepth = cv::imread(path + "/depth/" + fileName + ".png", CV_LOAD_IMAGE_UNCHANGED);

    // Crop image using objBB
    src = src(objBB);
    srcDepth = srcDepth(objBB);

    // Convert to double
    src.convertTo(src, CV_64FC1, 1.0 / 255.0);
    srcDepth.convertTo(srcDepth, CV_64FC1, 1.0 / 65536.0); // 16-bit

    return Template(
        this->idCounter, fileName, src, srcDepth, objBB,
        cv::Mat(3, 3, CV_64FC1, vCamRm2c.data()).clone(),
        cv::Vec3d(vCamTm2c[0], vCamTm2c[1], vCamTm2c[2])
    );
}

void TemplateParser::parseInfo(Template &tpl, cv::FileNode &infoNode) {
    // Init template param matrices
    std::vector<double> vCamK;
    int elev, mode;

    // Parse params contained in info.yml
    infoNode["cam_K"] >> vCamK;
    infoNode["elev"] >> elev;
    infoNode["mode"] >> mode;

    // Assign new params to template
    tpl.elev = elev;
    tpl.mode = mode;
    tpl.camK = cv::Mat(3, 3, CV_64FC1, vCamK.data()).clone();
}

void TemplateParser::setBasePath(std::string path) {
    if (path.compare(path.size() - 1, 1, "/") != 0) {
        path = path + "/";
    }

    this->basePath = path;
}

std::string TemplateParser::getBasePath() {
    return this->basePath;
}

void TemplateParser::setTplCount(unsigned int tplCount) {
    this->tplCount = tplCount;
}

unsigned int TemplateParser::getTplCount() {
    return this->tplCount;
}