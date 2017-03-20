#include "template_parser.h"

int TemplateParser::idCounter = 0;

TemplateParser::TemplateParser(std::string basePath, int tplCount) {
    setBasePath(basePath);
    setTplCount(tplCount);
}

void TemplateParser::parseTemplate(std::vector<Template> &templates, std::string tplName) {
    // Load obj_info
    cv::FileStorage fs;
    fs.open(this->basePath + tplName + "/obj_info.yml", cv::FileStorage::READ);

    for (int i = 0; i < this->tplCount; i++) {
        std::string index = "tpl_" + std::to_string(i);
        cv::FileNode objInfo = fs[index];

        // Parse template
        templates.push_back(parseTemplate(i, this->basePath + tplName, objInfo));
    }

    fs.release();
}

void TemplateParser::parseTemplate(std::vector<Template> &templates, std::string tplName, const int *indices, int indicesLength) {
    // Load obj_info
    cv::FileStorage fs;
    fs.open(this->basePath + tplName + "/obj_info.yml", cv::FileStorage::READ);

    for (int i = 0; i < indicesLength; i++) {
        int tplIndex = indices[i];
        std::string index = "tpl_" + std::to_string(tplIndex);
        cv::FileNode objInfo = fs[index];

        // Parse template
        templates.push_back(parseTemplate(tplIndex, this->basePath + tplName, objInfo));
        this->idCounter++;
    }

    fs.release();
}

Template TemplateParser::parseTemplate(int index, std::string path, cv::FileNode &node) {
    // Init template param matrices
    std::vector<double> vCamK, vCamRm2c, vCamTm2c;
    std::vector<int> vObjBB;

    // Nodes containing matrices and vectors to parseTemplate
    node["obj_bb"] >> vObjBB;
    node["cam_K"] >> vCamK;
    node["cam_R_m2c"] >> vCamRm2c;
    node["cam_t_m2c"] >> vCamTm2c;

    // Get other params from .yml file
    int elev = node["elev"];
    int mode = node["mode"];

    // Parse objBB and cam matrices and translation vector
    cv::Rect objBB(vObjBB[0], vObjBB[1], vObjBB[2], vObjBB[3]);
    cv::Mat camK(3, 3, CV_64FC1, vCamK.data());
    cv::Mat camRm2c(3, 3, CV_64FC1, vCamRm2c.data());
    cv::Vec3d camTm2c(vCamTm2c[0], vCamTm2c[1], vCamTm2c[2]);

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

    return Template(this->idCounter, fileName, src, srcDepth, objBB, camK.clone(), camRm2c.clone(), camTm2c, elev, mode);
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

int TemplateParser::getTplCount() {
    return this->tplCount;
}

void TemplateParser::setTplCount(int tplCount) {
    this->tplCount = (tplCount < 0) ? 0 : tplCount;
}
