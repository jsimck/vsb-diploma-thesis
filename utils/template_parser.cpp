#include "template_parser.h"

TemplateParser::TemplateParser(std::string basePath, int tplCount) {
    this->basePath = basePath;
    this->tplCount = tplCount;
}

void TemplateParser::parse(std::vector<Template> &templates) {
    // Load obj_info
    cv::FileStorage fs;
    fs.open(this->basePath + "/obj_info.yml", cv::FileStorage::READ);

    for (int i = 0; i < this->tplCount; i++) {
        std::string index = "tpl_" + std::to_string(i);
        cv::FileNode objInfo = fs[index];

        // Parse template
        templates.push_back(parseTemplate(i, objInfo));
    }

    fs.release();
}

void TemplateParser::parseRange(std::vector<Template> &templates, int *indices, int indicesLength) {
    // Load obj_info
    cv::FileStorage fs;
    fs.open(this->basePath + "/obj_info.yml", cv::FileStorage::READ);

    for (int i = 0; i < indicesLength; i++) {
        int tplIndex = indices[i];
        std::string index = "tpl_" + std::to_string(tplIndex);
        cv::FileNode objInfo = fs[index];

        // Parse template
        templates.push_back(parseTemplate(tplIndex, objInfo));
    }

    fs.release();
}

Template TemplateParser::parseTemplate(int index, cv::FileNode &node) {
    // Get template bounding box
    std::vector<int> objBB;
    node["obj_bb"] >> objBB;

    // Create filename from index
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << index;
    std::string fileName = ss.str();

    // Parse obj bounds
    cv::Rect bounds = cv::Rect(objBB[0], objBB[1], objBB[2], objBB[3]);

    // Load image
    cv::Mat src = cv::imread(this->basePath + "/rgb/" + fileName + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat srcDepth = cv::imread(this->basePath + "/depth/" + fileName + ".png", CV_LOAD_IMAGE_UNCHANGED);

    // Crop image using bounds
    src = src(bounds);
    srcDepth = srcDepth(bounds);

    // Convert to double
    src.convertTo(src, CV_64FC1, 1.0 / 255.0);
    srcDepth.convertTo(srcDepth, CV_64FC1, 1.0 / 65536.0); // 16-bit

    return Template(fileName, bounds, src, srcDepth);
}

void TemplateParser::setBasePath(std::string path) {
    this->basePath = path;
}

std::string TemplateParser::getBasePath() {
    return this->basePath;
}

int TemplateParser::getTplCount() {
    return this->tplCount;
}

void TemplateParser::setTplCount(int tplCount) {
    this->tplCount = tplCount;
}
