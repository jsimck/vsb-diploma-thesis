#include "template_parser.h"
#include <cassert>

int TemplateParser::idCounter = 0;

void TemplateParser::parse(std::vector<TemplateGroup> &groups, DatasetInfo &info) {
    // Checks
    assert(this->templateFolders.size() > 0);
    int parsedTemplatesCount = 0;

    // Reset dataset
    info.reset();

    // Parse
    for (auto &&tplName : this->templateFolders) {
        std::vector<Template> templates;

        // If indices are not null, parse specified ids
        if (this->indices) {
            parseTemplate(templates, info, tplName, this->indices);
        } else {
            parseTemplate(templates, info, tplName);
        }

        // Push to groups vector
        groups.push_back(TemplateGroup(tplName, templates));
        parsedTemplatesCount += templates.size();
        std::cout << "  |_ Parsed: " << tplName << ", templates size: " << templates.size() << std::endl;
    }

    std::cout << "  |_ Parsed total: " << parsedTemplatesCount << " templates" << std::endl;
}

void TemplateParser::parseTemplate(std::vector<Template> &templates, DatasetInfo &info, std::string tplName) {
    // Load obj_gt
    cv::FileStorage fs;
    fs.open(this->basePath + tplName + "/gt.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    for (int i = 0; i < this->tplCount; i++) {
        std::string index = "tpl_" + std::to_string(i);
        cv::FileNode objGt = fs[index];

        // Parse template gt file
        templates.push_back(parseGt(i, this->basePath + tplName, objGt, info));
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

void TemplateParser::parseTemplate(std::vector<Template> &templates, DatasetInfo &info, std::string tplName, std::unique_ptr<std::vector<int>> &indices) {
    // Load obj_gt
    cv::FileStorage fs;
    fs.open(this->basePath + tplName + "/gt.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    for (int i = 0; i < (*indices).size(); i++) {
        int tplIndex = (*indices)[i];
        std::string index = "tpl_" + std::to_string(tplIndex);
        cv::FileNode objGt = fs[index];

        // Parse template gt file
        templates.push_back(parseGt(tplIndex, this->basePath + tplName, objGt, info));
        this->idCounter++;
    }

    // Load obj_info
    fs.open(this->basePath + tplName + "/info.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    for (int i = 0; i < (*indices).size(); i++) {
        int tplIndex = (*indices)[i];
        std::string index = "tpl_" + std::to_string(tplIndex);
        cv::FileNode objGt = fs[index];

        // Parse template info file
        parseInfo(templates[i], objGt);
    }

    fs.release();
}

Template TemplateParser::parseGt(int index, std::string path, cv::FileNode &gtNode, DatasetInfo &info) {
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
    cv::Mat src = cv::imread(path + "/rgb/" + fileName + ".png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat srcDepth = cv::imread(path + "/depth/" + fileName + ".png", CV_LOAD_IMAGE_UNCHANGED);

    // Crop image using objBB
    src = src(objBB);
    srcDepth = srcDepth(objBB);

    // Convert to float
    src.convertTo(src, CV_32F, 1.0f / 255.0f);
    srcDepth.convertTo(srcDepth, CV_32F); // because of surface normal calculation, don't doo normalization

    // Find smallest object
    if (src.cols * src.rows < info.smallestTemplateSize.area()) {
        info.smallestTemplateSize.width = src.cols;
        info.smallestTemplateSize.height = src.rows;
    }

    // Find largest object
    if (src.cols * src.rows >= info.largestTemplateSize.area()) {
        info.largestTemplateSize.width = src.cols;
        info.largestTemplateSize.height = src.rows;
    }

    // Checks
    assert(!vObjBB.empty());
    assert(!vCamRm2c.empty());
    assert(!vCamTm2c.empty());
    assert(!src.empty());
    assert(!srcDepth.empty());

    // Matrix type checks
    assert(src.type() == 5); // CV_32FC1
    assert(srcDepth.type() == 5); // CV_32FC1

    return Template(
        this->idCounter, fileName, src, srcDepth, objBB,
        cv::Mat(3, 3, CV_32FC1, vCamRm2c.data()).clone(),
        cv::Vec3d(vCamTm2c[0], vCamTm2c[1], vCamTm2c[2])
    );
}

void TemplateParser::parseInfo(Template &tpl, cv::FileNode &infoNode) {
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

void TemplateParser::clearIndices() {
    this->indices = nullptr;
}

int TemplateParser::getIdCounter() {
    return idCounter;
}

void TemplateParser::setBasePath(std::string path) {
    assert(path.length() > 0);
    assert(path[path.length() - 1] == '/');
    this->basePath = path;
}

std::string TemplateParser::getBasePath() const {
    return this->basePath;
}

void TemplateParser::setTplCount(unsigned int tplCount) {
    assert(tplCount > 0);
    this->tplCount = tplCount;
}

unsigned int TemplateParser::getTplCount() const {
    return this->tplCount;
}

void TemplateParser::setTemplateFolders(const std::vector<std::string> &templateFolders) {
    this->templateFolders = templateFolders;
}

const std::vector<std::string> &TemplateParser::getTemplateFolders() const {
    return this->templateFolders;
}

void TemplateParser::setIndices(std::unique_ptr<std::vector<int>> &indices) {
    assert(indices->size() > 0);
    this->indices.swap(indices);
}

const std::unique_ptr<std::vector<int>> &TemplateParser::getIndices() const {
    return this->indices;
}