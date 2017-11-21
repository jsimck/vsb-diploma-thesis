#include "parser.h"
#include "../processing/processing.h"
#include "../objdetect/matcher.h"
#include "../objdetect/hasher.h"

void Parser::parse(std::string basePath, std::string modelsPath, std::vector<Template> &templates) {
    // Load obj_gt
    cv::FileStorage fs;
    fs.open(basePath + "/gt.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    // Parse diameters if empty
    if (diameters.empty()) {
        parseModelsInfo(modelsPath);
    }

    const auto size = static_cast<uint>((!indices.empty()) ? indices.size() : tplCount);
    for (uint i = 0; i < size; i++) {
        auto tplIndex = static_cast<uint>((!indices.empty()) ? indices[i] : i);
        std::string index = "tpl_" + std::to_string(tplIndex);
        cv::FileNode objGt = fs[index];

        // Parse template gt file
        templates.push_back(parseGt(tplIndex, basePath, objGt));
    }

    // Load obj_info
    fs.release();
    fs.open(basePath + "/info.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    for (uint i = 0; i < size; i++) {
        auto tplIndex = static_cast<uint>((!indices.empty()) ? indices[i] : i);
        std::string index = "tpl_" + std::to_string(tplIndex);
        cv::FileNode objGt = fs[index];

        // Parse template info file
        parseInfo(templates[i], objGt);
    }

    fs.release();
}

Template Parser::parseGt(uint index, const std::string &path, cv::FileNode &gtNode) {
    // Init template param matrices
    int id;
    std::vector<float> vCamRm2c, vCamTm2c;
    std::vector<int> vObjBB;

    // Nodes containing matrices and vectors to parseTemplate
    gtNode["obj_id"] >> id;
    gtNode["obj_bb"] >> vObjBB;
    gtNode["cam_R_m2c"] >> vCamRm2c;
    gtNode["cam_t_m2c"] >> vCamTm2c;
    float diameter = diameters[id];

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

    // Convert to gray and HSV
    cv::cvtColor(src, srcHSV, CV_BGR2HSV);
    cv::cvtColor(src, src, CV_BGR2GRAY);

    // Convert to float
    src.convertTo(src, CV_32F, 1.0f / 255.0f);

    // Find smallest object
    if (objBB.area() < criteria->info.smallestTemplate.area()) {
        criteria->info.smallestTemplate.width = objBB.width;
        criteria->info.smallestTemplate.height = objBB.height;
    }

    // Find largest object
    if (objBB.width >= criteria->info.maxTemplate.width) {
        criteria->info.maxTemplate.width = objBB.width;
    }
    if (objBB.height >= criteria->info.maxTemplate.height) {
        criteria->info.maxTemplate.height = objBB.height;
    }

    // Calculate quantizedGradients and quantizedNormals
    cv::Mat gradients, magnitudes;
    Processing::quantizedOrientationGradients(src, gradients, magnitudes);

    // Checks
    assert(!vObjBB.empty());
    assert(!vCamRm2c.empty());
    assert(!vCamTm2c.empty());
    assert(!src.empty());
    assert(!srcHSV.empty());
    assert(!srcDepth.empty());

    // Matrix type checks
    assert(src.type() == CV_32FC1);
    assert(srcDepth.type() == CV_16U);

    // TODO cleanup Template constructor (initializing empty normal matrix, since it needs to be done parseInfo) in  etc.
    return Template(
        index + (2000 * id), fileName, diameter, std::move(src), std::move(srcHSV), std::move(srcDepth),
        std::move(gradients), cv::Mat(), objBB, cv::Mat(3, 3, CV_32FC1, vCamRm2c.data()).clone(),
        cv::Vec3d(vCamTm2c[0], vCamTm2c[1], vCamTm2c[2])
    );
}

void Parser::parseInfo(Template &tpl, cv::FileNode &infoNode) {
    // Init template param matrices
    std::vector<float> vCamK;
    int elev, mode;

    // Parse train contained in info.yml
    infoNode["cam_K"] >> vCamK;
    infoNode["elev"] >> elev;
    infoNode["mode"] >> mode;

    // TODO max distance etc should be in criteria
    // Compute normals based on camK params
    Processing::quantizedNormals(tpl.srcDepth, tpl.quantizedNormals, vCamK[0], vCamK[4], 15000, 50);

    // Checks
    assert(!vCamK.empty());

    // Assign new train to template
    tpl.elev = elev;
    tpl.mode = mode;
    tpl.azimuth = 5 * ((tpl.id % 2000) % 72); // Training templates are sampled in 5 step azimuth
    tpl.camK = cv::Mat(3, 3, CV_32FC1, vCamK.data()).clone();
}

void Parser::parseModelsInfo(const std::string &modelsPath) {
    // Load modelsPath/info.yml
    cv::FileStorage fs;
    fs.open(modelsPath + "info.yml", cv::FileStorage::READ);
    assert(fs.isOpened());

    float diameter = 0;
    std::ostringstream oss;

    // Parse only diameters for now
    for (uint i = 0; i < modelCount; i++) {
        oss.str("");
        oss << "model_" << i;

        cv::FileNode modelNode = fs[oss.str()];
        modelNode["diameter"] >> diameter;
        diameters.push_back(diameter);
    }
}

