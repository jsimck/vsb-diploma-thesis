#include "template.h"

void Template::voteUp() {
    votes++;
}

void Template::resetVotes() {
    votes = 0;
}

void Template::applyROI() {
    // Apply roi to both sources
    src = src(objBB);
    srcHSV = srcHSV(objBB);
    srcDepth = srcDepth(objBB);
}

void Template::resetROI() {
    // Locate ROI
    cv::Point offset;
    cv::Size size;
    src.locateROI(size, offset);

    // Set to original [disable ROI]
    src.adjustROI(offset.y, size.height - src.rows, offset.x, size.width - src.cols);
    srcDepth.adjustROI(offset.y, size.height - src.rows, offset.x, size.width - src.cols);
    srcHSV.adjustROI(offset.y, size.height - src.rows, offset.x, size.width - src.cols);
}

bool Template::operator==(const Template &rhs) const {
    return id == rhs.id &&
           fileName == rhs.fileName &&
           elev == rhs.elev &&
           mode == rhs.mode;
}

bool Template::operator!=(const Template &rhs) const {
    return !(rhs == *this);
}

std::ostream &operator<<(std::ostream &os, const Template &t) {
    os << "Template ID: " << t.id << std::endl
       << "fileName: " << t.fileName << std::endl
       << "src (size): " << t.src.size()  << std::endl
       << "srcDepth (size): " << t.srcDepth.size() << std::endl
       << "objBB: " << t.objBB  << std::endl
       << "camK: " << t.camK  << std::endl
       << "camRm2c: " << t.camRm2c << std::endl
       << "camTm2c: " << t.camTm2c  << std::endl
       << "elev: " << t.elev  << std::endl
       << "mode: " << t.mode << std::endl
       << "minVotesPerTemplate: " << t.votes << std::endl
       << "depthMedian: " << t.features.depthMedian << std::endl
       << "orientationGradients size: " << t.features.orientationGradients.size() << std::endl
       << "surfaceNormals size: " << t.features.surfaceNormals.size() << std::endl
       << "depth size: " << t.features.depth.size() << std::endl
       << "color size: " << t.features.color.size() << std::endl;

    return os;
}