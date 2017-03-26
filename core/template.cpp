#include "template.h"

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
       << "votes: " << t.votes;

    return os;
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

void Template::voteUp() {
    votes++;
}

void Template::resetVotes() {
    votes = 0;
}

void Template::applyROI() {
    // Apply roi to both sources
    src = src(objBB);
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
}
