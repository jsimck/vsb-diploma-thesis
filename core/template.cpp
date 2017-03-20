#include "template.h"

std::ostream &operator<<(std::ostream &os, const Template &t) {
    os << "Template ID: " << 0 << std::endl // TODO id
       << "fileName: " << t.fileName << std::endl
       << "src (size): " << t.src.size()  << std::endl
       << "srcDepth (size): " << t.srcDepth.size() << std::endl
       << "objBB: " << t.objBB  << std::endl
       << "camK: " << t.camK  << std::endl
       << "camRm2c: " << t.camRm2c << std::endl
       << "camTm2c: " << t.camTm2c  << std::endl
       << "elev: " << t.elev  << std::endl
       << "mode: " << t.mode;

    return os;
}
