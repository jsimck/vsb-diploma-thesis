#include "result.h"

namespace tless {
    Result::Result(const Match &m) {
        objBB = m.normObjBB;
        objId = m.t->objId;
    }

    void operator>>(const cv::FileNode &node, Result &r) {
        node["objId"] >> r.objId;
        node["objBB"] >> r.objBB;
    }

    cv::FileStorage &operator<<(cv::FileStorage &fs, const Result &r) {
        fs << "{";
        fs << "objId" << r.objId;
        fs << "objBB" << r.objBB;
        fs << "}";

        return fs;
    }

    std::ostream &operator<<(std::ostream &os, const Result &r) {
        os << "objId: " << r.objId << ", objBB: " << r.objBB << std::endl;

        return os;
    }

    float Result::jaccard(const cv::Rect &r1) const {
        return (objBB & r1).area() / static_cast<float>((objBB | r1).area());
    }
}