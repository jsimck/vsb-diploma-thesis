#include "result.h"

namespace tless {
    Result::Result(const Match &m) {
        objBB = m.normObjBB;
        objId = m.t->objId;
        id = m.t->id;
        score = m.score;
        scale = m.scale;
        camera = m.t->camera;
    }

    void operator>>(const cv::FileNode &node, Result &r) {
        node["objBB"] >> r.objBB;
        node["objId"] >> r.objId;
        node["id"] >> r.id;
        node["score"] >> r.score;
        node["scale"] >> r.scale;
        node["camera"] >> r.camera;
    }

    cv::FileStorage &operator<<(cv::FileStorage &fs, const Result &r) {
        fs << "{";
        fs << "objBB" << r.objBB;
        fs << "objId" << r.objId;
        fs << "id" << r.id;
        fs << "score" << r.score;
        fs << "scale" << r.scale;
        fs << "camera" << r.camera;
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