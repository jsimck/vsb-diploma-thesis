#include "camera.h"

namespace tless {
    float Camera::fx() const {
        return K.at<float>(0, 0);
    }

    float Camera::fy() const {
        return K.at<float>(1, 1);
    }

    float Camera::cx() const {
        return K.at<float>(0, 2);
    }

    float Camera::cy() const {
        return K.at<float>(1, 2);
    }

    cv::Vec3f Camera::v(int x, int width, int y, int height, float d) const {
        return cv::Vec3f(((x - (width / 2)) * d) / fx(), ((y + (height / 2)) * d) / fy(), d);
    }

    std::ostream &operator<<(std::ostream &os, const Camera &camera) {
        os << "K: " << camera.K.size().width << "x" << camera.K.size().height
           << " R: " << camera.R.size().width << "x" << camera.R.size().height
           << " t: " << camera.t.size().width << "x" << camera.t.size().height
           << " elev: " << camera.elev << " azimuth: " << camera.azimuth << " mode: " << camera.mode;

        return os;
    }

    void operator>>(const cv::FileNode &node, Camera &t) {
        node["K"] >> t.K;
        node["R"] >> t.R;
        node["t"] >> t.t;
        node["elev"] >> t.elev;
        node["mode"] >> t.mode;
        node["azimuth"] >> t.azimuth;
    }

    cv::FileStorage &operator<<(cv::FileStorage &fs, const Camera &t) {
        fs << "{";
        fs << "K" << t.K;
        fs << "R" << t.R;
        fs << "t" << t.t;
        fs << "elev" << t.elev;
        fs << "mode" << t.mode;
        fs << "azimuth" << t.azimuth;
        fs << "}";

        return fs;
    }
}