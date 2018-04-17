#ifndef VSB_SEMESTRAL_PROJECT_CAMERA_H
#define VSB_SEMESTRAL_PROJECT_CAMERA_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <utility>
#include <ostream>

namespace tless {
    class Camera {
    public:
        cv::Mat R; //!< Rotation matrix R_m2c
        cv::Mat t; //!< Translation vector t_m2c
        cv::Mat K; //!< Intrinsic camera matrix K
        int elev = 0;
        int azimuth = 0;
        int mode = 0;

        Camera() = default;

        /**
         * @brief Returns x-focal length from K camera matrix.
         *
         * @return x-focal length
         */
        float fx() const;

        /**
         * @brief Returns y-focal length from K camera matrix.
         *
         * @return y-focal length
         */
        float fy() const;

        /**
         * @brief Computes vector in camera space on screen (x, y) coordinates.
         *
         * @param[in] x      Screen x position
         * @param[in] width  Screen width
         * @param[in] y      Screen y position
         * @param[in] height Screen height
         * @param[in] d      Depth value (z) at given (x, y)
         * @return           vector from camera to screen location (x, y) in camera space
         */
        cv::Vec3f v(int x, int width, int y, int height, float d) const;

        friend void operator>>(const cv::FileNode &node, Camera &t);
        friend cv::FileStorage &operator<<(cv::FileStorage &fs, const Camera &t);
        friend std::ostream &operator<<(std::ostream &os, const Camera &camera);
    };
}

#endif