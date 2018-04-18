#ifndef VSB_SEMESTRAL_PROJECT_VISUALIZER_H
#define VSB_SEMESTRAL_PROJECT_VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <memory>
#include "../core/window.h"
#include "../core/template.h"
#include "../core/classifier_criteria.h"
#include "../core/hash_table.h"
#include "../core/match.h"
#include "../core/scene.h"
#include "../glcore/mesh.h"
#include "../glcore/shader.h"

namespace tless {
    class Visualizer {
    private:
        const int SETTINGS_GRID = 0, SETTINGS_TITLE = 1, SETTINGS_INFO = 2, SETTINGS_FEATURE_POINT_STYLE = 3,
            SETTINGS_FEATURE_POINT = 4, SETTINGS_TPL_OVERLAY = 5;

        std::unordered_map<int, bool> settings;
        cv::Ptr<ClassifierCriteria> criteria;
        std::string templatesPath;

        /**
         * @brief Dynamic image loading for Templates.
         *
         * @param[in] t     Input template, used to compute path based on it's id and filename
         * @param[in] flags Color of the loaded source image (CV_LOAD_IMAGE_COLOR, CV_LOAD_IMAGE_UNCHANGED)
         * @return          Loaded image
         */
        cv::Mat loadTemplateSrc(const Template &t, int flags = CV_LOAD_IMAGE_COLOR);

        /**
         * @brief Vizualizes candidates for given window along with matched triplets and number of votes.
         *
         * @param[in]  src    8-bit rgb image of the scene we want to vizualize hashing on
         * @param[out] dst    Destination image annotated with current window
         * @param[in]  window Sliding window that passed hashing verification
         */
#ifdef VIZ_HASHING
        void windowCandidates(const cv::Mat &src, cv::Mat &dst, Window &window);
#endif

    public:
        static const int KEY_UP = 0, KEY_DOWN = 1, KEY_LEFT = 2, KEY_RIGHT = 3, KEY_SPACEBAR = 32,
                KEY_ENTER = 13, KEY_G = 103, KEY_S = 115, KEY_K = 107, KEY_T = 116, KEY_I = 105,
                KEY_C = 99, KEY_V = 118, KEY_L = 108, KEY_J = 106;

        Visualizer(cv::Ptr<ClassifierCriteria> criteria, const std::string &templatesPath = "data/108x108/primesense/");

        /**
         * @brief Draws label with surrounded background for better visibility on any source image.
         *
         * @param[in,out] dst       8-bit 3-channel RGB source image
         * @param[in]     label     Text that should be displayed
         * @param[in]     origin    Bottom left corner of the text, where it should be drawn
         * @param[in]     scale     Font scale (size)
         * @param[in]     padding   Padding inside text box
         * @param[in]     thickness Font thickness
         * @param[in]     fColor    Foreground color
         * @param[in]     bColor    Background color
         * @param[in]     fontFace  OpenCV Font Face
         */
        static void label(cv::Mat &dst, const std::string &label, const cv::Point &origin, double scale = 0.4,
                          int padding = 1, int thickness = 1, cv::Scalar fColor = cv::Scalar(255, 255, 255),
                          cv::Scalar bColor = cv::Scalar(0, 0, 0), int fontFace = CV_FONT_HERSHEY_SIMPLEX);

        /**
         * @brief Vizualizes candidates for given window array along with matched triplets and number of votes.
         *
         * @param[in] scene   Scene object we want to vizualize hashing on
         * @param[in] windows Array of sliding windows that passed hashing verification and contain candidates
         * @param[in] wait    Optional wait time in waitKey() function
         * @param[in] title   Optional image window title
         */
#ifdef VIZ_HASHING
        void windowsCandidates(const ScenePyramid &scene, std::vector<Window> &windows, int wait = 0,
                               const char *title = nullptr);
#else
        void windowsCandidates(const ScenePyramid &scene, std::vector<Window> &windows, int wait = 0,
                               const char *title = nullptr) { void(); }
#endif

        /**
         * @brief Vizualizes window locations after objectness detection has been performed.
         *
         * @param[in] scene   Scene object we want to vizualize window locations on
         * @param[in] windows Array of sliding windows that passed objectness detection
         * @param[in] wait    Optional wait time in waitKey() function
         * @param[in] title   Optional image window title
         */
#ifdef DVIZ_OBJECTNESS
        void objectness(const ScenePyramid &scene, std::vector<Window> &windows, int wait = 0, const char *title = nullptr);
#else
        void objectness(const ScenePyramid &scene, std::vector<Window> &windows, int wait = 0, const char *title = nullptr) { void(); }
#endif

        /**
         * @brief Vizualizes template feature points after template training.
         *
         * @param[in] t     Template object with generated feature points
         * @param[in] wait  Optional wait time in waitKey() function
         * @param[in] title Optional image window title
         */
#ifdef VIZ_TPL_FEATURES
        void tplFeaturePoints(const Template &t, int wait = 0, const char *title = nullptr);
#else
        void tplFeaturePoints(const Template &t, int wait = 0, const char *title = nullptr) { void(); }
#endif

        /**
         * @brief Vizualizes template matching features during template matching.
         *
         * @param[in] t            Template object with generated feature points
         * @param[in] features     Array of feature points + validity whether it was matched or not
         * @param[in] highlight    Index of source image to highlight feature points on (rgb, depth, normals, gradients, hue)
         * @param[in] patchOffset  Patch offset, e.g. area around feature point to look for match
         * @param[in] wait         Optional wait time in waitKey() function
         * @param[in] title        Optional image window title
         */
#if defined(VIZ_MATCHING) || defined(VIZ_NMS)
        void tplMatch(Template &t, const std::vector<std::pair<cv::Point, int>> &features,
                      int highlight, int patchOffset, int wait = 0, const char *title = nullptr);
#else
        void tplMatch(Template &t, const std::vector<std::pair<cv::Point, int>> &features,
                      int highlight, int patchOffset, int wait = 0, const char *title = nullptr) { void(); }
#endif

        /**
         * @brief Vizualizes matched features between scene and current candidate.
         *
         * @param[in] scene          Scene to visualize feature match on
         * @param[in] candidate      Current candidate we want to compare with the scene
         * @param[in] windows        Sliding windows that passed hashing verification
         * @param[in] currentIndex   Index of a currently processed window
         * @param[in] candidateIndex Index of a currently processed candidate
         * @param[in] scores         Array of matched scores and feature points
         * @param[in] patchOffset    Patch offset, e.g. area around feature point to look for match
         * @param[in] minThreshold   Minimum number of points that should match, to continue with other tests
         * @param[in] wait           Optional wait time in waitKey() function
         * @param[in] title          Optional image window title
         *
         * @return true/false used for navigation
         */
#ifdef VIZ_MATCHING
        bool matching(const ScenePyramid &scene, Template &candidate, std::vector<Window> &windows, int &currentIndex,
                      int &candidateIndex, const std::vector<std::vector<std::pair<cv::Point, int>>> &scores,
                      int patchOffset, int minThreshold, int wait = 0, const char *title = nullptr);
#else
        bool matching(const ScenePyramid &scene, Template &candidate, std::vector<Window> &windows, int &currentIndex,
                      int &candidateIndex, const std::vector<std::vector<std::pair<cv::Point, int>>> &scores,
                      int patchOffset, int minThreshold, int wait = 0, const char *title = nullptr) { return false; }
#endif

        /**
         * @brief Vizualizes final matches after they all passed through the cascade.
         *
         * @param[in] scene   Input scene we want to vizualize matches on
         * @param[in] matches Array of final matched matches
         * @param[in] wait    Optional wait time in waitKey() function
         * @param[in] title   Optional image window title
         */
#ifdef VIZ_RESULTS
        void matches(const ScenePyramid &scene, std::vector<Match> &matches, int wait = 0, const char *title = nullptr);
#else
        void matches(const ScenePyramid &scene, std::vector<Match> &matches, int wait = 0, const char *title = nullptr) { void(); }
#endif

        /**
         * @brief Vizualizes final matches pre non-maxima-suppression applied
         *
         * @param[in] scene   Input scene we want to vizualize matches on
         * @param[in] matches Array of final matched matches
         * @param[in] wait    Optional wait time in waitKey() function
         * @param[in] title   Optional image window title
         */
#ifdef VIZ_NMS
        void preNonMaxima(const ScenePyramid &scene, std::vector<Match> &matches, int wait = 0,
                          const char *title = nullptr);
#else
        void preNonMaxima(const ScenePyramid &scene, std::vector<Match> &matches, int wait = 0,
                          const char *title = nullptr) { void(); }
#endif
    };
}

#endif
