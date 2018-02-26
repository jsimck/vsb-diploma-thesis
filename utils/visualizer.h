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

namespace tless {
    class Visualizer {
    private:
        const int KEY_UP = 0, KEY_DOWN = 1, KEY_LEFT = 2, KEY_RIGHT = 3, KEY_SPACEBAR = 32, KEY_ENTER = 13;
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
        void windowCandidates(const cv::Mat &src, cv::Mat &dst, Window &window);

    public:
        Visualizer(cv::Ptr<ClassifierCriteria> criteria, const std::string &templatesPath = "data/108x108/")
                : criteria(criteria), templatesPath(templatesPath) {}

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
        void windowsCandidates(const Scene &scene, std::vector<Window> &windows, int wait = 0, const char *title = nullptr);

        /**
         * @brief Vizualizes window locations after objectness detection has been performed.
         *
         * @param[in] scene   Scene object we want to vizualize window locations on
         * @param[in] windows Array of sliding windows that passed objectness detection
         * @param[in] wait    Optional wait time in waitKey() function
         * @param[in] title   Optional image window title
         */
        void objectness(const Scene &scene, std::vector<Window> &windows, int wait = 0, const char *title = nullptr);

        /**
         * @brief Vizualizes template feature points after template training.
         *
         * @param[in] t     Template object with generated feature points
         * @param[in] wait  Optional wait time in waitKey() function
         * @param[in] title Optional image window title
         */
        void tplFeaturePoints(const Template &t, int wait = 0, const char *title = nullptr);

        // TODO REFACTOR ---->>>
        // Objectness
        static void visualizeMatches(cv::Mat &scene, float scale, std::vector<Match> &matches,
                                     const std::string &templatesPath = "data/",
                                     int wait = 0, const char *title = nullptr);

        // Tests
        static bool visualizeTests(Template &tpl, const cv::Mat &sceneHSV, const cv::Mat &sceneDepth, Window &window,
                                   std::vector<cv::Point> &stablePoints, std::vector<cv::Point> &edgePoints,
                                   int patchOffset, std::vector<int> &scoreI, std::vector<int> &scoreII,
                                   std::vector<int> &scoreIII, std::vector<int> &scoreIV, std::vector<int> &scoreV,
                                   int pointsCount, int minThreshold, int currentTest, bool continuous,
                                   const std::string &templatesPath, int wait, const char *title);
    };
}

#endif
