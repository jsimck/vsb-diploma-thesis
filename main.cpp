#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>

#define TPL_COUNT 1296

struct Template {
    Template(std::string fileName, cv::Rect bounds) : fileName(fileName), bounds(bounds) {}
    std::string fileName;
    cv::Rect bounds;
};

Template parseTemplate(int index, cv::FileNode &node) {
    // Get template bounding box
    std::vector<int> objBB;
    node["obj_bb"] >> objBB;

    // Create filename from index
    std::stringstream ss;
    ss << std::setw(4) << std::setfill('0') << index;

    return Template(ss.str(), cv::Rect(objBB[0], objBB[1], objBB[2], objBB[3]));
}

void parseObjInfo(std::vector<Template> &templates) {
    // Load obj_info
    cv::FileStorage fs;
    fs.open("data/obj_05/obj_info.yml", cv::FileStorage::READ);

    for (int i = 0; i < TPL_COUNT; i++) {
        std::string index = "tpl_" + std::to_string(i);
        cv::FileNode objInfo = fs[index];

        // Parse template
        templates.push_back(parseTemplate(i, objInfo));
    }

    fs.release();
};

void generateNGTrainingSet() {
    // Parse objInfo
    std::vector<Template> templates;
    parseObjInfo(templates);

    // Create file storage
    cv::FileStorage fsWrite;

    for (int i = 0; i < TPL_COUNT; i++) {
        // Load template and crop
        cv::Mat templ = cv::imread("data/obj_05/rgb/" + templates[i].fileName + ".png", CV_LOAD_IMAGE_GRAYSCALE);
        templ = templ(templates[i].bounds);
        templ.convertTo(templ, CV_32FC1, 1.0 / 255.0);

        // Run sobel
        cv::Mat templSobel_x, templSobel_y, templSobel;
        cv::Sobel(templ, templSobel_x, -1, 1, 0);
        cv::Sobel(templ, templSobel_y, -1, 0, 1);
        cv::addWeighted(templSobel_x, 0.5, templSobel_y, 0.5, 0, templSobel);

        // Calculate helper variables
        bool biggerHeight = templ.rows > templ.cols;
        int size = biggerHeight ? templ.rows : templ.cols;
        int offset = biggerHeight ? (size - templ.cols) / 2 : (size - templ.rows) / 2;
        int top = biggerHeight ? 0 : offset;
        int left = biggerHeight ? offset : 0;

        cv::imshow("BING", templSobel);
        cv::waitKey(0);

        // Copy to keep ratio and Resize
        cv::Mat dest = cv::Mat::zeros(size, size, CV_32FC1);
        templSobel.copyTo(dest(cv::Rect(left, top, templSobel.cols, templSobel.rows)));
//        cv::imshow("Image resized", dest);
//        cv::waitKey(0);
        cv::resize(templSobel, templSobel, cv::Size(8, 8));

        // Save file
        fsWrite.open("data/objectness/ObjNessB2W8I." + templates[i].fileName + ".yml.gz", cv::FileStorage::WRITE);
        fsWrite << "ObjNessB2W8I" + templates[i].fileName << templSobel;
    }

    // Release
    fsWrite.release();
}

int main() {
    generateNGTrainingSet();
    return 0;



    /*
    cv::Mat image = cv::imread("data/scene_05/rgb/0005.png");
    image.convertTo(image, CV_32FC3, 1.0/255.0);

    cv::saliency::ObjectnessBING objectnessBING;
    std::vector<cv::Vec4i> objectnessBoundingBox;
    objectnessBING.setTrainingPath("data/objectness/");

    if (objectnessBING.computeSaliency(image, objectnessBoundingBox) ) {
        std::vector<float> values = objectnessBING.getobjectnessValues();

        printf("detected candidates: %d\n", objectnessBoundingBox.size());
        printf("scores: %d\n", values.size());

        // The result are sorted by objectnessBING. We uonly use the first 20 boxes here.
        for (int i = 0; i < 20; i++) {

            cv::Mat clone = image.clone();
            cv::Vec4i bb = objectnessBoundingBox[i];
            printf("index=%d, value=%f\n", i, values[i]);
            rectangle(image, cv::Point(bb[0], bb[1]), cv::Point(bb[2], bb[3]), cv::Scalar(0, 0, 255), 4);

            rectangle(image, cv::Point(bb[0], bb[1]), cv::Point(bb[2], bb[3]), cv::Scalar(0, 0, 255), 4);

            char label[256];
            sprintf(label, "#%d", i+1);
            putText(clone, label, cv::Point(bb[0], bb[1]+30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 3);

            char filename[256];
            sprintf(filename, "bing_%05d.jpg", i);
        }

        cv::imshow("Saliency result", image);
        cv::waitKey(0);
    }
    */








    // Load template
    cv::Mat templ = cv::imread("data/obj_05/depth/0000.png", CV_LOAD_IMAGE_GRAYSCALE);
    templ = templ(cv::Rect(114, 149, 169, 99));
    int threshold = 100;

    // Run sobel over the image
    cv::Mat templSobel_x, templSobel_y, templSobel;
    cv::Sobel(templ, templSobel_x, -1, 1, 0);
    cv::Sobel(templ, templSobel_y, -1, 0, 1);
    cv::addWeighted(templSobel_x, 0.5, templSobel_y, 0.5, 0, templSobel);
    cv::threshold(templSobel, templSobel, 0, 255, CV_THRESH_BINARY);

    // Count edgels in template
    int templEdgels = 0;
    for (int y = 0; y < templSobel.rows; y++) {
        for (int x = 0; x < templSobel.cols; x++) {
            if (templSobel.at<uchar>(y, x) > threshold) {
                templEdgels++;
            }
        }
    }
    templEdgels *= 0.3;

    std::cout << "Template edgels: " << templEdgels << std::endl;

    // Calculate edgels in scene over each sliding window
    cv::Mat sceneRGB = cv::imread("data/scene_05/rgb/0000.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat scene = cv::imread("data/scene_05/depth/0000.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat sceneSobel_x, sceneSobel_y, sceneSobel;

    cv::Sobel(scene, sceneSobel_x, -1, 1, 0);
    cv::Sobel(scene, sceneSobel_y, -1, 0, 1);

    cv::addWeighted(sceneSobel_x, 0.5, sceneSobel_y, 0.5, 0, sceneSobel);
//    cv::threshold(sceneSobel, sceneSobel, 0, 255, CV_THRESH_BINARY);

    int sizeX = 169;
    int sizeY = 99;
    int step = 20;

    std::vector<cv::Rect> windows;
    for (int y = 0; y < sceneSobel.rows - sizeY; y += step) {
        for (int x = 0; x < sceneSobel.cols - sizeX; x += step) {

            int sceneEdgels = 0;
            for (int yy = y; yy < y + sizeY; yy++) {
                for (int xx = y; xx < x + sizeX; xx++) {
                    if (templSobel.at<uchar>(y, x) > threshold) {
                        sceneEdgels++;
                    }
                }
            }

            if (sceneEdgels > templEdgels) {
                windows.push_back(cv::Rect(x, y, x + sizeX, y + sizeY));
            }
        }
    }

    for (int i = 0; i < windows.size(); i++) {
        cv::Rect window = windows.at(i);
        cv::rectangle(sceneRGB, cv::Point(window.x, window.y), cv::Point(window.width, window.height), 255);
    }

    cv::imshow("Sobel Scene", sceneSobel);
    cv::imshow("Sobel Template", templSobel);
    cv::imshow("Scene", sceneRGB);
    cv::imshow("Depth", scene);
    cv::waitKey(0);

    return 0;
}