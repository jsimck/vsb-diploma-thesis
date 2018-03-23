#include <iostream>
#include <opencv2/opencv.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/ext.hpp>

#include "objdetect/classifier.h"
#include "processing/processing.h"
#include "utils/converter.h"
#include "glcore/shader.h"
#include "glcore/mesh.h"
#include "utils/glutils.h"

// Settings
const unsigned int SCR_WIDTH = 400;
const unsigned int SCR_HEIGHT = 400;
const float clipNear = 100;
const float clipFar = 2000;

using namespace tless;

int main() {
    // GLFW init and config
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    // GLFW window creation
    GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);

    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwHideWindow(window);
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);

    // Init Glew after GLFW init
    if (glewInit()) {
        std::cerr << "Failed to initialize GLXW" << std::endl;
        return 1;
    }

    // Load shader and mesh
    Shader shader("data/shaders/depth.vert", "data/shaders/depth.frag");
    Mesh mesh("data/models/obj_07.ply");

    // PMV data
    glm::mat3 K(
            1076.74064739f, 0, 203.98264967f,
            0, 1075.17825536f, 239.59181836f,
            0, 0, 1
    );
    glm::vec3 t(-4.06668495f, -18.75499854f, 634.87406861f);
    glm::mat3 R(
            -0.93424870f, -0.35660872f, -0.00292517f,
            -0.19815880f, 0.52592294f, -0.82712632f,
            0.29649935f, -0.77216270f, -0.56200858f
    );

    // PVM initialization
    glm::mat4 MMatrix;
    glm::mat4 VMatrix = vMat(R, t);
    glm::mat4 NormalMatrix = nMat(MMatrix, VMatrix);
    glm::mat4 PMatrix = pMat(K, 0, 0, SCR_WIDTH, SCR_HEIGHT, clipNear, clipFar, WindowCoords::Y_UP);
    glm::mat4 MVMatrix = mvMat(MMatrix, VMatrix);
    glm::mat4 MVPMatrix = mvpMat(MMatrix, VMatrix, PMatrix);

    // OpenGL settings
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);

    /// Init frame buffer
    GLuint frameBuffer;
    glGenFramebuffers(1, &frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

    // FB texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0); // Bind to frame buffer

    // The depth buffer
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

    // Always check that our framebuffer is ok€
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << glCheckFramebufferStatus(GL_FRAMEBUFFER) << std::endl;
        return 0;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0); /// unbind

    // Render
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
    glEnable(GL_DEPTH_TEST);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Activate shader and set uniforms
    shader.use();
    shader.setMat4("NormalMatrix", NormalMatrix);
    shader.setMat4("MVPMatrix", MVPMatrix);
    shader.setMat4("MVMatrix", MVMatrix);
    shader.setMat4("MMatrix", MMatrix);
    shader.setMat4("VMatrix", VMatrix);
    shader.setMat4("PMatrix", PMatrix);

    // Draw mesh
    mesh.draw();

    // Read data from frame buffer
    cv::Mat dst = cv::Mat::zeros(SCR_HEIGHT, SCR_WIDTH, CV_32FC3);
    glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_BGR, GL_FLOAT, dst.data);

    // Unbind frame buffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.

    cv::cvtColor(dst, dst, CV_BGR2GRAY);
    cv::normalize(dst, dst, 0, 1, CV_MINMAX);
    cv::imshow("data", dst);
    cv::waitKey(0);

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

//int main() {
//    // Convert templates from t-less to custom format
////    tless::Converter converter;
////    converter.convert("data/convert_primesense.txt", "data/models/", "data/108x108/primesense/", 108);
////    converter.convert("data/convert_kinectv2.txt", "data/models/", "data/108x108/kinectv2/", 108);
//
//    // Custom criteria
//    cv::Ptr<tless::ClassifierCriteria> criteria(new tless::ClassifierCriteria());
//
//    // Training params
//    criteria->tablesCount = 100;
//    criteria->minVotes = 3;
//    criteria->depthBinCount = 5;
//
//    // Detect params
//    criteria->matchFactor = 0.6f;
//
//    // Init classifier
//    tless::Classifier classifier(criteria);
//
////     Run classifier
//    // Primesense
////    classifier.train("data/templates_primesense.txt", "data/trained/primesense/");
////    classifier.detect("data/trained_primesense.txt", "data/trained/primesense/", "data/scenes/primesense/02/");
//
//    // Kinect
////    classifier.train("data/templates_kinectv2.txt", "data/trained/kinectv2/");
//    classifier.detect("data/trained_kinectv2.txt", "data/trained/kinectv2/", "data/scenes/kinectv2/02/");
//
//    return 0;
//}