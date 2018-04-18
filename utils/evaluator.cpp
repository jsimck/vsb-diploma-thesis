#include "evaluator.h"

namespace tless {
    void Evaluator::evaluate(const std::string &resultsFile, int sceneId) {
        std::vector<std::vector<Result>> results;
        std::vector<Result> sceneMatches;

        // Load results from file
        cv::FileStorage fs(resultsFile, cv::FileStorage::READ);
        cv::FileNode scenesNode = fs["scenes"];
        Result r;

        for (auto &&scene : scenesNode) {
            cv::FileNode matches = scene["matches"];

            for (auto &&m : matches) {
                m >> r;
                sceneMatches.emplace_back(r);
            }

            results.emplace_back(std::move(sceneMatches));
        }

        fs.release();

        // Evaluate
        evaluate(results, sceneId);
    }

    void Evaluator::evaluate(std::vector<std::vector<Result>> &results, int sceneId) {
        std::ostringstream oss;
        oss << scenesFolderPath << std::setfill('0') << std::setw(2) << sceneId << "/" << "gt.yml";

        // Load scene
        cv::FileStorage fs(oss.str(), cv::FileStorage::READ);
        oss.str("");

        // Init validation params
        int objId;
        cv::Rect objBB;
        int TP = 0, FP = 0, FN = 0;

        for (int i = 0; i < results.size(); ++i) {
            oss << "scene_" << i;
            cv::FileNode scene = fs[oss.str()];

            // Iterate GT
            for (auto &&gt : scene) {
                bool checked = false;
                gt["obj_id"] >> objId;
                gt["obj_bb"] >> objBB;

                for (auto &r : results[i]) {
                    if (r.validated) continue;

                    if (r.jaccard(objBB) > minOverlap && r.objId == objId) {
                        TP++;
                        r.validated = true;
                        checked = true;
                        break;
                    }
                }

                if (!checked) {
                    FN++;
                }
            }

            // Count FP
            for (auto &r : results[i]) {
                if (!r.validated) {
                    FP++;
                }
            }

            oss.str("");
        }

        // Calculate results
        float precision = static_cast<float>(TP) / (TP + FP);
        float recall = static_cast<float>(TP) / (TP + FN);
        float f1score = 2 * (precision * recall) / (precision + recall);

        // Print results
        std::cout << "TP: " << TP << ", FP: " << FP << ", FN: " << FN
            << ", Total: " << (TP + FP + FN) << std::endl;
        std::cout << "precision: " << precision << std::endl
            << "recall: " << recall << std::endl
            << "f1score: " << (f1score * 100) << "%" << std::endl;

        fs.release();
    }
}