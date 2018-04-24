#include "evaluator.h"

namespace tless {
    void Evaluator::evaluate(const std::string &resultsFolder, const std::vector<int> &indices,
                             const std::string &resultsFileFormat) {
        // Evaluate each scene in indices
        for (auto &sceneId : indices) {
            std::vector<std::pair<int, std::vector<Result>>> results;
            std::vector<Result> sceneMatches;

            // Load results from file
            std::string resultPath = cv::format((resultsFolder + resultsFileFormat).c_str(), sceneId);
            cv::FileStorage fs(resultPath, cv::FileStorage::READ);
            cv::FileNode scenesNode = fs["scenes"];
            double tScene = 0, tObjectness = 0, tHashing = 0, tMatching = 0, tNms = 0, tFinePose = 0;
            int sceneIndex, timerCount = 0;

            for (auto &&scene : scenesNode) {
                cv::FileNode matches = scene["matches"];
                scene["index"] >> sceneIndex;

                for (auto &&m : matches) {
                    Result r;
                    m >> r;
                    sceneMatches.push_back(r);
                }

                results.emplace_back(sceneIndex, std::move(sceneMatches));

                // Timers
                double timer;
                cv::FileNode timers = scene["timers"];
                timers["scene"] >> timer;
                tScene += timer;
                timers["objectness"] >> timer;
                tObjectness += timer;
                timers["hashing"] >> timer;
                tHashing += timer;
                timers["matching"] >> timer;
                tMatching += timer;
                timers["nms"] >> timer;
                tNms += timer;
                timers["finePose"] >> timer;
                tFinePose += timer;
                timerCount++;
            }

            fs.release();

            // AVG times
            tScene /= static_cast<float>(timerCount);
            tObjectness /= static_cast<float>(timerCount);
            tHashing /= static_cast<float>(timerCount);
            tMatching /= static_cast<float>(timerCount);
            tNms /= static_cast<float>(timerCount);
            tFinePose /= static_cast<float>(timerCount);

            // Print timers
            std::cout << "TIMERS" << std::endl;
            std::cout << "  |_ scene: " << tScene << "s" << std::endl;
            std::cout << "  |_ objectness: " << tObjectness << "s" << std::endl;
            std::cout << "  |_ hashing: " << tHashing << "s" << std::endl;
            std::cout << "  |_ matching: " << tMatching << "s" << std::endl;
            std::cout << "  |_ nms: " << tNms << "s" << std::endl;
            std::cout << "  |_ finePose: " << tFinePose << "s" << std::endl;
            std::cout << "  |_ Detection sum: " << tFinePose << "s" << std::endl;

            // Evaluate
            evaluate(results, sceneId);
        }
    }

    void Evaluator::evaluate(std::vector<std::pair<int, std::vector<Result>>> &results, int sceneId) {
        // Load scene GT
        std::string sceneGtPath = cv::format((scenesFolder + "%02d/gt.yml").c_str(), sceneId);
        cv::FileStorage fs(sceneGtPath, cv::FileStorage::READ);

        // Init validation params
        int objId;
        cv::Rect objBB;
        int TP = 0, FP = 0, FN = 0;

        for (int i = 0; i < results.size(); ++i) {
            cv::FileNode scene = fs[cv::format("scene_%d", results[i].first)];

            // Iterate GT
            for (auto &&gt : scene) {
                bool checked = false;
                gt["obj_id"] >> objId;
                gt["obj_bb"] >> objBB;

                for (auto &r : results[i].second) {
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
            for (auto &r : results[i].second) {
                if (!r.validated) {
                    FP++;
                }
            }
        }

        // Calculate results
        float precision = static_cast<float>(TP) / (TP + FP);
        float recall = static_cast<float>(TP) / (TP + FN);
        float f1Score = 2 * (precision * recall) / (precision + recall);

        // Print results
        std::cout << "Scene " << sceneId << "." << std::endl;
        std::cout << "  |_ TP: " << TP << ", FP: " << FP << ", FN: " << FN
            << ", Total: " << (TP + FP + FN) << std::endl;
        std::cout << "  |_ F1: " << (f1Score * 100) << "%" << ", "
                << "Precision: " << precision << ", "
                << "Recall: " << recall << std::endl << std::endl;

        fs.release();
    }

    float Evaluator::getMinOverlap() const {
        return minOverlap;
    }

    void Evaluator::setMinOverlap(float minOverlap) {
        Evaluator::minOverlap = minOverlap;
    }

    const std::string &Evaluator::getScenesFolder() const {
        return scenesFolder;
    }

    void Evaluator::setScenesFolder(const std::string &scenesFolder) {
        Evaluator::scenesFolder = scenesFolder;
    }
}