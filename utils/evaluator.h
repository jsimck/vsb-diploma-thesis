#ifndef VSB_SEMESTRAL_PROJECT_VALIDATOR_H
#define VSB_SEMESTRAL_PROJECT_VALIDATOR_H

#include <string>
#include "../core/result.h"

namespace tless {
    class Evaluator {
    private:
        float minOverlap;
        std::string scenesFolder;

        void evaluate(std::vector<std::vector<Result>> &results, int sceneId);
    public:
        Evaluator(const std::string &scenesFolder, float minOverlap = 0.5f)
                : minOverlap(minOverlap), scenesFolder(scenesFolder) {}

        void evaluate(const std::string &resultsFile, int sceneId);
    };
}

#endif
