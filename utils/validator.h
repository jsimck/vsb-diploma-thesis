#ifndef VSB_SEMESTRAL_PROJECT_VALIDATOR_H
#define VSB_SEMESTRAL_PROJECT_VALIDATOR_H

#include <string>
#include "../core/result.h"

namespace tless {
    class Validator {
    private:
        float minOverlap;
        std::string scenesFolderPath;

    public:
        Validator(const std::string &scenesFolder, float minOverlap = 0.5f)
                : scenesFolderPath(scenesFolder), minOverlap(minOverlap) {}

        void validate(std::vector<std::vector<Result>> &results, int sceneId);
        void validate(const std::string &resultsFile, int sceneId);
    };
}

#endif
