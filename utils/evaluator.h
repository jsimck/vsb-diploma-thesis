#ifndef VSB_SEMESTRAL_PROJECT_VALIDATOR_H
#define VSB_SEMESTRAL_PROJECT_VALIDATOR_H

#include <string>
#include "../core/result.h"

namespace tless {
    class Evaluator {
    private:
        float minOverlap;
        std::string scenesFolder;

        /**
         * @brief Evaluates given results agains GT available in gt.yml and prints F1 score.
         *
         * @param[in] results      Array of results, loaded from results files
         * @param[in] sceneId      Current scene ID
         */
        void evaluate(std::vector<std::pair<int, std::vector<Result>>> &results, int sceneId);
    public:
        Evaluator(const std::string &scenesFolder, float minOverlap = 0.5f)
                : minOverlap(minOverlap), scenesFolder(scenesFolder) {}

        /**
         * @brief Loads and parses saved results for given indicies (scenes) and evaluates them.
         *
         * @param[in] resultsFolder     Path to results folder
         * @param[in] indices           Indices identifying specific results files
         * @param[in] resultsFileFormat Individual results file name format
         */
        void evaluate(const std::string &resultsFolder, const std::vector<int> &indices,
                      const std::string &resultsFileFormat = "results_%02d.yml.gz");

        void setScenesFolder(const std::string &scenesFolder);
        void setMinOverlap(float minOverlap);

        const std::string &getScenesFolder() const;
        float getMinOverlap() const;
    };
}

#endif
