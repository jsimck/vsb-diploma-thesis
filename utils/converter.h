#ifndef VSB_SEMESTRAL_PROJECT_CONVERTOR_H
#define VSB_SEMESTRAL_PROJECT_CONVERTOR_H

#include <string>
#include <opencv2/core/hal/interface.h>
#include <vector>
#include <opencv2/core/types.hpp>
#include "../core/template.h"

namespace tless {
    class Converter {
    private:
        void resizeAndSave(std::vector<Template> &objectTemplates, const std::string &outputPath, int outputSize);

    public:
        void convert(std::string templatesListPath, std::string outputPath, std::string modelsPath, int outputSize = 108);
    };
}

#endif