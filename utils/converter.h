#ifndef VSB_SEMESTRAL_PROJECT_CONVERTOR_H
#define VSB_SEMESTRAL_PROJECT_CONVERTOR_H

#include <string>
#include <opencv2/core/hal/interface.h>
#include <vector>

namespace tless {
    class Converter {
    public:
        void convert(std::string templatesListPath, std::string resultPath, std::string modelsPath, std::vector<uint> indices = {});
    };
}

#endif