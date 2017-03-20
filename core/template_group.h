#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H

#include <string>
#include "template.h"

struct TemplateGroup {
public:
    TemplateGroup(std::string folderName, std::vector<Template> templates)
            : folderName(folderName), templates(templates) {}
private:
    std::string folderName;
    std::vector<Template> templates;
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H
