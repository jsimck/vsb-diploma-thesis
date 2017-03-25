#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H

#include <string>
#include "template.h"

struct TemplateGroup {
private:
    std::string folderName;
public:
    std::vector<Template> templates;

    TemplateGroup(std::string folderName, std::vector<Template> templates)
            : folderName(folderName), templates(templates) {}

    const std::string &getFolderName() const;
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H
