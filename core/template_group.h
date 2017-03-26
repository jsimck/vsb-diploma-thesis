#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H

#include <string>
#include "template.h"

/**
 * struct TemplateGroup
 *
 * Simple structure, served only for purpose of separating each parsed template
 * into it's own group
 */
struct TemplateGroup {
public:
    std::string folderName;
    std::vector<Template> templates;

    // Constructors
    TemplateGroup() {}
    TemplateGroup(std::string folderName, std::vector<Template> templates)
            : folderName(folderName), templates(templates) {}
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H
