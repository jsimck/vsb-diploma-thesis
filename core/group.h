#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H

#include <string>
#include "template.h"

/**
 * struct Group
 *
 * Only for purpose of separating each parsed template into it's own group
 */
struct Group {
public:
    std::string folderName;
    std::vector<Template> templates;

    // Constructors
    Group() {}
    Group(std::string folderName, std::vector<Template> templates)
            : folderName(folderName), templates(templates) {}
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATE_GROUP_H
