#ifndef VSB_SEMESTRAL_PROJECT_GROUP_H
#define VSB_SEMESTRAL_PROJECT_GROUP_H

#include <string>
#include "template.h"

/**
 * struct Group
 *
 * Separates templates of different objects into separate groups.
 */
struct Group {
public:
    std::string folderName;
    std::vector<Template> templates;

    // Constructors
    Group() {}
    Group(const std::string folderName, std::vector<Template> templates)
            : folderName(folderName), templates(templates) {}
};

#endif //VSB_SEMESTRAL_PROJECT_GROUP_H
