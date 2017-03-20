#include "template_group.h"

const std::vector<Template> &TemplateGroup::getTemplates() const {
    return templates;
}

const std::string &TemplateGroup::getFolderName() const {
    return folderName;
}