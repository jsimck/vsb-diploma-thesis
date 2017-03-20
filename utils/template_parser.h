#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H

#include <string>
#include "../core/template.h"
#include "../core/template_group.h"

class TemplateParser {
public:
    static int idCounter;

    TemplateParser(std::string basePath, int tplCount = 1296) : basePath(basePath), tplCount(tplCount) {}

    void parse(std::vector<TemplateGroup> &groups);
    void parseTemplate(std::vector<Template> &templates);
    void parseTemplate(std::vector<Template> &templates, const int *indices, int indicesLength);

    void setBasePath(std::string path);
    void setTplCount(int tplCount);

    std::string getBasePath();
    int getTplCount();
private:
    std::string basePath;
    int tplCount;

    Template parseTemplate(int index, cv::FileNode &node);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
