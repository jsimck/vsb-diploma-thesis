#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H


#include <string>
#include "../core/template.h"

class TemplateParser {
public:
    TemplateParser(std::string basePath, int tplCount = 1296) : basePath(basePath), tplCount(tplCount) {}

    void parse(std::vector<Template> &templates);
    void parse(std::vector<Template> &templates, int *indices, int indicesLength);

    void setBasePath(std::string path);
    std::string getBasePath();

    int getTplCount();
    void setTplCount(int tplCount);
private:
    std::string basePath;
    int tplCount;

    Template parseTemplate(int index, cv::FileNode &node);
};


#endif //VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
