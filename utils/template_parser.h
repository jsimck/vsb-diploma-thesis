#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H

#include <string>
#include "../core/template.h"
#include "../core/template_group.h"

class TemplateParser {
public:
    static int idCounter;

    TemplateParser(const std::string basePath, unsigned int tplCount = 1296);

    void parse(std::vector<TemplateGroup> &groups, std::vector<std::string> tplNames);
    void parseTemplate(std::vector<Template> &templates, std::string tplName);
    void parseTemplate(std::vector<Template> &templates, std::string tplName, std::vector<int> indices);

    void setBasePath(std::string path);
    void setTplCount(unsigned int tplCount);

    std::string getBasePath();
    unsigned int getTplCount();
private:
    std::string basePath;
    unsigned int tplCount;

    Template parseGt(int index, std::string path, cv::FileNode &gtNode);
    void parseInfo(Template &tpl, cv::FileNode &infoNode);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
