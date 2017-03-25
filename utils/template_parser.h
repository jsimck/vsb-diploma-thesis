#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H

#include <string>
#include "../core/template.h"
#include "../core/template_group.h"

class TemplateParser {
private:
    std::string basePath;
    unsigned int tplCount;
    std::vector<std::string> templateFolders;
    std::unique_ptr<std::vector<int>> indices;

    Template parseGt(int index, std::string path, cv::FileNode &gtNode);
    void parseInfo(Template &tpl, cv::FileNode &infoNode);
public:
    static int idCounter;

    TemplateParser() {}
    TemplateParser(const std::string basePath, std::vector<std::string> templateFolders, unsigned int tplCount = 1296)
        : basePath(basePath), templateFolders(templateFolders), tplCount(tplCount) {}

    void parse(std::vector<TemplateGroup> &groups);
    void parseTemplate(std::vector<Template> &templates, std::string tplName);
    void parseTemplate(std::vector<Template> &templates, std::string tplName, std::unique_ptr<std::vector<int>> &indices);

    // Getters
    static int getIdCounter();
    std::string getBasePath() const;
    unsigned int getTplCount() const;
    const std::vector<std::string> &getTemplateFolders() const;
    const std::unique_ptr<std::vector<int>> &getIndices() const;

    // Setters
    void setBasePath(std::string path);
    void setTplCount(unsigned int tplCount);
    void setTemplateFolders(const std::vector<std::string> &templateFolders);
    void setIndices(std::unique_ptr<std::vector<int>> &indices);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
