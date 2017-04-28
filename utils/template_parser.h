#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H

#include <string>
#include "../core/template.h"
#include "../core/group.h"
#include "../core/dataset_info.h"

/**
 * class TemplateParser
 *
 * Utility class used to parse images downloaded from http://cmp.felk.cvut.cz/t-less/
 * into form which can be then modified and further used in the code.
 */
class TemplateParser {
private:
    static uint idCounter;

    uint tplCount;
    std::string basePath;
    std::vector<std::string> folders;
    std::vector<int> indices;

    Template parseGt(const int index, const std::string path, cv::FileNode &gtNode, DataSetInfo &info);
    void parseInfo(Template &tpl, cv::FileNode &infoNode);
    void parseTemplate(std::vector<Template> &templates, DataSetInfo &info, std::string tplName);
public:
    TemplateParser(const std::string basePath = "/data", std::vector<std::string> templateFolders = {}, uint tplCount = 1296)
        : basePath(basePath), folders(templateFolders), tplCount(tplCount) {}

    void parse(std::vector<Group> &groups, DataSetInfo &info);
    void clearIndices();

    // Getters
    static int getIdCounter();
    std::string getBasePath() const;
    uint getTplCount() const;
    const std::vector<std::string> &getTemplateFolders() const;
    const std::vector<int> &getIndices() const;

    // Setters
    void setBasePath(std::string path);
    void setTplCount(uint tplCount);
    void setFolders(const std::vector<std::string> &folders);
    void setIndices(const std::vector<int> &indices);
};

#endif //VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
