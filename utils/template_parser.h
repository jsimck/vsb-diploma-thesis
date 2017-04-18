#ifndef VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H
#define VSB_SEMESTRAL_PROJECT_TEMPLATEPARSER_H

#include <string>
#include "../core/template.h"
#include "../core/template_group.h"
#include "../core/dataset_info.h"

/**
 * class TemplateParser
 *
 * Utility class used to parse images downloaded from http://cmp.felk.cvut.cz/t-less/
 * into form which can be then modified and further used in the code.
 */
class TemplateParser {
private:
    std::string basePath;
    unsigned int tplCount;
    std::vector<std::string> templateFolders;
    std::unique_ptr<std::vector<int>> indices;

    void parseInfo(Template &tpl, cv::FileNode &infoNode);
    Template parseGt(int index, std::string path, cv::FileNode &gtNode, DatasetInfo &info);
    void parseTemplate(std::vector<Template> &templates, DatasetInfo &info, std::string tplName);
    void parseTemplate(std::vector<Template> &templates, DatasetInfo &info, std::string tplName, std::unique_ptr<std::vector<int>> &indices);
public:
    static int idCounter;

    TemplateParser(const std::string basePath = "/data", std::vector<std::string> templateFolders = {}, unsigned int tplCount = 1296)
        : basePath(basePath), templateFolders(templateFolders), tplCount(tplCount) {}

    void parse(std::vector<TemplateGroup> &groups, DatasetInfo &info);
    void clearIndices();

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
