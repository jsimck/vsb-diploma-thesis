#ifndef VSB_SEMESTRAL_PROJECT_PARSER_H
#define VSB_SEMESTRAL_PROJECT_PARSER_H

#include <string>
#include "../core/template.h"
#include "../core/group.h"
#include "../core/dataset_info.h"

/**
 * class Parser
 *
 * Utility class used to parse images downloaded from http://cmp.felk.cvut.cz/t-less/
 * into form which can be then modified and further used in the code.
 */
class Parser {
private:
    static uint idCounter;

    std::string basePath;
    std::vector<uint> indices;
    std::vector<std::string> folders;
    uint tplCount;

    Template parseGt(const uint index, const std::string path, cv::FileNode &gtNode, DataSetInfo &info);
    void parseInfo(Template &tpl, cv::FileNode &infoNode);
    void parseTemplate(std::vector<Template> &templates, DataSetInfo &info, std::string tplName);
public:
    Parser(const std::string basePath = "/data", std::vector<std::string> templateFolders = {}, uint tplCount = 1296)
        : basePath(basePath), folders(templateFolders), tplCount(tplCount) {}

    void parse(std::vector<Group> &groups, DataSetInfo &info);
    void clearIndices();

    // Getters
    static int getIdCounter();
    std::string getBasePath() const;
    uint getTplCount() const;
    const std::vector<std::string> &getTemplateFolders() const;
    const std::vector<uint> &getIndices() const;

    // Setters
    void setBasePath(std::string path);
    void setTplCount(uint tplCount);
    void setFolders(const std::vector<std::string> &folders);
    void setIndices(const std::vector<uint> &indices);
};

#endif //VSB_SEMESTRAL_PROJECT_PARSER_H
