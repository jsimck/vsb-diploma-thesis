#ifndef VSB_SEMESTRAL_PROJECT_PARSER_H
#define VSB_SEMESTRAL_PROJECT_PARSER_H

#include <string>
#include <utility>
#include "../core/template.h"
#include "../core/dataset_info.h"

/**
 * class Parser
 *
 * Utility class used to parse images downloaded from http://cmp.felk.cvut.cz/t-less/
 * into form which can be then modified and further used in the code.
 */
class Parser {
private:
    uint idCounter;

    Template parseGt(uint index, const std::string &path, cv::FileNode &gtNode, DataSetInfo &info);
    void parseInfo(Template &tpl, cv::FileNode &infoNode);
public:
    std::vector<uint> indices;
    uint tplCount;

    Parser(uint tplCount = 1296) : tplCount(tplCount) {}

    void parse(const std::string &basePath, std::vector<Template> &templates, DataSetInfo &info);
};

#endif //VSB_SEMESTRAL_PROJECT_PARSER_H
