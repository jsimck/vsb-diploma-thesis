#ifndef VSB_SEMESTRAL_PROJECT_PARSER_H
#define VSB_SEMESTRAL_PROJECT_PARSER_H

#include <string>
#include <utility>
#include <memory>
#include "../core/template.h"
#include "../core/classifier_terms.h"

/**
 * class Parser
 *
 * Utility class used to parse images downloaded from http://cmp.felk.cvut.cz/t-less/
 * into form which can be then modified and further used in the code.
 */
class Parser {
private:
    std::shared_ptr<ClassifierTerms> terms;
    uint idCounter;

    Template parseGt(uint index, const std::string &path, cv::FileNode &gtNode);
    void parseInfo(Template &tpl, cv::FileNode &infoNode);
public:
    std::vector<uint> indices;
    uint tplCount;

    Parser(std::shared_ptr<ClassifierTerms> terms, uint tplCount = 1296) : terms(std::move(terms)), tplCount(tplCount) {}

    void parse(const std::string &basePath, std::vector<Template> &templates);
};

#endif //VSB_SEMESTRAL_PROJECT_PARSER_H
