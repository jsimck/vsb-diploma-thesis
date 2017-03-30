#ifndef VSB_SEMESTRAL_PROJECT_UTILS_H
#define VSB_SEMESTRAL_PROJECT_UTILS_H

#include <string>

#define SQR(x) ((x) * (x))

#define SAFE_DELETE(p) { \
    if ( p != NULL ) \
    { \
        delete p; \
        p = NULL; \
    } \
}

#define SAFE_DELETE_ARRAY(p) { \
    if ( p != NULL ) \
    { \
        delete [] p; \
        p = NULL; \
    } \
}

// ---------------------------------------
// | Mat type |  C1 |  C2  |  C3  |  C4  |
// ---------------------------------------
// | CV_8U    |  0  |  8   |  16  |  24  |
// | CV_8S    |  1  |  9   |  17  |  25  |
// | CV_16U   |  2  |  10  |  18  |  26  |
// | CV_16S   |  3  |  11  |  19  |  27  |
// | CV_32S   |  4  |  12  |  20  |  28  |
// | CV_32F   |  5  |  13  |  21  |  29  |
// | CV_64F   |  6  |  14  |  22  |  30  |
 // ---------------------------------------
std::string matType2Str(int type);

#endif //VSB_SEMESTRAL_PROJECT_UTILS_H
