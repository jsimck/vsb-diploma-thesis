#ifndef VSB_SEMESTRAL_PROJECT_UTILS_H
#define VSB_SEMESTRAL_PROJECT_UTILS_H

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

#endif //VSB_SEMESTRAL_PROJECT_UTILS_H
