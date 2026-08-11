#pragma once

#include <vector>

struct NvDsInferDims {
    int numDims;
    int d[8];
    int numElements;
};

enum NvDsInferDataType {
    FLOAT = 0,
    HALF = 1,
    INT8 = 2,
    INT32 = 3,
};

struct NvDsInferLayerInfo {
    void* buffer;
    const char* layerName;
    NvDsInferDims inferDims;
    NvDsInferDataType dataType;
};

struct NvDsInferNetworkInfo {
    unsigned int width;
    unsigned int height;
};

struct NvDsInferParseDetectionParams {
    unsigned int numClassesConfigured;
    std::vector<float> perClassPreclusterThreshold;
};

struct NvDsInferParseObjectInfo {
    float left;
    float top;
    float width;
    float height;
    int classId;
    float detectionConfidence;
};

#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(name)
