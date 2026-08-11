#include "nvdsinfer_custom_impl.h"

#include <cassert>
#include <cmath>
#include <vector>

extern "C" bool NvDsInferParseCustomYoloDynamic(
    const std::vector<NvDsInferLayerInfo>&,
    const NvDsInferNetworkInfo&,
    const NvDsInferParseDetectionParams&,
    std::vector<NvDsInferParseObjectInfo>&);

static void verify(float* data, int first, int last) {
    NvDsInferLayerInfo layer{};
    layer.buffer = data;
    layer.inferDims.numDims = 2;
    layer.inferDims.d[0] = first;
    layer.inferDims.d[1] = last;
    layer.inferDims.numElements = 10;
    NvDsInferNetworkInfo network{100, 100};
    NvDsInferParseDetectionParams detection{1, {0.5F}};
    std::vector<NvDsInferParseObjectInfo> objects;
    assert(NvDsInferParseCustomYoloDynamic({layer}, network, detection, objects));
    assert(objects.size() == 1);
    assert(std::fabs(objects[0].left - 8.0F) < 1e-6F);
    assert(std::fabs(objects[0].top - 17.0F) < 1e-6F);
    assert(std::fabs(objects[0].width - 4.0F) < 1e-6F);
    assert(std::fabs(objects[0].height - 6.0F) < 1e-6F);
    assert(std::fabs(objects[0].detectionConfidence - 0.9F) < 1e-6F);
}

int main() {
    float row_major[] = {10, 20, 4, 6, 0.9F, 30, 40, 2, 2, 0.1F};
    verify(row_major, 2, 5);
    float channel_major[] = {10, 30, 20, 40, 4, 2, 6, 2, 0.9F, 0.1F};
    verify(channel_major, 5, 2);
    return 0;
}
