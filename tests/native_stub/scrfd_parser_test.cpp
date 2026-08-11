#include "nvdsinfer_custom_impl.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

extern "C" bool NvDsInferParseCustomSCRFD(
    const std::vector<NvDsInferLayerInfo>&,
    const NvDsInferNetworkInfo&,
    const NvDsInferParseDetectionParams&,
    std::vector<NvDsInferParseObjectInfo>&);

static NvDsInferLayerInfo layer(float* data, int rows, int width, const char* name) {
    NvDsInferLayerInfo value{};
    value.buffer = data;
    value.layerName = name;
    value.inferDims.numDims = 2;
    value.inferDims.d[0] = rows;
    value.inferDims.d[1] = width;
    value.inferDims.numElements = rows * width;
    value.dataType = FLOAT;
    return value;
}

int main() {
    NvDsInferNetworkInfo network{32, 32};
    NvDsInferParseDetectionParams detection{1, {0.5F}};
    std::vector<std::vector<float>> storage;
    std::vector<NvDsInferLayerInfo> layers;
    for (const auto& [stride, rows] : std::vector<std::pair<int, int>>{{8, 32}, {16, 8}, {32, 2}}) {
        storage.emplace_back(rows, 0.0F);
        storage.emplace_back(rows * 4, 0.0F);
        storage.emplace_back(rows * 10, 0.0F);
        const std::size_t offset = storage.size() - 3;
        std::string* names = new std::string[3]{
            "score_" + std::to_string(stride),
            "bbox_" + std::to_string(stride),
            "kps_" + std::to_string(stride),
        };
        layers.push_back(layer(storage[offset].data(), rows, 1, names[0].c_str()));
        layers.push_back(layer(storage[offset + 1].data(), rows, 4, names[1].c_str()));
        layers.push_back(layer(storage[offset + 2].data(), rows, 10, names[2].c_str()));
    }
    // stride-8, cell (1,1), first anchor => center (8,8), box (4,4)-(12,12).
    storage[0][10] = 0.9F;
    for (int index = 0; index < 4; ++index) storage[1][10 * 4 + index] = 0.5F;
    std::vector<NvDsInferParseObjectInfo> objects;
    assert(NvDsInferParseCustomSCRFD(layers, network, detection, objects));
    assert(objects.size() == 1);
    assert(std::fabs(objects[0].left - 4.0F) < 1e-6F);
    assert(std::fabs(objects[0].top - 4.0F) < 1e-6F);
    assert(std::fabs(objects[0].width - 8.0F) < 1e-6F);
    assert(std::fabs(objects[0].height - 8.0F) < 1e-6F);

    // DS9 can expose explicit outputs as [rows,width,1] and buffalo_l uses
    // numeric binding names. Shape fallback must still distinguish all roles.
    auto padded_numeric_layers = layers;
    const char* numeric_names[] = {
        "448", "451", "454", "471", "474", "477", "494", "497", "500",
    };
    for (std::size_t index = 0; index < padded_numeric_layers.size(); ++index) {
        auto& padded = padded_numeric_layers[index];
        padded.layerName = numeric_names[index];
        padded.inferDims.numDims = 3;
        padded.inferDims.d[2] = 1;
    }
    assert(NvDsInferParseCustomSCRFD(
        padded_numeric_layers, network, detection, objects));
    assert(objects.size() == 1);

    for (const NvDsInferDataType dtype : {HALF, INT8, INT32}) {
        auto non_float_layers = layers;
        non_float_layers[0].dataType = dtype;
        assert(!NvDsInferParseCustomSCRFD(non_float_layers, network, detection, objects));
        assert(objects.empty());
    }

    auto missing_dims = layers;
    missing_dims[0].inferDims.numDims = 0;
    assert(!NvDsInferParseCustomSCRFD(missing_dims, network, detection, objects));
    assert(objects.empty());

    auto inconsistent_dims = layers;
    --inconsistent_dims[0].inferDims.numElements;
    assert(!NvDsInferParseCustomSCRFD(inconsistent_dims, network, detection, objects));
    assert(objects.empty());
    return 0;
}
