// Strict parser for the standard 9-output SCRFD face-detector contract.

#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

constexpr std::array<int, 3> kStrides{8, 16, 32};
constexpr int kAnchors = 2;

enum class Role { kUnknown, kScore, kBox, kKeypoints };

const char* role_name(Role role) {
    switch (role) {
        case Role::kScore:
            return "score";
        case Role::kBox:
            return "bbox";
        case Role::kKeypoints:
            return "kps";
        default:
            return "unknown";
    }
}

struct TensorView {
    const float* data{nullptr};
    int rows{0};
    int width{0};
    bool row_major{true};

    float at(int row, int column) const {
        return row_major ? data[row * width + column] : data[column * rows + row];
    }
};

struct OutputGroup {
    TensorView score;
    TensorView box;
    TensorView keypoints;
};

float clamp_value(float value, float lower, float upper) {
    return std::min(upper, std::max(lower, value));
}

bool contains(const std::string& value, const char* token) {
    return value.find(token) != std::string::npos;
}

Role role_from_name(const char* raw_name) {
    if (raw_name == nullptr) {
        return Role::kUnknown;
    }
    std::string name(raw_name);
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (contains(name, "kps") || contains(name, "keypoint") || contains(name, "landmark")) {
        return Role::kKeypoints;
    }
    if (contains(name, "bbox") || contains(name, "box") || contains(name, "loc")) {
        return Role::kBox;
    }
    if (contains(name, "score") || contains(name, "cls") || contains(name, "conf")) {
        return Role::kScore;
    }
    return Role::kUnknown;
}

int width_for(Role role) {
    switch (role) {
        case Role::kScore:
            return 1;
        case Role::kBox:
            return 4;
        case Role::kKeypoints:
            return 10;
        default:
            return 0;
    }
}

bool valid_dimensions(const NvDsInferLayerInfo& layer) {
    const int num_dims = static_cast<int>(layer.inferDims.numDims);
    const int max_dims =
        static_cast<int>(sizeof(layer.inferDims.d) / sizeof(layer.inferDims.d[0]));
    const std::size_t elements = static_cast<std::size_t>(layer.inferDims.numElements);
    if (num_dims <= 0 || num_dims > max_dims || elements == 0) {
        return false;
    }

    std::size_t product = 1;
    for (int index = 0; index < num_dims; ++index) {
        const int dimension = static_cast<int>(layer.inferDims.d[index]);
        if (dimension <= 0 ||
            product > std::numeric_limits<std::size_t>::max() /
                          static_cast<std::size_t>(dimension)) {
            return false;
        }
        product *= static_cast<std::size_t>(dimension);
    }
    return product == elements;
}

Role role_from_shape(const NvDsInferLayerInfo& layer) {
    if (!valid_dimensions(layer)) {
        return Role::kUnknown;
    }
    const int num_dims = static_cast<int>(layer.inferDims.numDims);
    // DeepStream may pad an explicit TensorRT output with trailing singleton
    // dimensions, e.g. [rows,4,1]. Inspect every dimension instead of only
    // the first/last one so numeric SCRFD binding names still work.
    for (int index = 0; index < num_dims; ++index) {
        if (static_cast<int>(layer.inferDims.d[index]) == 10) {
            return Role::kKeypoints;
        }
    }
    for (int index = 0; index < num_dims; ++index) {
        if (static_cast<int>(layer.inferDims.d[index]) == 4) {
            return Role::kBox;
        }
    }
    return Role::kScore;
}

bool make_view(const NvDsInferLayerInfo& layer, Role role, TensorView& view) {
    const int width = width_for(role);
    const std::size_t elements = static_cast<std::size_t>(layer.inferDims.numElements);
    const std::size_t tensor_width = static_cast<std::size_t>(width);
    if (width == 0 || layer.dataType != FLOAT || layer.buffer == nullptr ||
        !valid_dimensions(layer) || elements % tensor_width != 0 ||
        elements / tensor_width > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    const int num_dims = static_cast<int>(layer.inferDims.numDims);
    const int rows = static_cast<int>(elements / tensor_width);
    bool row_major = true;
    if (width != 1) {
        int width_index = -1;
        int rows_index = -1;
        for (int index = 0; index < num_dims; ++index) {
            const int dimension = static_cast<int>(layer.inferDims.d[index]);
            if (dimension == width && width_index == -1) width_index = index;
            if (dimension == rows && rows_index == -1) rows_index = index;
        }
        if (width_index == -1 || rows_index == -1 || width_index == rows_index) {
            return false;
        }
        // [rows,width,(1...)] has interleaved attributes; [width,rows,(1...)]
        // is channel-major. Singleton padding after either axis is irrelevant.
        row_major = rows_index < width_index;
    }
    view = TensorView{static_cast<const float*>(layer.buffer), rows, width, row_major};
    return true;
}

void log_contract(const std::vector<NvDsInferLayerInfo>& layers) {
    for (const auto& layer : layers) {
        const int num_dims = static_cast<int>(layer.inferDims.numDims);
        Role role = role_from_name(layer.layerName);
        if (role == Role::kUnknown) role = role_from_shape(layer);
        TensorView view;
        const bool valid_view = make_view(layer, role, view);
        std::cerr << "SCRFD parser: layer="
                  << (layer.layerName == nullptr ? "<null>" : layer.layerName)
                  << " dims=[";
        for (int index = 0; index < num_dims; ++index) {
            if (index != 0) std::cerr << ',';
            std::cerr << layer.inferDims.d[index];
        }
        std::cerr << "] elements=" << layer.inferDims.numElements
                  << " role=" << role_name(role)
                  << " rows=" << (valid_view ? view.rows : 0)
                  << " row_major=" << (valid_view && view.row_major ? "true" : "false")
                  << std::endl;
    }
}

int stride_for_rows(int rows, const NvDsInferNetworkInfo& network) {
    for (int stride : kStrides) {
        const int expected = static_cast<int>(network.width / stride) *
                             static_cast<int>(network.height / stride) * kAnchors;
        if (rows == expected) return stride;
    }
    return 0;
}

}  // namespace

extern "C" bool NvDsInferParseCustomSCRFD(
    const std::vector<NvDsInferLayerInfo>& output_layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& detection,
    std::vector<NvDsInferParseObjectInfo>& objects) {
    objects.clear();
    if (output_layers.size() != 9 || detection.numClassesConfigured != 1 ||
        detection.perClassPreclusterThreshold.empty()) {
        std::cerr << "SCRFD parser: expected 9 outputs and exactly one class" << std::endl;
        return false;
    }
    std::map<int, OutputGroup> groups;
    for (const auto& layer : output_layers) {
        if (layer.dataType != FLOAT) {
            std::cerr << "SCRFD parser: output tensors must use FP32" << std::endl;
            return false;
        }
        if (!valid_dimensions(layer)) {
            std::cerr << "SCRFD parser: invalid output tensor dimensions" << std::endl;
            return false;
        }
        Role role = role_from_name(layer.layerName);
        if (role == Role::kUnknown) role = role_from_shape(layer);
        TensorView view;
        if (role == Role::kUnknown || !make_view(layer, role, view)) {
            std::cerr << "SCRFD parser: unsupported output name/shape" << std::endl;
            return false;
        }
        OutputGroup& group = groups[view.rows];
        TensorView* destination = nullptr;
        if (role == Role::kScore) destination = &group.score;
        if (role == Role::kBox) destination = &group.box;
        if (role == Role::kKeypoints) destination = &group.keypoints;
        if (destination == nullptr || destination->data != nullptr) {
            std::cerr << "SCRFD parser: duplicate output role for one stride" << std::endl;
            log_contract(output_layers);
            return false;
        }
        *destination = view;
    }
    if (groups.size() != kStrides.size()) {
        std::cerr << "SCRFD parser: incomplete stride groups" << std::endl;
        return false;
    }

    for (const auto& [rows, group] : groups) {
        const int stride = stride_for_rows(rows, network);
        if (stride == 0 || group.score.data == nullptr || group.box.data == nullptr ||
            group.keypoints.data == nullptr) {
            std::cerr << "SCRFD parser: rows do not match strides 8/16/32 with two anchors"
                      << std::endl;
            return false;
        }
        const int grid_width = static_cast<int>(network.width / stride);
        const float threshold = detection.perClassPreclusterThreshold.at(0);
        for (int row = 0; row < rows; ++row) {
            const float score = group.score.at(row, 0);
            if (!std::isfinite(score) || score < threshold || score > 1.0F) continue;
            const int location = row / kAnchors;
            const float center_x = static_cast<float>((location % grid_width) * stride);
            const float center_y = static_cast<float>((location / grid_width) * stride);
            const float left = group.box.at(row, 0) * stride;
            const float top = group.box.at(row, 1) * stride;
            const float right = group.box.at(row, 2) * stride;
            const float bottom = group.box.at(row, 3) * stride;
            if (!std::isfinite(left) || !std::isfinite(top) || !std::isfinite(right) ||
                !std::isfinite(bottom)) {
                continue;
            }
            const float x1 = clamp_value(center_x - left, 0.0F, static_cast<float>(network.width));
            const float y1 = clamp_value(center_y - top, 0.0F, static_cast<float>(network.height));
            const float x2 = clamp_value(center_x + right, 0.0F, static_cast<float>(network.width));
            const float y2 = clamp_value(center_y + bottom, 0.0F, static_cast<float>(network.height));
            if (x2 - x1 < 1.0F || y2 - y1 < 1.0F) continue;
            NvDsInferParseObjectInfo object{};
            object.left = x1;
            object.top = y1;
            object.width = x2 - x1;
            object.height = y2 - y1;
            object.classId = 0;
            object.detectionConfidence = score;
            objects.push_back(object);
        }
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSCRFD);
