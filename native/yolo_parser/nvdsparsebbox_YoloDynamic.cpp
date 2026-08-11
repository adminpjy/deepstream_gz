// Generic host-output parser for Ultralytics YOLO8/9/11 detection exports.
// The TensorRT graph performs inference; this code only decodes raw candidates.

#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

float clamp_value(float value, float lower, float upper) {
    return std::min(upper, std::max(lower, value));
}

}  // namespace

extern "C" bool NvDsInferParseCustomYoloDynamic(
    const std::vector<NvDsInferLayerInfo>& output_layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& detection,
    std::vector<NvDsInferParseObjectInfo>& objects) {
    if (output_layers.empty() || detection.numClassesConfigured == 0) {
        std::cerr << "YOLO dynamic parser: missing output or configured classes" << std::endl;
        return false;
    }
    const NvDsInferLayerInfo& layer = output_layers.front();
    if (layer.buffer == nullptr || layer.inferDims.numElements <= 0 ||
        layer.inferDims.numDims < 1) {
        std::cerr << "YOLO dynamic parser: invalid output tensor" << std::endl;
        return false;
    }

    const int classes = static_cast<int>(detection.numClassesConfigured);
    const int attributes = 4 + classes;
    if (layer.inferDims.numElements % attributes != 0) {
        std::cerr << "YOLO dynamic parser: tensor element count "
                  << layer.inferDims.numElements << " is not divisible by 4+C="
                  << attributes << std::endl;
        return false;
    }
    const int rows = layer.inferDims.numElements / attributes;
    const int first = layer.inferDims.d[0];
    const int last = layer.inferDims.d[layer.inferDims.numDims - 1];
    const bool row_major = last == attributes;
    const bool channel_major = first == attributes;
    if (!row_major && !channel_major) {
        std::cerr << "YOLO dynamic parser: expected [rows,4+C] or [4+C,rows], got first="
                  << first << " last=" << last << " C=" << classes << std::endl;
        return false;
    }

    const float* data = static_cast<const float*>(layer.buffer);
    const auto value_at = [=](int row, int attribute) -> float {
        return row_major ? data[row * attributes + attribute]
                         : data[attribute * rows + row];
    };
    objects.clear();
    objects.reserve(static_cast<std::size_t>(std::min(rows, 1000)));
    for (int row = 0; row < rows; ++row) {
        int best_class = 0;
        float best_score = value_at(row, 4);
        for (int class_id = 1; class_id < classes; ++class_id) {
            const float score = value_at(row, 4 + class_id);
            if (score > best_score) {
                best_score = score;
                best_class = class_id;
            }
        }
        if (!std::isfinite(best_score) || best_score < 0.0F || best_score > 1.0F) {
            continue;
        }
        const float threshold = detection.perClassPreclusterThreshold.at(best_class);
        if (best_score < threshold) {
            continue;
        }

        const float center_x = value_at(row, 0);
        const float center_y = value_at(row, 1);
        const float width = value_at(row, 2);
        const float height = value_at(row, 3);
        if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
            !std::isfinite(width) || !std::isfinite(height) || width <= 0.0F ||
            height <= 0.0F) {
            continue;
        }
        const float x1 = clamp_value(center_x - width / 2.0F, 0.0F,
                                     static_cast<float>(network.width));
        const float y1 = clamp_value(center_y - height / 2.0F, 0.0F,
                                     static_cast<float>(network.height));
        const float x2 = clamp_value(center_x + width / 2.0F, 0.0F,
                                     static_cast<float>(network.width));
        const float y2 = clamp_value(center_y + height / 2.0F, 0.0F,
                                     static_cast<float>(network.height));
        if (x2 - x1 < 1.0F || y2 - y1 < 1.0F) {
            continue;
        }

        NvDsInferParseObjectInfo object{};
        object.left = x1;
        object.top = y1;
        object.width = x2 - x1;
        object.height = y2 - y1;
        object.classId = best_class;
        object.detectionConfidence = best_score;
        objects.push_back(object);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloDynamic);

