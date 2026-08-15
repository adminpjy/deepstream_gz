// Generic host-output parser for Ultralytics YOLO8/9/11 detection exports.
// The TensorRT graph performs inference; this code only decodes raw candidates.

#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

namespace {

float clamp_value(float value, float lower, float upper) {
    return std::min(upper, std::max(lower, value));
}

struct TensorView {
    const float* data;
    int rows;
    int attributes;
    bool row_major;

    float value_at(int row, int attribute) const {
        return row_major ? data[row * attributes + attribute]
                         : data[attribute * rows + row];
    }
};

bool make_tensor_view(
    const NvDsInferLayerInfo& layer,
    int attributes,
    TensorView& view,
    const char* parser_name) {
    if (layer.buffer == nullptr || layer.inferDims.numElements <= 0 ||
        layer.inferDims.numDims < 1 || attributes <= 4) {
        std::cerr << parser_name << ": invalid output tensor" << std::endl;
        return false;
    }
    if (layer.inferDims.numElements % attributes != 0) {
        std::cerr << parser_name << ": tensor element count "
                  << layer.inferDims.numElements << " is not divisible by attributes="
                  << attributes << std::endl;
        return false;
    }
    const int rows = layer.inferDims.numElements / attributes;
    const int first = layer.inferDims.d[0];
    const int last = layer.inferDims.d[layer.inferDims.numDims - 1];
    const bool row_major = last == attributes;
    const bool channel_major = first == attributes;
    if (!row_major && !channel_major) {
        std::cerr << parser_name << ": expected [rows," << attributes << "] or ["
                  << attributes << ",rows], got first=" << first << " last=" << last
                  << std::endl;
        return false;
    }
    view = TensorView{
        static_cast<const float*>(layer.buffer),
        rows,
        attributes,
        row_major,
    };
    return true;
}

bool append_object(
    const TensorView& view,
    int row,
    int class_id,
    float score,
    const NvDsInferNetworkInfo& network,
    std::vector<NvDsInferParseObjectInfo>& objects) {
    const float center_x = view.value_at(row, 0);
    const float center_y = view.value_at(row, 1);
    const float width = view.value_at(row, 2);
    const float height = view.value_at(row, 3);
    if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
        !std::isfinite(width) || !std::isfinite(height) || width <= 0.0F ||
        height <= 0.0F) {
        return false;
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
        return false;
    }

    NvDsInferParseObjectInfo object{};
    object.left = x1;
    object.top = y1;
    object.width = x2 - x1;
    object.height = y2 - y1;
    object.classId = class_id;
    object.detectionConfidence = score;
    objects.push_back(object);
    return true;
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
    const int classes = static_cast<int>(detection.numClassesConfigured);
    const int attributes = 4 + classes;
    TensorView view{};
    if (!make_tensor_view(layer, attributes, view, "YOLO dynamic parser")) {
        return false;
    }

    objects.clear();
    objects.reserve(static_cast<std::size_t>(std::min(view.rows, 1000)));
    for (int row = 0; row < view.rows; ++row) {
        int best_class = 0;
        float best_score = view.value_at(row, 4);
        for (int class_id = 1; class_id < classes; ++class_id) {
            const float score = view.value_at(row, 4 + class_id);
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
        append_object(view, row, best_class, best_score, network, objects);
    }
    return true;
}

// yolo11n.onnx is a standard 80-class COCO detector, not a purpose-trained
// two-class eating/drinking network. In production it runs as a secondary GIE
// on each PeopleNet person ROI. The business rule intentionally mirrors the
// previously validated opsvision EatingDrinking rule:
//   - YOLO first chooses the candidate's highest COCO class
//   - prop confidence >= 0.45
//   - prop center must lie in the top 40% of the person box (mouth/upper-body area)
//   - drinking: bottle, cup, wine glass, bowl
//   - eating: apple, banana, sandwich, orange, pizza, donut, cake, hot dog
// The parser exposes only two business classes to the existing behavior layer:
//   0 = EATING
//   1 = DRINKING
extern "C" bool NvDsInferParseCustomYoloEatDrinkCoco(
    const std::vector<NvDsInferLayerInfo>& output_layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& detection,
    std::vector<NvDsInferParseObjectInfo>& objects) {
    constexpr int kCocoClasses = 80;
    constexpr int kAttributes = 4 + kCocoClasses;
    constexpr int kEating = 0;
    constexpr int kDrinking = 1;
    constexpr float kMouthRegionRatio = 0.40F;

    // COCO class ids. Keep this set aligned with opsvision/eating_drinking.py.
    constexpr std::array<std::pair<int, int>, 12> kBusinessClasses{{
        {39, kDrinking},  // bottle
        {40, kDrinking},  // wine glass
        {41, kDrinking},  // cup
        {45, kDrinking},  // bowl
        {46, kEating},    // banana
        {47, kEating},    // apple
        {48, kEating},    // sandwich
        {49, kEating},    // orange
        {52, kEating},    // hot dog
        {53, kEating},    // pizza
        {54, kEating},    // donut
        {55, kEating},    // cake
    }};

    if (output_layers.empty() || detection.numClassesConfigured < 2 ||
        detection.perClassPreclusterThreshold.size() < 2) {
        std::cerr << "YOLO eat/drink parser: requires output and 2 business classes"
                  << std::endl;
        return false;
    }

    TensorView view{};
    if (!make_tensor_view(
            output_layers.front(), kAttributes, view, "YOLO eat/drink parser")) {
        return false;
    }

    objects.clear();
    objects.reserve(static_cast<std::size_t>(std::min(view.rows, 512)));
    for (int row = 0; row < view.rows; ++row) {
        int best_coco_class = 0;
        float best_score = view.value_at(row, 4);
        for (int coco_class = 1; coco_class < kCocoClasses; ++coco_class) {
            const float score = view.value_at(row, 4 + coco_class);
            if (score > best_score) {
                best_score = score;
                best_coco_class = coco_class;
            }
        }
        if (!std::isfinite(best_score) || best_score < 0.0F || best_score > 1.0F) {
            continue;
        }

        int best_business_class = -1;
        for (const auto& mapping : kBusinessClasses) {
            if (mapping.first == best_coco_class) {
                best_business_class = mapping.second;
                break;
            }
        }
        if (best_business_class < 0) {
            continue;
        }

        const float threshold =
            detection.perClassPreclusterThreshold.at(best_business_class);
        if (best_score < threshold) {
            continue;
        }
        const float center_y = view.value_at(row, 1);
        if (!std::isfinite(center_y) ||
            center_y > static_cast<float>(network.height) * kMouthRegionRatio) {
            continue;
        }
        append_object(
            view,
            row,
            best_business_class,
            best_score,
            network,
            objects);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloDynamic);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloEatDrinkCoco);
