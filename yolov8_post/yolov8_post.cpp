#include <iostream>
#include <sys/time.h> 
#include <opencv2/opencv.hpp>

#include "yolov8_post.h"

#define V8_REG_MAX      (16)
#define V8_CONF_THRESH  (0.6)
#define V8_NMS_THRESH   (0.6)


void scale_boxes(
    Objects* objs, 
    const std::vector<int>& indices, 
    const std::vector<int>& img_shape)
{
    float height = img_shape[0];
    float width = img_shape[1];
    float gain = std::min(640 / width, 640 / height);
    float pad_w = (640 - width * gain) / 2;
    float pad_h = (640 - height * gain) / 2;

    cv::Rect* bbox = NULL;

    gain = 1.0 / gain;

    for (const auto& idx : indices) {
        bbox = &objs->bboxes[idx];
        bbox->x = (bbox->x - pad_w) * gain + 0.5;
        bbox->y = (bbox->y - pad_h) * gain + 0.5;
        bbox->width = bbox->width * gain + 0.5;
        bbox->height = bbox->height * gain + 0.5;
    }
}


static float sigmoid(float x)
{
    return 1.f / (1.f + exp(-x));
}


static float dfl(float* ptr)
{
    float list[V8_REG_MAX] = {
        0.f, 1.f, 2.f,  3.f,  4.f,  5.f,  6.f,  7.f, 
        8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f
    };

    float alpha = 0.f;
    float denominator = 0.f;

    cv::Mat feat(1, V8_REG_MAX, CV_32FC1, ptr);
    cv::Mat weights(1, V8_REG_MAX, CV_32FC1, list);

    alpha = *std::max_element(feat.begin<float>(), feat.end<float>());
    cv::exp(feat - alpha, feat);
    denominator = cv::sum(feat)[0];
    feat /= denominator;

    return feat.dot(weights);
}


void proposal(
    float* data_ptr, 
    const int* featmap_shape,
    Objects* objs,
    int& obj_num)
{
    int c = featmap_shape[1];
    int h = featmap_shape[2];
    int w = featmap_shape[3];

    float x1, y1, x2, y2;

    cv::Rect* bbox = NULL;

    const int REGX4 = 4 * V8_REG_MAX;
    int stride = (80 / w) << 3; 

    cv::Mat featmap_tmp(c, h * w, CV_32FC1, data_ptr);
    cv::transpose(featmap_tmp, featmap_tmp);
    cv::Mat featmap(h, w, CV_32FC(c), featmap_tmp.data);

    for (int i = 0; i < h; i++) {
        float* ptr = featmap.ptr<float>(i);

        for (int j = 0; j < w; j++) {
            float prob = -FLT_MAX;
            int cls_id = -1;

            for (int k = REGX4; k < c; k++) {
                float tmp = ptr[k];
                if (tmp > prob) {
                    prob = tmp;
                    cls_id = k;
                }
            }

            if (prob < V8_CONF_THRESH) { 
                ptr += c;
                continue;
            }

            float* ptr_c = ptr;
            x1 = j + 0.5f - dfl(ptr_c);
  
            ptr_c += V8_REG_MAX;
            y1 = i + 0.5f - dfl(ptr_c);

            ptr_c += V8_REG_MAX;
            x2 = j + 0.5f + dfl(ptr_c);

            ptr_c += V8_REG_MAX;
            y2 = i + 0.5f + dfl(ptr_c);

            objs->cls_ids[obj_num] = cls_id - REGX4;
            objs->scores[obj_num] = sigmoid(prob);
            bbox = &objs->bboxes[obj_num];
            bbox->x = x1 * stride;
            bbox->y = y1 * stride;
            bbox->width = (x2 - x1) * stride;
            bbox->height = (y2 - y1) * stride;

            obj_num++;
            ptr += c;
        }
    }
}


void nms(
    const Objects* objs, 
    std::vector<int>& indices)
{
    cv::dnn::NMSBoxes(
        objs->bboxes, 
        objs->scores, 
        V8_CONF_THRESH, 
        V8_NMS_THRESH, 
        indices);
}

