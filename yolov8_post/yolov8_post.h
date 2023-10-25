#ifndef YOLOV8_POST_H
#define YOLOV8_POST_H

#include <vector>


struct Objects
{
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> cls_ids;

    Objects() : bboxes(8400), scores(8400), cls_ids(8400)
    {

    }
};


void proposal(
    float* data_ptr, 
    const int* featmap_shape,
    Objects* objs,
    int& obj_num);


void nms(
    const Objects* objs, 
    std::vector<int>& indices);


void scale_boxes(
    Objects* objs,
    const std::vector<int>& indices,
    const std::vector<int>& img_shape);

#endif  //YOLOV8_POST_H
