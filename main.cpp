#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <opencv2/opencv.hpp>
#include "yolov8_post.h"


static void pack_results(
    const Objects* objs,
    const std::vector<int>& indices,
    py::list& results)
{
    py::list bboxes;
    py::list scores;
    py::list cls_ids;

    for (const auto& idx : indices) {
        const cv::Rect& rect = objs->bboxes[idx];
        py::tuple bbox = py::make_tuple(rect.x, rect.y, rect.width, rect.height);
        bboxes.append(bbox);
        scores.append(objs->scores[idx]);
        cls_ids.append(objs->cls_ids[idx]);
    }

    results.append(bboxes);
    results.append(scores);
    results.append(cls_ids);
}


py::list process(
    std::vector<py::array_t<float, py::array::c_style> >& featmaps,
    const std::vector<int>& img_shape)
{
    Objects objs;
    std::vector<int> indices;
    int obj_num = 0;
    int featmap_shape[4] = {0};

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    for (auto& featmap : featmaps) {
        const auto& shape = featmap.shape();
        for (ssize_t i = 0; i < featmap.ndim(); i++) {
            featmap_shape[i] = shape[i];
        }

        float* data_ptr = static_cast<float*>(featmap.mutable_data());

        proposal(data_ptr, featmap_shape, &objs, obj_num);
    }

    printf("proposal num: %d\n", obj_num);

    nms(&objs, indices);
    scale_boxes(&objs, indices, img_shape);

    gettimeofday(&t2, NULL);
    double timeuse  = 1000000*(t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec;
    printf("proposal timeuse: %f ms\n", timeuse*1e-3);

    py::list results;
    pack_results(&objs, indices, results);

    return results;
}


PYBIND11_MODULE(v8_post, m) 
{
    m.def("process", &process, "yolov8 post-process test");
}

