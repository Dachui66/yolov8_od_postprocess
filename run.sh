export PYTHONPATH=./module:$PYTHONPATH

python onnx_run.py data/yolov8s.onnx data/test.jpg
