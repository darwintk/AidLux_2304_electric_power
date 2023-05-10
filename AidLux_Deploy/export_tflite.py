import os
import sys
sys.path.append("/home/lml/code/onnx2tflite")
from converter import onnx_converter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def onnx2tflite(onnx_path):
    onnx_converter(
        onnx_mode_path = onnx_path,
        need_simplify = False,
        output_path = os.path.dirname(onnx_path),
        target_formats = ['tflite'],
        weight_quant = False,
        int8_model = False,
        int8_mean = None,
        int8_std = None,
        image_root = None
    )

if __name__ == "__main__":
    onnx2tflite("./weights/r-retinanet-statedict.onnx")
    