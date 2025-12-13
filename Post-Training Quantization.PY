import tf2onnx
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

spec = (tf.TensorSpec((None, time_step, 3), tf.float32, name="input_1"),)
output_model_path = "single_layer_model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=spec, 
    opset=13, 
    output_path=output_model_path
)

class PondDataReader(CalibrationDataReader):
    def __init__(self, data_array):
        self.data_iter = iter(data_array)
    def get_next(self):
        try:
            sample = next(self.data_iter)
            return {"input_1": np.expand_dims(sample, axis=0)}
        except StopIteration:
            return None

dr = PondDataReader(X_train[:200])

quantized_model_path = "single_layer_model.int8.onnx"

quantize_static(
    model_input=output_model_path,
    model_output=quantized_model_path,
    calibration_data_reader=dr,
    quant_format=QuantFormat.QDQ,   
    per_channel=False,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)
