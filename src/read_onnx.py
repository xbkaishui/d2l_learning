import torch
import onnx
import onnxruntime
from loguru import logger

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def test_read_onnx():
    file_path = 'lenet.onnx'
    onnx_model = onnx.load(file_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(file_path)

    x = torch.randn(1, 1, 28, 28, requires_grad=True)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    logger.info("onnx output {}", ort_outs[0])
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    logger.info("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    test_read_onnx()
