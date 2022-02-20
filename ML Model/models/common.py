
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

def export_formats():
    # YOLOv5 export formats
    x = [['PyTorch', '-', '.pt'],
         ['ONNX', 'onnx', '.onnx'],]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix'])

class DetectMultiBackend(nn.Module):
    def __init__(self, weights='yolov5s.pt', device=None, dnn=False, data=None):
        # from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt,  onnx = self.model_type(w)  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        # w = attempt_download(w)  # download if not local
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names
        if onnx:  # ONNX Runtime
            cuda = torch.cuda.is_available()
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        if self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        if self.pt  or self.onnx :  # warmup types
            if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
                self.forward(im)  # warmup

    @staticmethod
    def model_type(p='path/to/model.pt'):
        suffixes = list(export_formats().Suffix)  # export suffixes
        pt, onnx = (s in p for s in suffixes)
        return pt, onnx