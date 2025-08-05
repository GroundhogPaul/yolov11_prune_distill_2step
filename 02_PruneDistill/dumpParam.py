import utilPruneDistill
import torch
import os
from ultralytics import YOLO
from copy import deepcopy

def GPUorCPU(model):
    pass

def dump_param(sModelPath, sLayerName, sOutputPath):
    '''
    Dumps the parameters of the model to a file.
    '''
    
    assert os.path.exists(sModelPath), f"Model path {sModelPath} does not exist."

    # Load the model
    model = YOLO(sModelPath, task="pose")
    print(model.device)

    # if torch.cuda.is_available():
    #     model = model.to('cuda')
    #     print("model has been transferred to GPU ")
    #     print("current device:", model.device)
    # else:
    #     print("no availabe gpu")

    for name, m in model.model.model.named_modules():
        print(name)
        if name == "14.cv4.1.0":
            m_cpu = deepcopy(m).cpu()
            print(type(m_cpu))
            # print(m_cpu.weight.shape)

if __name__ == "__main__":
    sModelPath = "./runs/LapaTrain/176x1000epoch_MileStone/weights/best.pt"
    dump_param(sModelPath, "model.model.0", "dumped_params.txt")