from ultralytics import YOLO
import torch
from ultralytics.nn.modules import Bottleneck, Conv, DWConv, C2f, SPPF, Detect, Pose, C3k2, C3k
from torch.nn.modules.container import Sequential
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class PRUNE():
    def __init__(self) -> None:
        self.threshold = None

    def get_threshold(self, model, factor=0.8):
        '''
        collect all the BatchNorm2d weights and biases, and use the factor% small weight as threshold
        '''
        ws = []
        bs = []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                w = m.weight.abs().detach()
                b = m.bias.abs().detach()
                ws.append(w)
                bs.append(b)
                print(f"{name}: wMax = {w.max().item():.3f}, wMin = {w.min().item():.3f}")
                # print(f"{name}: wMax = {w.max().item():.3f}, wMin = {w.min().item():.3f}, bMax = {b.max().item():.3f}, bMin = {b.min().item():.3f}")
        # keep
        ws = torch.cat(ws)
        self.threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
        print("----- self.threshold = ", self.threshold.item(), " -----")

        # print for each layer, how many channels will be kept
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                w = m.weight.abs().detach()
                b = m.bias.abs().detach()
                wAfterPrune = w[w >= self.threshold]
                print(f"{name}: weight number before prune = {len(w)}, after = {len(wAfterPrune)}")

    def prune_conv(self, conv1: Conv, conv2: Conv):
        ## TODO: assert the yype of conv1 must be Conv, and conv2 can be Conv or Sequential
        ## Normal Pruning
        gamma = conv1.bn.weight.data.detach()
        beta = conv1.bn.bias.data.detach()

        keep_idxs = []
        local_threshold = self.threshold
        atLeastKeep = min(16, len(gamma))
        while len(keep_idxs) < atLeastKeep:  ## 若剩余卷积核<8, 则降低阈值重新筛选
            keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
            local_threshold = local_threshold * 0.5
        n = len(keep_idxs)
        # n = max(int(len(idxs) * 0.8), p)
        conv1.bn.weight.data = gamma[keep_idxs]
        print("  n before = ", len(gamma), ", n after = ", n)
        conv1.bn.bias.data = beta[keep_idxs]
        conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
        conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
        conv1.bn.num_features = n
        conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs]
        conv1.conv.out_channels = n

        if isinstance(conv2, list) and len(conv2) > 3 and conv2[-1]._get_name() == "Proto":
            proto = conv2.pop()
            proto.cv1.conv.in_channels = n
            proto.cv1.conv.weight.data = proto.cv1.conv.weight.data[:, keep_idxs]
        if conv1.conv.bias is not None:
            conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]

        ## Regular Pruning
        if not isinstance(conv2, list): # to compatible with list input
            conv2 = [conv2]
        for item in conv2:
            if item is None: continue
            if isinstance(item, Conv):
                conv = item.conv
            else:
                conv = item
            if isinstance(item, Sequential):
                conv1 = item[0]
                assert isinstance(conv1, DWConv)
                conv = item[1].conv
                conv1.conv.in_channels = n
                conv1.conv.out_channels = n
                conv1.conv.groups = n
                conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs, :]
                conv1.bn.bias.data = conv1.bn.bias.data[keep_idxs]
                conv1.bn.weight.data = conv1.bn.weight.data[keep_idxs]
                conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
                conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
                conv1.bn.num_features = n
            conv.in_channels = n
            conv.weight.data = conv.weight.data[:, keep_idxs]

    def prune_C3k2(self, m: C3k2):
        assert isinstance(m, C3k2), "m1 must be an instance of C3k2"
        conv1 = m.cv1
        if isinstance(m.m[0], Bottleneck):
            assert len(m.m) == 1
            conv2b = m.m[0].cv1
            print("I will prune you")
            assert isinstance(conv2b, Conv), "m1 must be an instance of C3k2"
        elif isinstance(m.m[0], C3k):
            print("I will prune you, but later")
            return
        else:
            raise TypeError(f"Unsupported type {type(m.m[0])} in C3k2") 
        conv2a = m.cv2

        # ----- Prune conv1 output ----- #
        # --- find keep idx --- #
        gamma = conv1.bn.weight.data.detach()
        beta = conv1.bn.bias.data.detach()
        nOld = len(gamma)
        keep_idxs = []
        local_threshold = self.threshold
        atLeastKeep = min(16, len(gamma))
        while len(keep_idxs) < atLeastKeep:  ## 若剩余卷积核<16, 则降低阈值重新筛选
            keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
            local_threshold = local_threshold * 0.5
        nNew = len(keep_idxs)
        if nNew + 1 >= nOld:  # 如果剪枝后卷积核数量没有减少，则不进行剪枝
            return
        
        # --- Prune conv1 output --- 
        # if isinstance(m.m[0], Bottleneck):
        keep_idxs = torch.cat((keep_idxs[keep_idxs < (nOld // 2)], torch.range(nOld//2, nOld - 1, dtype=torch.int64)), dim = 0)
        nNew = len(keep_idxs)
        print("  nOld = ", nOld, ", nNew = ", nNew)
        conv1.bn.weight.data = gamma[keep_idxs]
        conv1.bn.bias.data = beta[keep_idxs]
        conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
        conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
        conv1.bn.num_features = nNew
        conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs]
        conv1.conv.out_channels = nNew
        if conv1.conv.bias is not None:
            conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]
        # --- allocate the keep_idxs to the 1st half and the 2nd half --- #
        print(keep_idxs)
        m.nCV1out_1stHalf = (keep_idxs < nOld // 2).sum()
        m.nCV1out_2ndHalf = (keep_idxs >= nOld // 2).sum()

        # ----- Prune conv2a input ----- # Too complicated because of BottoleNeck structure

        keep_idxs_2ndHalf = keep_idxs[keep_idxs >= (nOld//2)] - (nOld//2)
        conv2b.conv.in_channels = len(keep_idxs_2ndHalf)
        conv2b.conv.weight.data = conv2b.conv.weight.data[:, keep_idxs_2ndHalf]
        
        # ----- Prune conv2b input ----- #
        keep_idxs_1stHalf = keep_idxs[keep_idxs < (nOld//2)]
        keep_idxs_1stHalf_remain = torch.range(nOld//2, conv2a.conv.in_channels - 1, dtype=torch.int64)
        keep_idxs_1stHalf = torch.cat((keep_idxs_1stHalf, keep_idxs_1stHalf_remain), dim=0)
        print(keep_idxs_1stHalf)
        # keep_idx_1stHalf = np.concatenate(keep_idxs_1stHalf, np.range(nOld//2keep_idxs_2ndHalf + (nOld//2)))
        conv2a.conv.weight.data = conv2a.conv.weight.data[:, keep_idxs_1stHalf]
        conv2a.conv.in_channels = len(keep_idxs_1stHalf)

        pass

    def prune(self, m1, m2):
        if isinstance(m1, C3k2):  # C3k2 as a top conv
            m1 = m1.cv2
        if isinstance(m1, Sequential):
            assert isinstance(m1[0], DWConv)
            assert isinstance(m1[1], Conv)
            m1 = m1[1]
        if not isinstance(m2, list):  # m2 is just one module
            m2 = [m2]
        for i, item in enumerate(m2):
            if isinstance(item, C3k2) or isinstance(item, SPPF):
                m2[i] = item.cv1


        self.prune_conv(m1, m2)

def do_pruning(modelpath, savepath, pruning_rate=0.8):
    assert os.path.exists(modelpath), f"Model path {modelpath} does not exist."

    pruning = PRUNE()

    ### 0. 加载模型
    yolo = YOLO(modelpath)  # build a new model from scratch
    pruning.get_threshold(yolo.model, pruning_rate)  # 这里的0.8为剪枝率。

    ### 1. 剪枝C3k2 中的Bottleneck
    for name, m in yolo.model.named_modules():
        if isinstance(m, Bottleneck):
            print("剪枝C3k2中的Bottleneck的隐藏层: ", name, end=' ')
            pruning.prune_conv(m.cv1, m.cv2)

    ## 1.5 剪枝C3k2的conv1
    for name, m in yolo.model.named_modules():
        if isinstance(m, C3k2):
            print("剪枝C3k2的C1, 它是m[0]和C2的输入: ", name)
            pruning.prune_C3k2(m)

    ### 2. 指定剪枝不同模块之间的卷积核
    seq = yolo.model.model
    for i in [3,5]:
        print(f"剪枝模块{i}和模块{i+1}的连接: ", end=' ')
        pruning.prune(seq[i], seq[i + 1])

    ### 3. 对检测头进行剪枝
    # 在P3层: seq[15]之后的网络节点与其相连的有 seq[16]、detect.cv2[0] (box分支)、detect.cv3[0] (class分支)
    # 在P4层: seq[18]之后的网络节点与其相连的有 seq[19]、detect.cv2[1] 、detect.cv3[1]
    # 在P5层: seq[21]之后的网络节点与其相连的有 detect.cv2[2] 、detect.cv3[2]
    seq = yolo.model.model
    detect: Detect = seq[-1]
    # proto = detect.proto
    last_inputs = [seq[10], seq[13]]
    colasts = [seq[11], None]
    for idx, (last_input, colast, cv2, cv3, cv4) in enumerate(zip(last_inputs, colasts, detect.cv2, detect.cv3, detect.cv4)):
        print(f"剪枝输出模块anchor {idx}")
        pruning.prune(last_input, [colast, cv2[0], cv3[0], cv4[0]])
        pruning.prune(cv2[0], cv2[1])
        pruning.prune(cv2[1], cv2[2])
        pruning.prune(cv3[0], cv3[1])
        pruning.prune(cv3[1], cv3[2])

        pruning.prune(cv4[0], cv4[1])
        pruning.prune(cv4[1], cv4[2])
        pass

    ### 4. 模型梯度设置与保存
    for name, p in yolo.model.named_parameters():
        p.requires_grad = True

    # yolo.val(data='data.yaml', batch=2, device=0, workers=0)
    torch.save(yolo.ckpt, savepath)



if __name__ == "__main__":
    modelpath = "runs/segment/Constraint/weights/best.pt"
    savepath = "runs/segment/Constraint/weights/last_prune.pt"
    do_pruning(modelpath, savepath)
