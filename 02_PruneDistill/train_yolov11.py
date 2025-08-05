import utilPruneDistill
from ultralytics import YOLO
import os
# from utils.yolo.attention import add_attention

imgSize = 176 # this value should be constant through steps

# def step1_train():
#     model = YOLO(pretrained_model_path)
#     model.train(data=yaml_path, device="0", imgsz=640, epochs=50, batch=2, workers=0, save_period=1)  # train the model

def step2_constraint_train(
        step1_train_model_path, 
        step2_constraint_train_model_path, yaml_dataset_path):
    assert os.path.exists(step1_train_model_path), f"Model path {step1_train_model_path} does not exist."

    model = YOLO(step1_train_model_path)
    model.train(data=yaml_dataset_path, device="0", imgsz=imgSize, epochs=50, batch=2, amp=False, workers=0, save_period=1,
                name=step2_constraint_train_model_path)  # train the model

def step3_pruning(sModel_step2_constraint_trained, sModel_step3_pruned, pruning_rate=0.8):
    # from utils.yolo.seg_pruning import do_pruning  use for seg
    from utils.yolo.det_pruning import do_pruning  # use for det
    do_pruning(sModel_step2_constraint_trained, sModel_step3_pruned , pruning_rate)


# def step4_finetune():
#     model = YOLO(step3_prune_after_model_path)  # load a pretrained model (recommended for training)
#     for param in model.parameters():
#         param.requires_grad = True
#     model.train(data=yaml_path, device="0", imgsz=640, epochs=200, batch=2, workers=0,
#                 name=step4_finetune_model_path)  # train the model


# def step5_distillation():
#     layers = ["6", "8", "13", "16", "19", "22"]
#     model_t = YOLO(step5_teacher_model_path)  # the teacher model
#     model_s = YOLO(step5_student_model_path)  # the student model
#     model_s = add_attention(model_s)

#     model_s.train(data=yaml_path, Distillation=model_t.model, loss_type='mgd', layers=layers, amp=False, imgsz=1280,
#                   epochs=300,
#                   batch=2, device=0, workers=0, lr0=0.001, name=step5_output_model_path)


if __name__ == '__main__':
    # ----- step1: pretraining -----
    sModel_step1_pretrained = "./runs/lapaTrain/176x1000epoch_MileStone/weights/best.pt"
    # step1_train()

    # ----- step2: constraint training -----
    sModel_step2_constraint_trained = "./runs/lapaTrain/176x500epoch_PruneStep2_/weights/best.pt"
    # step2_constraint_train(sModel_step1_pretrained, sModel_step2_constraint_trained)

    # ----- step3: pruning -----
    pruning_rate = 0.8
    sModel_step3_pruned = sModel_step1_pretrained.replace("best.pt", f"pruned_{pruning_rate}.pt")
    step3_pruning(sModel_step2_constraint_trained, sModel_step3_pruned)

    # ----- step4: finetuning -----
    # step4_finetune()
    
    # ----- step5: distillation -----
    # step5_distillation()
