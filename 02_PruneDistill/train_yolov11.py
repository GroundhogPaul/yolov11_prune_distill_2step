import os

from ultralytics import YOLO

# from utils.yolo.attention import add_attention

imgSize = 176  # this value should be constant through steps

# def step1_train():
#     model = YOLO(pretrained_model_path)
#     model.train(data=yaml_path, device="0", imgsz=640, epochs=50, batch=2, workers=0, save_period=1)  # train the model


def step2_constraint_train(sModel_step1_pretrained, sModelName_step2_constraint_trained, sYamlData, sYamlHyper):
    assert os.path.exists(sModel_step1_pretrained), f"Model path {sModel_step1_pretrained} does not exist."
    assert os.path.exists(sYamlData), f"yaml {sYamlData} does not exist."
    assert os.path.exists(sYamlHyper), f"yaml {sYamlHyper} does not exist."

    model = YOLO(sModel_step1_pretrained)
    model.train(
        cfg=sYamlHyper, data=sYamlData, amp=False, save_period=3, name=sModelName_step2_constraint_trained
    )  # train the model


def step3_pruning(sModel_step2_constraint_trained, sModel_step3_pruned, pruning_rate=0.8):
    assert os.path.exists(sModel_step2_constraint_trained), (
        f"Model path {sModel_step2_constraint_trained} does not exist."
    )

    # from utils.yolo.seg_pruning import do_pruning  use for seg
    from utils.yolo.det_pruning import do_pruning  # use for det

    do_pruning(sModel_step2_constraint_trained, sModel_step3_pruned, pruning_rate)


def step4_finetune(sModel_step3_pruned, sName_step4_finetune, sDataYaml, sHyperYaml):
    assert os.path.exists(sModel_step3_pruned), f"Model path {sModel_step3_pruned} does not exist."

    model = YOLO(sModel_step3_pruned)  # load a pretrained model (recommended for training)
    for param in model.parameters():
        param.requires_grad = True
    model.train(
        data=sDataYaml, device="0", imgsz=640, epochs=200, batch=2, workers=0, name=sName_step4_finetune
    )  # train the model


def step5_distillation(sModel_step5_teacher_model, sModel_step5_student_model, sYamlData, sYamlHyper):
    assert os.path.exists(sModel_step5_teacher_model), f"Model path {sModel_step5_teacher_model} does not exist."
    assert os.path.exists(sModel_step5_student_model), f"Model path {sModel_step5_student_model} does not exist."

    layers = ["4", "6", "7", "10", "12"]  # the output of C3k2
    model_t = YOLO(sModel_step5_teacher_model)
    model_s = YOLO(sModel_step5_student_model)
    #     model_s = add_attention(model_s)

    model_s.train(
        data=sYamlData,
        Distillation=model_t.model,
        loss_type="mgd",
        layers=layers,
        amp=False,
        imgsz=1280,
        epochs=300,
        batch=64,
        device=0,
        workers=0,
        lr0=0.001,
        name="Distillation",
    )


if __name__ == "__main__":
    # ----- step1: pretraining -----
    sYamlData = "./01_BasicRunPoseSetting/DataSet-lapa-pose-plus.yaml"
    # step1_train()

    # ----- step2: constraint training -----
    sModel_step1_pretrained = "./runs/LapaTrain/176x1000epoch_MileStone/weights/best.pt"
    sModelName_step2_constraint_trained = "176x500epoch_PruneStep2"
    sYaml_step2_Hyper = "./02_PruneDistill/02_PruneStep2Conf.yaml"
    # step2_constraint_train(sModel_step1_pretrained = sModel_step1_pretrained,
    #                     sModelName_step2_constraint_trained = sModelName_step2_constraint_trained,
    #                     sYamlData = sYamlData,
    #                     sYamlHyper = sYaml_step2_Hyper)

    # ----- step3: pruning -----
    pruning_rate = 0.8
    sModel_step2_constraint_trained = os.path.join(
        "./runs/LapaTrain", sModelName_step2_constraint_trained, "weights/best.pt"
    )
    sModel_step3_pruned = "pruned_0.8.pt"
    # step3_pruning(sModel_step2_constraint_trained, sModel_step3_pruned, pruning_rate)

    # ----- step4: finetuning -----
    sModel_step3_pruned = sModel_step2_constraint_trained.replace("best.pt", sModel_step3_pruned)
    sName_step4_finetuned = "176x500epoch_PruneStep4"
    sYaml_step4_Hyper = "./02_PruneStep4Conf.yaml"
    # step4_finetune(sModel_step3_pruned = sModel_step3_pruned,
    #                sName_step4_finetune = sName_step4_finetuned,
    #                sDataYaml = sYamlData,
    #                sHyperYaml = sYaml_step4_Hyper)

    # ----- step5: distillation -----
    step5_student_model_path = "./runs/lapaTrain/176x500epoch_PruneStep4/weights/best.pt"
    sYaml_step5_Hyper = "./02_PruneStep5Conf.yaml"
    # step5_distillation(
    #     sModel_step5_teacher_model = sModel_step2_constraint_trained,
    #     sModel_step5_student_model = step5_student_model_path,
    #     sYamlData = sYamlData,
    #     sYamlHyper = sYaml_step5_Hyper)
