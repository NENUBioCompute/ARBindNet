import torch

class Config:
    # ========== 预训练权重 ==========
    pretrained_model_path = "./output/best_model.pt"   # 预训练好的权重
    # ----- Training hyperparameters -----
    batch_size = 128
    epochs = 500
    lr = 1e-4
    weight_decay = 1e-5
    patience = 10
    seed = 42
    num_workers = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----- Model dimensions -----
    atom_node_dim = 8
    atom_edge_dim = 19
    res_node_s_dim = 23
    res_node_v_dim = 3
    res_edge_s_dim = 19
    res_edge_v_dim = 3
    num_atom_types = 5

    # ----- Encoder parameters -----
    hidden_dim = 128
    egnn_layers = 3
    egnn_heads = 4
    use_attention = True
    gvp_hidden_s = 128
    gvp_hidden_v = 32
    gvp_layers = 3
    gvp_dropout = 0.1
    mask_ratio = 0.1

    # ----- Projection head -----
    proj_hidden = 256
    proj_out = 128
    temperature = 0.07

    # ----- Loss weights -----
    lambda_contrast = 1.0
    lambda_recon = 1.0
    lambda_coord = 0.1

    # # ========== 训练超参数 ==========
    # batch_size = 128
    # finetune_epochs = 3
    # lr = 1e-5
    # weight_decay = 1e-5
    # patience = 10
    # seed = 42
    # num_workers = 10
    # device = 'cuda'

    # # ========== 模型维度 ==========
    # atom_node_dim = 8
    # atom_edge_dim = 19
    # res_node_s_dim = 23
    # res_node_v_dim = 3
    # res_edge_s_dim = 19
    # res_edge_v_dim = 3
    # num_atom_types = 5

    # hidden_dim = 128
    # egnn_layers = 3
    # egnn_heads = 4
    # use_attention = True
    # gvp_hidden_s = 128
    # gvp_hidden_v = 32
    # gvp_layers = 3
    # gvp_dropout = 0.1

    # proj_hidden =256
    # proj_out = 128
    # # ========== 微调策略 ==========
    # freeze_encoders = False          # 是否冻结编码器
    # # 分类头参数
    # classifier_dropout = 0.3



