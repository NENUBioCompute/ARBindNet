class Config:
    # ========== 预训练权重 ==========
    pretrained_model_path = "/datapool/home/info_wang/tongyanting/GCL/ARBindNet/model/pretrain/output/best_model.pt"   # 预训练好的权重

    # ========== 训练超参数 ==========
    batch_size = 64
    finetune_epochs = 3
    lr = 1e-5
    weight_decay = 1e-5
    patience = 10
    seed = 42
    num_workers = 10
    device = 'cuda'

    # ========== 模型维度 ==========
    atom_node_dim = 8
    atom_edge_dim = 19
    res_node_s_dim = 23
    res_node_v_dim = 3
    res_edge_s_dim = 19
    res_edge_v_dim = 3
    num_atom_types = 5

    hidden_dim = 128
    egnn_layers = 3
    egnn_heads = 4
    use_attention = True
    gvp_hidden_s = 128
    gvp_hidden_v = 16
    gvp_layers = 3
    gvp_dropout = 0.1

    # ========== 微调策略 ==========
    freeze_encoders = False          # 是否冻结编码器
    # 分类头参数
    classifier_dropout = 0.3