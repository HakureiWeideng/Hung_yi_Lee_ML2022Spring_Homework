# Improving

## 0

```python
    # NN strcture
    input_dim = dim_frame * n_frames # 39 * 11 = 429
    output_dim = n_class   # 41
    n_hidden_layer = 2
    hidden_dim = 256

    # training
    begin_example = 1
    n_example = math.inf
    n_epoch = 300
    early_stop_epoch = 150
    batch_size = 256
    learning_rate = 1e-4

    # mutiple step
    loading_step_size = 720
    n_loading_step = math.ceil(4268 / loading_step_size)

```

- train loss = 0.7538
- valid loss = 1.1194
- Kaggle socre = 0.60689（above simple line）
