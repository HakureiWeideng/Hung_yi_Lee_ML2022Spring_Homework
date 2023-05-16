# Improving

Kaggle Score：Private Score .

## 0

初始版本。

```python
class MyNN(nn.Module):
    def __init__(self, input_dim):
        super(MyNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        t = self.layers(x)
        t = t.squeeze()   # 一次输出一个 batch 的结果:(X, 1)，将其变为一行。
        return t 

config = {
    'seed': 923,
    'use_feature': [0, 1, 2, 15],   # state 的 37 列全部保留，列表为空代表全选，列表内数字代表要选择的非 state 的 feature 的编号（从 0 开始）。
    'valid_ratio': 0.2,
    'batch_size': 64,
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'n_epoch': 500,
    'early_stop': 200,
    'save_path': './save_model/model.ckpt',
    'predict_filename': 'PredictResult.csv'
}
```

Kaggle Score：105.87056

## 1

- 调整特征选取。
- 增加 epoch 。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
config = {
    'seed': 923,
    'use_feature': list(range(0, 16)),   # id 列默认不选，state 的 37 列以 -1 表示，列表为空代表全选，列表内数字代表要选择的非 state 的 feature 的编号（从 0 开始）。
    'valid_ratio': 0.2,
    'batch_size': 64,
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'n_epoch': 1000,
    'early_stop': 300,
    'save_path': 'model.ckpt',
    'predict_filename': 'predict.csv'
}
```

- train loss：1.1155
- valid loss：1.0686
- Kaggle Score：None

## 2

- 增大 batch size 。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
config = {
    'seed': 923,
    'use_feature': list(range(0, 16)),   # id 列默认不选，state 的 37 列以 -1 表示，列表为空代表全选，列表内数字代表要选择的非 state 的 feature 的编号（从 0 开始）。
    'valid_ratio': 0.2,
    'batch_size': 256,
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'n_epoch': 1000,
    'early_stop': 300,
    'save_path': 'model.ckpt',
    'predict_filename': 'predict.csv'
}
```

- valid loss: 0.9514
- Kaggle Score: 120.35807

## sample code

- Train loss: 1.7633
- Valid loss: 2.4166
- Kaggle Score: 1.85279

### 3

输出格式问题？

特征不够？

- 使用所有特征。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
config = {
    'seed': 923,
    'use_feature': [],   # id 列默认不选，state 的 37 列以 -1 表示，列表为空代表全选，列表内数字代表要选择的非 state 的 feature 的编号（从 0 开始）。
    'valid_ratio': 0.2,
    'batch_size': 256,
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'n_epoch': 1000,
    'early_stop': 300,
    'save_path': 'model.ckpt',
    'predict_filename': 'predict.csv'
}
```

- Valid loss: 1.5929
- Kaggle Score: 109.27010

## 4

猜测为格式问题，但概率较小，检查中。

seed 问题？概率较小。

特征选择函数？

陡峭的局部最优解？

- 特征只用前 4 天的 tested_positive 。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
config = {
    'seed': 923,
    'use_feature': [15],   # id 列默认不选，state 的 37 列以 -1 表示，列表为空代表全选，列表内数字代表要选择的非 state 的 feature 的编号（从 0 开始）。
    'valid_ratio': 0.2,
    'batch_size': 256,
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'n_epoch': 1000,
    'early_stop': 300,
    'save_path': 'model.ckpt',
    'predict_filename': 'predict.csv'
}
```

- valid loss: 0.9127
- Kaggle Score: 118.50997

## 5

epoch 不够？

- epoch 改为 3000 。（排除）

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
config = {
    'seed': 923,
    'use_feature': [],   # id 列默认不选，state 的 37 列以 -1 表示，列表为空代表全选，列表内数字代表要选择的非 state 的 feature 的编号（从 0 开始）。
    'valid_ratio': 0.2,
    'batch_size': 256,
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'n_epoch': 3000,
    'early_stop': 400,
    'save_path': 'model.ckpt',
    'predict_filename': 'predict.csv'
}
```

- valid loss: 1.4725
- Kaggle Score: 112.21478

## 6

- 改 seed 。（排除）

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
config = {
    'seed': 5201314,
    'use_feature': [],   # id 列默认不选，state 的 37 列以 -1 表示，列表为空代表全选，列表内数字代表要选择的非 state 的 feature 的编号（从 0 开始）。
    'valid_ratio': 0.2,
    'batch_size': 256,
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'n_epoch': 3000,
    'early_stop': 400,
    'save_path': 'model.ckpt',
    'predict_filename': 'predict.csv'
}
```

- valid loss：1.6524
- Kaggle Score: 117.86591

## 7

代码问题。（测试数据乱序问题）

- 更改 Dataloader 的 shuffle 为 False 。

```python
config = {
    'seed': 923,
    'use_feature': list(range(16)),   # id 列默认不选，state 的 37 列以 -1 表示，列表为空代表全选，列表内数字代表要选择的非 state 的 feature 的编号（从 0 开始）。
    'valid_ratio': 0.2,
    'batch_size': 256,
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'n_epoch': 1000,
    'early_stop': 300,
    'save_path': 'model.ckpt',
    'predict_filename': 'predict.csv'
}
```

- valid loss: 0.9513
- Kaggle Score: 1.19898

## 8

- 更改 feature 。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
config = {
    'seed': 923,
    'use_feature': [15],   # id 列默认不选，state 的 37 列以 -1 表示，列表为空代表全选，列表内数字代表要选择的非 state 的 feature 的编号（从 0 开始）。
    'valid_ratio': 0.2,
    'batch_size': 256,
    'learning_rate': 1e-5,
    'momentum': 0.9,
    'n_epoch': 1000,
    'early_stop': 300,
    'save_path': 'model.ckpt',
    'predict_filename': 'predict.csv'
}
```

- valid loss: 0.9127042
- Kaggle Score: 1.11565
