# version

## 1

- 训练随机取 item 时随机增强。
- 验证不增强，测试增强。

```python
class Config:
    # data path
    training_path = 'food11/training/'
    validaiton_path = 'food11/validation/'
    test_path = 'food11/test/'

    kaggle_path = '../input/ml2022spring-hw3b/'
    current_path = os.getcwd()
    if 'kaggle/working' in current_path:
        training_path = kaggle_path + training_path
        validaiton_path = kaggle_path + validaiton_path
        test_path = kaggle_path + test_path
    
    # environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 923
    model_save_path = 'model.ckpt'
    predict_save_path = 'prediction.csv'

    # data format
    image_size = (128, 128)

    # training
    n_epoch = 1000
    n_example = math.inf
    batch_size = 256
    laerning_rate = 1e-3
    weight_decay = 1e-5
```

- valid loss: 1.7854
- acc: 0.3712

- Kaggle Score: 0.40204

## 2

- 减小 batch_size 。
- 验证也增强。
- 同时保存训练最好模型和验证最好模型。

```python
class Config:
    # data path
    training_path = 'food11/training/'
    validaiton_path = 'food11/validation/'
    test_path = 'food11/test/'

    kaggle_path = '../input/ml2022spring-hw3b/'
    current_path = os.getcwd()
    if 'kaggle/working' in current_path:
        training_path = kaggle_path + training_path
        validaiton_path = kaggle_path + validaiton_path
        test_path = kaggle_path + test_path
    
    # environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 923
    train_best_model_save_path = 'train_best_model.ckpt'
    valid_best_model_save_path = 'valid_best_model.ckpt'
    train_best_predict_save_path = 'train_best_prediction.csv'
    valid_best_predict_save_path = 'valid_best_prediction.csv'

    # data format
    image_size = (128, 128)

    # training
    n_epoch = 1000
    n_example = math.inf
    batch_size = 128
    laerning_rate = 1e-3
    weight_decay = 1e-5
```

- best valid loss: 1.9731
- best train loss prediction kaggle score：0.10926
- best valid loss prediction kaggle score：0.12121

## 3

是否是 batch_size 有问题？概率较小。

- 不使用数据增强。

```python
    # training
    n_epoch = 1000
    n_example = math.inf
    batch_size = 128
    laerning_rate = 1e-3
    weight_decay = 1e-5
```

- best valid loss: 1.8448,
- acc = 0.3508
- best train loss prediction kaggle score：0.11907
- best valid loss prediction kaggle score：0.11950

## 4

预测格式出问题了？因为版本 1 的预测格式没有问题，所以概率较小。

- 减小 epoch 。（观察 loss cureve，300 足够）
- 减小 batch size 。（128 的 acc 最高可以到 0.4 几）
- 不使用数据增强。
- 存储最优验证准确率（acc）状态的模型。

```python
    # environment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 923
    best_train_model_save_path = 'best-train-model.ckpt'
    best_valid_model_save_path = 'best-valid-model.ckpt'
    best_acc_model_save_path = 'best-acc-model.ckpt'
    best_train_prediction_save_path = 'best-train-prediction.csv'
    best_valid_prediction_save_path = 'best-valid-prediction.csv'
    best_acc_predicttion_save_path = 'best-acc-prediction.csv'

    # data format
    image_size = (128, 128)

    # training
    n_epoch = 300
    n_example = math.inf
    batch_size = 64
    laerning_rate = 1e-3
    weight_decay = 1e-5
```

- best train loss = 0.0000
- best valid loss = 1.8207
- best acc = 0.4363
- best valid prediction Kaggle score: 0.10968
- best acc prediction  kaggle score: 0.11481

## 5

检查输出格式？与版本 1 一致，正常。

- 改回 batch size  为 256 。

```python
    # training
    n_epoch = 300
    n_example = math.inf
    batch_size = 256
    laerning_rate = 1e-3
    weight_decay = 1e-5
```

- best valid prediction Kaggle socre: 0.11566
- best acc prediction Kaggle socre: 0.10968

## 6

过拟合？看版本 6 的结果是没有使用数据增强，导致泛化性不好。

- 将训练集随机增强改为原图 1 份，随机增强 4 份，即打上数据增强标记构建数据集，再训练。
- 训练集增强，验证采用测试集增强的方法，测试增强。
- 减小 batch size

```python
batch size = 128 
```

超过 12 小时，使用保存的模型预测。

- best trian prediction Kaggle score: 0.35595
- best valid prediction Kaggle score: 0.22279
- best acc prediction Kaggle score: 0.36875

## 7

训练时没有保留原图？（已检查，有保留）

训练集，验证集，测试集，先增强并将数据存下来，节约数据转换的时间。？（降低 epoch 足够）

过拟合？（数据增强后泛化性增加，应当是过拟合）

- 减小 epoch 。
- 简化卷积层。

```python
epoch = 150

self.in_channels = [3, 32, 64, 128]
self.out_channels = [32, 64, 128, 128]
self.cnn_layers = [ CNNLayer(self.in_channels[i], self.out_channels[i]).to(Config.device) for i in range(5) ]

self.fc = nn.Sequential(
    nn.Linear(128*8*8, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 11)
)
```

- best acc prediction Kaggle score: 0.11609
- best train prediction Kaggle socer: 0.09987
- best valid prediction Kaggle socer: 0.11267

## 8

过拟合？

是否要添加残差？（先不弄残差了）

- 减小 batch size 。
- 还原卷积层结构。
-  缩减 FC 结构。

```python
    n_epoch = 150
    batch_size = 64  
    
    	self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 11),
        )
```

- best train Kaggle score: 0.13017

## 9

valid loss 一直上升。从头到尾检查代码。

- 修正 TTA 函数。
- 200 次 epoch 看效果。
- 恢复 FC 结构。

```python
    n_epoch = 200
```

忘记修改 `runing_test` 。

## 10

同 version 9 。

- best acc prediction Kaggle score: 0.11907
- best valid prediction Kaggle score: 0.1131

## 11

继续检查代码。

- 训练集增强后没有打乱？

	Dateset 可设置每个 epoch 重新 shuffle 。

- 循环编码错误导致未增强？

	无问题。

- 梯度裁剪？

	估计不是。

- 网络结构引用代码问题？

	应该不是。

修改：

- 减小学习率。（保持与示例代码一致）

- 验证不增强。（主要加快速度）

```python
    laerning_rate = 1e-4
```

- loss
	- best train loss = 0.1360
	- best valid loss = 1.7524
	- best acc = 0.4319
- prediction
	- best acc prediction：0.11352

## 12

- 实验更小的学习率

```python
    laerning_rate = 1e-5
```

- loss
	- best train loss = 0.2813
	- best valid loss = 1.6896
	- best acc = 0.4373
- prediction
	- best acc prediction：0.12035
	- best valid prediction：0.1259

## 13

问题：

- 代码错了，没能正确预测？？？

- 模型加载没有覆盖？

  应该不是，verison 1 可以有效果。

- 正则化？

	weight decay 已设置正则化。

策略：

- 加强正则化。

```python
	laerning_rate = 1e-4
    weight_decay = 1e-2
```

- best train loss = 0.4437
- best valid loss = 1.7311
- best acc loss = 0.4392

## 14

策略：

- 在 version 13 基础上去掉梯度裁剪。

裁剪影响不大。

- best train loss = 0.4066
- best valid loss = 1.7350
- best acc = 0.4449
- best acc prediction: 0.11694

加强正则后 valid loss 上升趋势受到了抑制。

## 15

问题：

- 增强策略有什么问题？

	查看转换函数。

策略：

- 对训练集使用 version 1 的增强策略。

结果：

- best train loss = 0.7659
- best valid loss = 1.7361
- best acc = 0.4196
- best acc prediction 0.12462

结果问题：

- 预测部分的代码的问题？

	改为测试时不增强，使用 best acc model 得到 best acc prediction 0.42851，确定为 TTA 函数出错。

## 16

策略：

- 在 version 14 基础上加强正则。

```
    weight_decay = 1
```

结果：

正则过强，loss 坏掉了。

## 17

问题：

- TTA 函数错误，`model.eval()` 的作用域问题？

	由 version 18 可知，不是这个问题，该函数作用时间可以持续到调用 `.train()` 时。

策略：

- 修改正则化强度。
- 修正 TTA 函数。
- 采用每次随机变换，需提升 epoch 。

```
    weight_decay = 0.1
    epoch = 600
```

结果：

loss 整体下降，但震动严重。

loss：

- best train loss = 1.8594
- best valid loss = 1.8927
- best acc = 0.3534

kaggle socre：

- best acc prediction：0.13529

问题：

- 为什么分数又坏掉了？

	- 下载模型本地预测，提交结果正常。

	- `eval()` 位置不够精细，在 windows 和 Linux 上精细度不同？

		应该不会是这个问题。

## 18

策略：

- 减弱过强的正则化，以减少震荡。
- 提升 epoch ，加强下降。
- 出结果后再测试一次 version 17 的情况。

```python
    weight_decay = 0.05
    epoch = 800
```

结果：

loss：

- best train loss = 1.1716
- best valid loss = 1.7288
- best acc = 0.4197

Kaggle score：

- best acc prediction Kaggle 直接提交：0.12121
- best acc prediction 下载模型本地预测再提交：0.42125

问题：

- loss 在下降中震荡，不稳定。

## 19

策略：

- 加大 batch size ，提升稳定度。
- 应用梯度裁剪，避免梯度爆炸。
- 提升 epoch 。
- 按 version 17 的正则。

```python
epoch = 1000
batch_size = 128
weight_decay = 0.1
```

loss：

- best train loss = 1.8231
- best valid loss = 1.8546
- best acc = 0.3588

Kaggle score：

- best acc prediction 本地预测：0.34229

问题：

- loss 降不下去。

## 20

策略：

- 调大初始学习率。

```pyton
epoch = 200
batch_size = 128
learning_rate = 0.01
```

loss：

- 直接坏掉。

## 21

策略：

- 减小学习率调大的程度。

```python
epoch = 150
learning_rate = 1e-3
```

loss：

- best train loss = 2.1257
- best valid loss = 2.0707
- best acc = 0.2731

## 22

策略：

- 减小初始学习率

```python
learning_rate = 1e-5
```

loss：

- best train loss = 1.9010
- best valid loss = 1.8973
- best acc = 0.3764

问题：

- 增加 epoch ？
- 过早趋于平稳，如何让 loss 保持下降？

## 23

策略：

- 尝试再减小初始学习率。

```python
learning_rate = 1e-6
```

loss：

- best train loss = 2.1104
- best valid loss = 2.0644
- best acc = 0.3077

分析：

- loss 降得更慢，但有一定下降趋势。

## 24

策略：

- 提升训练时间。

```python
n_epoch = 500
```

loss：

- best train loss = 1.9622
- best valid loss = 1.9363
- best acc = 0.3586

时间：

- 4 小时。是否要继续提升 epoch ？

	卷积很强，不需要太多 epoch 。

问题：

- loss 到极限？

	到达了设置的三层 FC 的极限。

## 25

问题：

- 检查 ModuleList？？？
- 不同平台卷积初始参数有一定差异，所以 Kaggle 直接提交准确率直接坏掉？？？

策略：

- 使用 ModuleList 。

```
    n_epoch = 600
    batch_size = 128
    laerning_rate = 0.01
    weight_decay = 0.1
```

loss:

- best train loss = 2.3066
- best valid loss = 2.2966
- best acc = 0.1830

问题：

- 为什么本地运行 loss 很低，Kaggle 上却很高？？？

## 26

想法：

- 本地跑一遍所有数据，本地预测。

```python
    n_epoch = 10
    batch_size = 128
    laerning_rate = 1e-4
    weight_decay = 1e-2
```

loss:

- best train loss = 0.7623
- best valid loss = 0.7353
- best acc = 0.6847

Kaggle score:

- best acc prediction：0.26333

问题：

- 正则？？？

## 27

策略：

- 加强正则

```python
    n_epoch = 10
    batch_size = 128
    laerning_rate = 1e-4
    weight_decay = 1e-1
```

本地跑 loss:

- best train loss = 0.8797
- best valid loss = 0.8420
- best acc = 0.6465

本地跑 kaggle score:

- best acc prediction: 0.2501

云端跑 loss：

- best train loss = 1.7185
- best valid loss = 1.7763
- best acc = 0.3877

云端跑 kaggle score：

- best acc prediction: 0.13529

策略：

- 云端训练更长。

```
    n_epoch = 100
```

云端 loss:

- best train loss = 0.8519
- best valid loss = 1.5114
- best acc = 0.4970

云端 kaggle score：

- best acc prediction：0.10627

问题：

- 还是泛化性？？？

## 28

实验策略：

- 本地运行。
- 尝试 dropout `p=0.2`。

```python
        weight_decay = 1e-2
```

loss：

- best train loss = 1.1278
- best valid loss = 0.9662
- best acc = 0.5954

kaggle score：

- 0.25052

策略：

- 弱化结构
- 先不使用 droput

loss:

- best train loss = 1.0237
- best valid loss = 0.9659
- best acc = 0.5883

kaggle score:

- 0.25052

策略：

- 继续弱化结构

loss:

- best train loss = 1.1002
- best valid loss = 1.0648
- best acc = 0.5541

kaggle score:

- 0.23303

## 28

策略：

- 回复上一个结构，应用 droput `p=0.5` 。
- 延长 epoch 观察。
- kaggle 跑，kaggle 测试同时本地测试。

loss：

- best train loss = 2.0959
- best valid loss = 2.1109
- best acc = 0.2751

问题：

- loss 一直缓慢降，更多 epoch ？？？
- 网络结构复杂一点再用 droput ？？？

## 29

策略：

- 加强网络结构
- droput = `0.5`

loss：

- best train loss = 1.8498
- best valid loss = 1.8051
- best acc = 0.3584

云端预测 kaggle score：

- best acc prediction：0.134

问题：

- loss 降得慢。

## 30

策略：

- 减小 batch_size
- 去掉梯度裁剪。
- 恢复示例结构。
- 去掉卷积层的 droput 。

```python
    n_epoch = 200
    batch_size = 64
    laerning_rate = 1e-4
    weight_decay = 1e-2
```

loss：

- best train loss = 0.2454
- best valid loss = 1.3986
- best acc = 0.5865

云端预测 kaggle score：

- best acc prediction：0.10285

本地预测 kaggle score：

- best acc prediction：0.58642（终于过 baseline 了！！！！）

问题：

- valid loss 降不下去，进一步加强正则？？？
- 为什么本地预测行，kaggle 云端预测结果直接坏掉？？？





不搞了，目前过 baseline 就够了。



