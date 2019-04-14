### 执行方式
```python
python test.py model_name

eg: python test.py NFFM
```

### 说明
DNN做ctr预估的优势在于对大规模离散特征建模，paper关注点大都放在ID类特征如何做embedding上，至于连续特征如何处理很少讨论，大概有以下3种方式：
```python
--不做embedding
  1. concat[continuous, emb_vec]
  
--做embedding
  2. 数值特征离散化之后embedding
  3. 数值特征和离散特征同等看待，直接做embedding
```
我们采取的是第三种方案：采用hash的方法，将每一个特征的特征值映射到embedding空间的某个位置上。所以hash空间不能太小，否则会有很多不同的特征值使用相同的embedding。

由于上面策略，DCN的实现方式和原版论文不同，没有`concat[continuous, emb_vec]`，而是对所有特征embedding后传入左右两个部分。

### ToDo
- 正则
    - DeepFM模型已经实现L2正则
- early_stop
- tensorboard
- 指数衰减学习率

