import math

# 余弦相似度用向量空间中两个向量夹角的的余弦值衡量两个样本的差异的大小。其计算方法如下：

# 现在有（1,2,3）和（2,3,1）两个向量，请计算它们的余弦相似度（保留两位小数）。

def cos_dist(a,b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

if __name__ == "__main__":
    d = (1,2,3)
    q = (2,3,1)
    print(cos_dist(d,q))
