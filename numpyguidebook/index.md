# NumpyGuidebook






本教程希望为入门数据科学、DeepLearning的同学提供Numpy的基本操作指南。

<!--more-->
## Numpy 入门指南
### array基本属性

Numpy的主要对象是同构多维数组。它是一个元素表，所有类型都相同，由非负整数元组构成索引。

Numpy的数组类被调用为ndarray。存在以下属性：

- ndarray.ndim：数组的轴（维度）的个数。
- ndarray.shape：数组的维度。一个整数元组，表示每个维度中数组的大小。对于有n行和m列的矩阵，shape将是(n,m)，即shape元组长度就是rank或者维度的个数ndim。
- ndarray.size：数组元素的总数。
- ndarray.dtype： 一个描述数组中元素类型的对象 。
- ndarray.itemsize：数组中每个元素的字节大小。例如，元素为 `float64` 类型的数组的 `itemsize` 为8（=64/8），而 `complex32` 类型的数组的 `itemsize` 为4（=32/8）。它等于 `ndarray.dtype.itemsize` 。

```
import numpy as np 

#如何将列表转化为矩阵
array=np.array([[1,2,3],
                [2,3,4]])
print(array)
#查看维度ndim
print('number of dim: ',array.ndim)
##output:  number of dim:  2
#查看几行几列
print('shape: ',array.shape)
##output:  shape:  (2, 3)
#查看元素个数 
print('size: ',array.size)
##output:  size:  6
```

### 创建数组

#### np.array

使用array函数从python元组中创建数组, 默认情况下，创建的数组的dtype是 `float64` 类型的。

```
import numpy as np
#创建一维数组，ndim=1
a=np.array([2,23,4],dtype=np.int32)
print(a)
##output:[ 2 23  4]

#创建二维数组
b = np.array([(1.5,2,3), (4,5,6)])
print(b)
##output: [[ 1.5  2.   3. ]
##         [ 4.   5.   6. ]]
```

注意：常见错误是，调用array时候传入多个数字参数，而不提供单个数字的列表类型作为参数。

```
>>> a = np.array(1,2,3,4)    # WRONG
>>> a = np.array([1,2,3,4])  # RIGHT
```

#### np.zeros

创建一个全为0的数组 .

```
>>> np.zeros( (3,4) )
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]])
```

#### np.ones

创建一个全为1的数组 .

```
>>> np.ones((2,3,4), dtype=np.int16) # dtype can also be specified
array([[[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]],
       [[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]]], dtype=int16)
```

#### np.empty

创建一个数组，其初始内容是随机的，取决于内存的状态。

```
>>> np.empty( (2,3) )    # uninitialized, output may vary
array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
       [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
```

#### np.arange

该函数返回指定范围内数组而不是列表 。（注意是左包含即[start,stop) ）

numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)

主要参数：start--开始；step--结束；step:步长

```
>>> np.arange( 10, 30, 5 )
array([10, 15, 20, 25])
>>> np.arange( 0, 2, 0.3 ) # it accepts float arguments
array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])
```

#### np.linspace

当`arange`与浮点参数一起使用时，由于有限的浮点精度，通常不可能预测所获得的元素的数量。出于这个原因，通常最好使用`linspace`函数来接收我们想要的元素数量的函数，而不是步长（step）

```
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,axis=0):
>>> from numpy import pi
>>> np.linspace( 0, 2, 9 )# 9 numbers from 0 to 2
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
>>> x = np.linspace( 0, 2*pi, 100 )# useful to evaluate function at lots of points
>>> f = np.sin(x)
```

### 数组基本运算

#### 加减运算

```
import numpy as np 
#加减运算
a=np.array([10,20,30,40])
b=np.arange(4)
print(a,b)
##[10 20 30 40] [0 1 2 3]
c=a+b
d=a-b
print(c,d)
##[10 21 32 43] [10 19 28 37]
```

#### 点乘、叉乘

```
import numpy as np 

a=np.array([10,20,30,40])
b=np.arange(4)
#叉乘
c=a*b
print("\n叉乘运算:",c)
##output:叉乘运算: [  0  20  60 120]

#点乘
aa=np.array([[1,1],[0,1]])
bb=np.arange(4).reshape((2,2))
c_dot=np.dot(aa,bb)
c_dot_2=aa.dot(bb)

print("\n点乘运算之一:",c_dot)
##点乘运算之一: [[2 4]
##             [2 3]]
print("\n点乘运算之二:",c_dot_2)
##点乘运算之二: [[2 4]
##             [2 3]]
```

#### 乘方

使用a**b表示a的b次方

```
import numpy as np 
b=np.arange(4)
#乘方运算
f=b**2
print("\n乘方运算:",f)
#output:[0 1 4 9]
```

#### 逻辑运算

快速查找数组中符合条件的值，涉及到>、<、==、>=、 <= 、!=，返回一个全为布尔值的数组

```
import numpy as np 
b=np.arange(4)
##output：[0 1 2 3]
#快速查找符合要求的值,逻辑判断
print(b==3,'\n')
#output :[False False False  True]
print(b!=3,'\n')
#output：[ True  True  True False]
```

#### 转秩

```
import numpy as np
B=np.arange(14,2, -1).reshape((3,4))
# B :array([[14, 13, 12, 11],
#           [10,  9,  8,  7],
#           [ 6,  5,  4,  3]])
print(np.transpose(B))
#[[14 10  6]
# [13  9  5]
# [12  8  4]
# [11  7  3]]
print(B.T)
#[[14 10  6]
# [13  9  5]
# [12  8  4]
# [11  7  3]]
```

#### np.sort

对矩阵中的所有值从大到小排序。

```
#排序函数，sort(),针对每一行进行从小到大排序操作
B=np.arange(14,2, -1).reshape((3,4))
# B :array([[14, 13, 12, 11],
#           [10,  9,  8,  7],
#           [ 6,  5,  4,  3]])
print(np.sort(B))

# B':array([[11,12,13,14],
#           [ 7, 8, 9,10],
#           [ 3, 4, 5, 6]])
```

#### np.clip

clip函数，clip(Array,Array_min,Array_max)，Array指的是将要被执行用的矩阵，而后面的最小值最大值则用于让函数判断矩阵中元素是否有比最小值小的或者比最大值大的元素，并将这些指定的元素转换为最小值或者最大值。

```
import numpy as np 

A=np.arange(2,14).reshape((3,4))

print(np.clip(A,5,9))
```

#### np.argmin

查找矩阵中的最小值的索引值

#### np.argmax

查找矩阵中的最大值的索引值

```
import numpy as np 

A=np.arange(2,14).reshape((3,4))
#[[ 2  3  4  5]
# [ 6  7  8  9]
# [10 11 12 13]]
#numpy基本运算
print(A)
#求矩阵中最小元素
print('最小值的索引值',np.argmin(A))
##最小值的索引值 0
#求矩阵中最大元素
print('最大值的索引值',np.argmax(A))
#最大值的索引值 11
```

#### np.mean

求矩阵所有值的均值,亦写成A.mean()

同np.average( )

#### np.average

```
import numpy as np 

A=np.arange(2,14).reshape((3,4))

#求矩阵的均值
print('矩阵平均值表示之一',np.mean(A),'|',A.mean())
#矩阵平均值表示之一 7.5 | 7.5

print('矩阵平均值表示之二',np.average(A))
#矩阵平均值表示之二 7.5
```

#### np.cumsum

```
import numpy as np 

A=np.arange(2,14).reshape((3,4))
#求矩阵n项累加
#eg: array([ [ 2, 3, 4, 5]
#            [ 6, 7, 8, 9]        
#            [10,11,12,13] ])
#     --->[2 5 9 14 20 27 35 44 54 65 77 90]
print('矩阵前n项累加',np.cumsum(A)) 
```

#### np.diff

```
import numpy as np 

A=np.arange(2,14).reshape((3,4))
#累差运算函数diff,计算的便是每一行中后一项与前一项之差.
#eg: array([ [ 2, 3, 4, 5],             array([[1,1,1],
#            [ 6, 7, 8, 9],       --->          [1,1,1],
#            [10,11,12,13] ])                   [1,1,1]]) 
print(np.diff(A))
```

#### np.exp

求e的幂次方。

```
>>> b=np.array([2,4,6])
>>> np.exp(b)
array([  7.3890561 ,  54.59815003, 403.42879349])
```

#### np.sqrt

开方函数

```
>>> c=np.array([4,9,16])
>>> np.sqrt(c)
array([2., 3., 4.])
```

### 索引、切片和迭代

#### 一维数组

一维的数组可以进行索引、切片和迭代操作。

```
>>> import numpy as np
>>> a=np.arange(10)**3
>>> a
array([ 0, 1, 8, 27, 64, 125, 216, 343, 512, 729], dtype=int32)
>>> a[2]  #获取第二个值
8
>>> a[2:5] #获取第二到第五个值，以数组形式返回
array([ 8, 27, 64], dtype=int32)
>>> a[:6:2]=-1000  #修改第零个、第二个、第六个值为-1000
>>> a
array([ -1000, 1, -1000, 27, -1000, 125, 216, 343, 512,
 729], dtype=int32)
>>> a[ : :-1] #倒序a
array([ 729, 512, 343, 216, 125, -1000, 27, -1000, 1, -1000], dtype=int32)
>>> for i in a:
...     print(i**(1/3.))
...
nan
1.0
nan
3.0
nan
5.0
5.999999999999999
6.999999999999999
7.999999999999999
8.999999999999998
```

#### 多维数组

多维数组的每一个轴都有一个索引，这些索引以逗号的形式分隔的元组给出：

```
>>> def f(x,y):
...     return 5*x+y
...
>>> b=np.fromfunction(f,(5,4),dtype=int)
>>> b
array([[ 0,  1,  2,  3],
       [ 5,  6,  7,  8],
       [10, 11, 12, 13],
       [15, 16, 17, 18],
       [20, 21, 22, 23]])
>>> b[2,3]  #第二行第三列的数字
13
>>> b[0:5,1] #第0~5行第1列的数字，以数组形式返回
array([ 1,  6, 11, 16, 21])
>>> b[ : ,1] #第1列的数字，以数组形式返回
array([ 1,  6, 11, 16, 21])
>>> b[1:3,:] #第1~3行的数字，以数组形式返回
array([[ 5,  6,  7,  8],
       [10, 11, 12, 13]])
```

对多维数组进行迭代（iterating）是相对于第一个轴完成的。

```
>>> for row in b:
...     print(row)
...
[0 1 2 3]
[5 6 7 8]
[10 11 12 13]
[15 16 17 18]
[20 21 22 23]
```

#### 迭代操作

如果想要对数组中的每个元素执行操作，可以使用`flat`属性，该属性是数组的所有元素的[迭代器](https://docs.python.org/tutorial/classes.html#iterators) :

```
>>> for element in b.flat:
...     print(element)
...
0
1
2
3
5
6
7
8
10
11
12
13
15
16
17
18
20
21
22
23
```

### array形状操作

#### 改变数组的形状

##### array.ravel()

化成1*n的矩阵。

```
>>> a=np.floor(10*np.random.random((3,4)))
>>> a
array([[9., 8., 7., 4.],
       [5., 3., 5., 9.],
       [9., 4., 0., 0.]])
>>> a.shape
(3, 4)
>>> a.ravel()
array([9., 8., 7., 4., 5., 3., 5., 9., 9., 4., 0., 0.])
>>> a.reshape(-1)
array([9., 8., 7., 4., 5., 3., 5., 9., 9., 4., 0., 0.])
```

Ps: **array.ravel()作用等同于array.reshape(-1)**

##### array.T

转置矩阵 。

```
>>> a.T
array([[9., 5., 9.],
       [8., 3., 4.],
       [7., 5., 0.],
       [4., 9., 0.]])
>>> a.T.shape
(4, 3)
```

##### array.reshape()

改变为任意形状 。

```
>>> a = np.arange(6).reshape((3, 2))#将1*6矩阵转为3*2矩阵
>>> a
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> np.reshape(a, (2, 3)) #将3*2矩阵转为2*3矩阵
array([[0, 1, 2],
       [3, 4, 5]])
>>> a.reshape(2,-1) #reshape操作中将size指定为-1，则会自动计算其他的size大小：
array([[0, 1, 2],
       [3, 4, 5]])
```

##### array.resize( )

该方法会直接修改数组本身的shape和size。

```
>>> a=np.arange(12).reshape(3,4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> a.resize((2,6))
>>> a
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]])
```

#### 堆叠数组

##### np.vstack

属于一种上下合并的情况。

```
import numpy as np
#合并Array
A=np.array([1,1,1])
B=np.array([2,2,2])
#vstack:属于一种上下合并
print(np.vstack((A,B))) #Vertical stack
#output:  [[1 1 1]
#          [2 2 2]]
```

##### np.hstack

属于一种左右合并的情况

```
import numpy as np

A=np.array([1,1,1])
B=np.array([2,2,2])

D=np.hstack((A,B))
print(D)
#[1 1 1 2 2 2]
E=np.hstack((B,A))
print(E)
#[2 2 2 1 1 1]
```

##### np.concatenate

```
#针对多个矩阵或序列进行合并操作，借助
# np.concatenate((A,A,A,...),axis=0 或 1)
>>> a = np.array([[1, 2], [3, 4]])
>>>a
>>>array([[1, 2],
          [3, 4]])
>>> b = np.array([[5, 6]])
>>> b
array([[5, 6]])
>>> np.concatenate((a, b), axis=0)#合并列
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a, b.T), axis=1) #合并行
array([[1, 2, 5],
       [3, 4, 6]])
>>> np.concatenate((a, b), axis=None)
array([1, 2, 3, 4, 5, 6])
```

#### 分割数组

##### numpy.split

```
import numpy as np
A=np.arange(12).reshape((3,4))
print(A)

#分割函数np.split(array,number of split row/column,axis= 0 or 1)
print(np.split(A,2,axis=1))#把四列分成2块（2列一块）
# [array([ [0, 1],
#          [4, 5],
#          [8, 9]]), array([[ 2,  3],
#                           [ 6,  7],
#                           [10, 11]])]
#axis=0,表示按行分割；axis=1,表示按列分割
print(np.split(A,3,axis=0))  #把三行按行分成3块（一行一块）
#[array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[   8,  9, 10, 11]])]
```

##### np.hsplit

按列拆开数组。

```
>>> x = np.arange(16.0).reshape(4, 4)
>>> x
array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.],
       [12., 13., 14., 15.]])
>>> np.hsplit(x, 2)
[array([[ 0.,  1.],
       [ 4.,  5.],
       [ 8.,  9.],
       [12., 13.]]), 
 array([[ 2.,  3.],
        [ 6.,  7.],
        [10., 11.],
        [14., 15.]])]
```

##### np.vsplit

按行拆开数组。

```
>>> x = np.arange(16.0).reshape(4, 4)
>>> x
array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.],
       [12., 13., 14., 15.]])

>>> np.vsplit(x, 2)
[array([[0., 1., 2., 3.],
       [4., 5., 6., 7.]]), 
 array([[ 8.,  9., 10., 11.],
       [12., 13., 14., 15.]])]
```

##### np.array_split

将一个数组拆分为大小相等或近似相等的多个子数组。如果无法进行均等划分，则不会引发异常。

```
>>> x = np.arange(8.0)
>>> np.array_split(x, 3)
[array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7.])]
```

### 拷贝和深拷贝

当计算和操作数组时，有时会将数据复制到新数组中，有时则不会 。

存在以下3种情况：

#### 完全不复制

简单分配不会复制数组对象或其数据。

```
import numpy as np

a=np.arange(4)
# =的赋值方式会带有关联性
b=a
c=a
d=b
#改变a的第一个值，b、c、d的第一个值也会同时改变。
```

#### 浅拷贝

不同的数组对象可以共享相同的数据。`view`方法创建一个查看相同数据的新数组对象。

```
>>> import numpy as np
>>> a=np.arange(12).reshape(3,4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> c=a.view()
>>> c is a
False
>>> c.base is a
False
>>> c
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> c.shape = 2,6
>>> c
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]])
>>> a.shape
(3, 4)
>>> c[0,4] = 1234
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
>>> c
array([[   0,    1,    2,    3, 1234,    5],
       [   6,    7,    8,    9,   10,   11]])
```

#### 深拷贝copy()

该`copy`方法生成数组及其数据的完整副本。

```
import numpy as np

a=np.arange(4)
#copy()的赋值方式没有关联性
b=a.copy()
print(b)
a[3]=45
print('a:',a)
#a: [11  1  2 45]
print('b:',b)
#b: [11  1  2  3]
```

**@all right save,ZhangGehang.**

