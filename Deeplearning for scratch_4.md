# Deeplearning from scratch

[TOC]



## CHAPTER 4 신경망 학습



### 1. 데이터에서 학습



학습이란?

- 훈련 데이터로부터 가중치 매개변수의 최적값을 자동을 획득하는 것.



이번 장에서는 신경망이 학습할 수 있도록 해주는 **지표**인 손실 함수를 소개합니다.

이 손실 함수의 결괏값을 가장 작게 만드는 가중치 매개변수를 찾는 것이 학습의 목표입니다.



#### 1.1 데이터 주도 학습



기계학습은 데이터가 생명. 데이터에서 답을 찾고 데이터에서 패턴을 발견하고 데이터로 이야기를 만드는, 그것이 바로 기계학습입니다.

![image-20201217152102460](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201217152102460.png)

> 회색블록은 사람이 개입하지 않음을 뜻한다.
>
> 딥러닝을 **종단간 기계학습**이라고도 합니다. 



#### 1.2 훈련 데이터와 시험 데이터



기계학습에서 데이터를 취급할 때 주의점

기계학습 문제는 데이터를 **훈련 데이터**와 **시험데이터**로 나눠 학습과 실험을 수행하는 것이 일반적입니다. 

> **why?**

우리가 원하는 것은 **범용적**으로 사용할 수 있는 모델이기 때문에 범용 능력을 제대로 평가하기 위해 데이터를 나누는 것입니다.

범용 능력은 아직 보지 못한 데이터로도 문제를 올바르게 풀어내는 능력입니다. 이 능력을 획득하는 것이 기계학습의 최종 목표입니다.

한 데이터셋에만 지나치게 최적화된 상태를 **오버피팅**이라고 합니다.



### 2. 손실 함수



신경망 학습에서는 현재의 상태를 '하나의 지표'로 표현합니다. 그리고 그 지표를 가장 좋게 만들어주는 가중치 매개변수의 값을 탐색하는 것 입니다.



신경망 학습에서 사용하는 지표는 **손실함수**라고 합니다. 이 손실 함수는 임의의 함수를 사용할 수도 있지만 일반적으로는 오차제곱합과 교차 엔트로피 오차를 사용합니다.



#### 2.1 오차 제곱합



가장 많이 쓰이는 손실 함수는 **오차제곱합**입니다.



![image-20201217153619794](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201217153619794.png)

- yk는 신경망의 출력(신경망이 추정한 값), tk는 정답 레이블, k는 데이터의 차원 수를 나타냅니다.

```python
y = [0.1, 0.05,0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] #첫 번째 인덱스부터 순서대로 0,1,2 일 때의 값 

t = [0,0,1,0,0,0,0,0,0,0]
```

 여기서 신경망의 출력 y는 소프트맥스 함수의 출력입니다. 이 예에서는 0일 확률은 0.1, 1일 확률은 0.05이고 정답 레이블인 t에서 두번째가 1이고 나머지 0이므로 답은 2이며 한 원소만 1로하고 나머지를 0으로 나타내는 표기법을 **원-핫 인코딩**이라고 합니다.



오차제곱합은 위 식과 같이 원소의 출력(추정 값)과 정답 레이블(참 값)의 차(yk - tk)를 제곱한 후, 그 총합을 구합니다. 



```python
 def sum_squared_error(y, t):
     return 0.5*np.sum((y-t)**2)
```

여기서 인수 y와 t는 넘파이 배열입니다.

이 함수를 실제로 사용해봅시다.

```python
# 예1 : '2'일 확률이 가장 높다고 추정함(0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
sum_squares_error(np.array(y), np.array(t))
>>> 0.09750000000000003

# 예2 : '7'일 확률이 가장 높다고 추정함(0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
sum_squares_error(np.array(y), np.array(t))

>>> 0.5975
```

이 실험의 결과로 첫 번재 예의 손실 함수 쪽 출력이 작으며 정답 레이블과의 오차도 작은 것을 알 수 있습니다. 즉 오차제곱합 기준으로  첫번 째 추정 결과가 (더 작으므로) 정답에 더 가까울 것으로 판단할 수 있습니다.



#### 2.2 교차 엔트로피 오차



교차 엔트로피 오차도 자주 쓰이며 아래와 같습니다.

![image-20201217155611359](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201217155611359.png)

여기서 log는 밑이 e인 자연로그(log e)입니다.yk는 신경망의 출력, tk는 정답 레이블입니다. 또 tk는 정답에 해당하는 인덱스의 원소만 1이고 나머지는 0입니다.(원-핫 인코딩). 그래서 실직적으로 정답일 때의 추정(tk가 1일 때의yk)의 자연로그를 계산하는 식이 됩니다.



<img src="C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201217160912406.png" alt="image-20201217160912406" style="zoom:50%;" />

x가 1일 때 y는 0이 되고 x가 0에 가까워질수록 y의 값은 점점 작아집니다.

정답에 해당하는 출력이 커질수록 0에 다가가다가, 그 출력이 1일 때 0이 됩니다. 반대로 정답일 때의 출력이 작아질수록 오차는 커집니다.

구현해보면 아래와 같습니다.

```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
```

여기서 y와 t는 넘파이 배열입니다. 마지막에 delta를 더하는데 이는 np.log() 함수에 0을 입력하면 마이너스 무한대를 뜻하는 -inf가 되어 더 이상 계산을 진행할 수가 없기 때문 입니다. 아주 작은 값을 더해서 절대 0이 되지 않도록 하는 것 입니다.

아래는 함수를 써서 계산해 본 것 입니다.

```python
t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

cross_entropy_error(np.array(y), np.array(t))
>>> 0.510825457099338

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

cross_entropy_error(np.array(y), np.array(t))
>>> 2.302584092994546
```

 위 결과도 오차 제곱합의 판단과 일치하는 모습입니다.



#### 2.3 미니배치 학습



지금까지 데이터 하나에 대한 손실 함수만 생각해 왔으니, 이제 훈련데이터 모두에 대한 손실 함수의 합을 구하는 방법을 생각해보겠습니다.

![image-20201217162424083](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201217162424083.png)

수식이 복잡해 보이지만 데이터 하나에 대한 손실 함수를 단순히 n개의 데이터로 확장했을 뿐 입니다.

다만 마지막에 n으로 나누어 정규화하고 있습니다.

n으로 나눕으로써 '평균 손실 함수'를 구하는 것 입니다. 

> why?

데이터 개수와 관계없이 언제든 통일된 지표를 얻을 수 있습니다.



데이터가 너무 많은 경우 일부를 추려 전체의 '근사치'로 이용할 수 있습니다. 신경망 학습에서도 훈련 데이터로부터 **일부**만 골라 학습을 수행합니다. 이 일부를 **미니배치**라고 합니다. 6만장의 훈련 데이터 중에서 100장을 무작위로 뽑아 그 100장을 사용하여 학습하는 것입니다. 이러한 학습 방법을 미니배치 학습 이라고 합니다.

이를 코드로 만들어봅시다.

```python
import sys, os

sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = 
	load_mnist(normalize = True, one_hot_label = True)
  
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

```

이 함수는 훈련 데이터와 시험 데이터를 읽습니다. 호출할 때 one_hot_label=True로 지정하여 원-핫 인코딩으로, 즉 정답 위치의 원소만 1이고 나머지가 0인 배열을 얻을 수 있습니다.

앞의 코드에서 데이터를 읽은 결과, 훈련 데이터는 6만개, 입력데이터는 784열 인 이미지 데이터임을 알 수 있습니다. 또, 정답 레이블은 10줄짜리 데이터 입니다.

이 훈련 데이터에서 무작위로 10장만 빼내려면 np.random.choice() 함수를 쓰면 됩니다.

```python
train_size = x_train.shape[0] # 60000
batch-size = 10
batch_mask = np.random.choice(train_size, batch_size) # randomly select indecies from 0-59999
x_batch = x_train[batch_mask]
t_train = t_train[batch_mask]
```

np.random.choice()로는 지정한 범위의 수 중에서 무작위로 원하는 개수만 꺼낼 수 있습니다.

```python
np.random.choice(60000, 10)
>>> array([32010, 58308,  8622,  2726, 52569, 51412, 44120, 47100, 10286,
       48558])
```

이제 무작위로 선택한 이 인덱스를 사용해 미니배치를 뽑아내기만 하면 됩니다. 손실 함수도 이 미니배치로 계산합니다.

> 텔레비전 시청률도 모든 세대의 텔레비전이 아니라 선택된 일부 가구의 텔레비전만을 대상으로 구합니다.



#### 2.4 (배치용) 교차 엔트로피 오차 구하기



미니배치 같은 배치 데이터를 지원하는 교차 엔트로피 오차는 조금만 바꿔주면 됩니다.



```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```



이 코드에서 y는 신경망의 출력, t는 정답 레이블 입니다. y가 1차원이라면, 즉 데이터 하나당 교차 엔트로피 오차를 구하는 경우는 reshape 함수로 데이터의 형상을 바꿔줍니다. 그리고 배치의 크기로 나눠 정규화 하고 이미지 1장당 평균 교차 엔트로피 오차를 계산합니다.

정답 레이블이 원-핫 인코딩이 아니라 2,7 등의 숫자 레이블로 주어졌을 때의 교차 엔트로피 오차는 다음과 같습니다.



```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```



이 구현에서는 원-핫 인코딩일 때 t가 0인 원소는 교차 엔트로피 오차도 0이므로, 그 계산은 무시해도 좋다는 것이 핵심입니다.

다시말하면 정답에 해당하는 신경망의 출력만으로 교차 엔트로피 오차를 계산할 수 있습니다.

np.log(y[np.arange(batch_size) 에 대해서.

np.arange(batch_size)는 0부터 batch_size - 1까지 배열을 생성합니다. 즉 batch_size = 5이면 [0,1,2,3,4] 라는 넘파이 배열을 생성합니다. t에는 레이블이 [2,7,0,9,4] 와 같이 저장되어 있으므로 각 데이터의 정답 레이블에 해당하는 신경망의 출력을 추출합니다. 이 예에서는 [y[0,2],y[1,7], ... ] 인 넘파이 배열을 생성합니다.



#### 2.5 왜 손실 함수를 설정하는가?

숫자 인식의 경우도 우리의 궁극적인 목적은 높은 정확도를 끌어내는 매개변수 값을 찾는 것인데 왜 정확도라는 지표를 놔두고 손실 함수의 값이라는 우회적인 방법을 택할까요?

신경망 학습에서는 최적의 매개변수(가중치와 편향)을 탐색할 때 손실 함수의 값을 가능한 한 작게 하는 매개변수를 찾습니다. 이때 매개변수의 미분(정확히는 기울기)을 계산하고, 그 미분값을 단서로 매개변수의 값을 서서히 갱신하는 과정을 반복합니다. 



> 신경망을 학습할 때 정확도를 지표로 삼아서는 안 된다. 정확도를 지표로 하면 매개변수의 미분이 대부분의 장소에서 0이 되기 때문이다



- 손실 함수의 미분이란
  - 가중치 매개변수의 값을 아주 조금 변화시켰을 때 손실 함수가 어떻게 변하는가
  - 음수라면 가중치 매개변수를 양의 방향으로 변화시켜 값을 줄일 수 있다.
  - 반대로 양수라면 매개변수 방향을 음의 방향으로 변화한다.
  - 미분 값이 0이라면 어느쪽으로도 움직이지 않는다.
- 정확도를 사용하기 어려운 이유는 이 미분값이 0으로 되어 양의 방향이나 음의 방향 모두 효과가 나타나지 않기 때문
- 예를 들어, 정확도가 32%인 모델이 있는 경우,
  - 조금 매개변수를 조정해도 32%가 나올 확률이 높다. 이 경우에는 미분을 하여도 증분이 0이기 때문에 학습을 수행하기 매우 어렵다.
- 이와는 다르게 손실함수의 값을 삼는 경우
  - 값이 0.9253… -> 0.9343… 식으로 변하기 때문에 값을 구하기 쉽다.



### 3. 수치 미분

경사법 = 기울기(경사) 값을 기준으로 나아갈 방향을 정합니다.



#### 3.1 미분

![image-20201217174720250](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201217174720250.png)



다음은 함수의 미분을 나타낸 식입니다. 좌변은 f(x)의 x에 대한 미분(x에 대한 f(x)의 변화량)을 나타내는 기호 입니다. 결국 x의'작은변화'가 함수 f(x)를 얼마나 변화시키느냐를 의미합니다. 이 때 시간의 작은 변화, 즉 시간을 뜻하는 h를 한없이 0에 가깝게 한다는 의미를 lim 로 나타냅니다.

```python
# 나쁜 구현의 예
def numerical_diff(f, x):
    h = 10e-50
    return(f(x + h) - f(x)) / h

```

이 구현에서는 h에 가급적 작은 값을 대입하고 싶었기에 10e-50이라는 작은 값을 이용했는데 이 값은 0.00...1 형태에서 소수점 아래 0이 49개라는 의미죠. 그러나 이 방식은 반올림 오차 문제를 일으키게 됩니다. 

반올림오차는  작은 값이 생략되어 최종 계산 결과에 오차가 생기게 합니다. 아래와 같은 문제죠

```python
np.float32(1e-50)
>>> 0.0
```

이와 같이 올바로 표현할 수 없습니다. 이 미세한 값을 10-4 정도의 값을 사용하면 좋은 결과를 얻습니다.



- 두 번째 개선은 함수 f의 차분과 관련한 것입니다.
  - 앞의 구현에서는 x + h와 x 사이의 함수 f의 차분을 계산 하고 있지만, 애당초 이 계산에는 오차가 있다는 사실에 주의해야 합니다.
  -  그림과 같이 진정한 미분은 x 위치의 함수의 기울기(접선)에 해당하지만, 이번 구현에서의 미분은 (x +h) 와 x 사이의 기울기에 해당합니다.
  -  그래서 진정한 미분과 이번 구현의 값은 엄밀히는 일치하지 않습니다. 이 차이는 h를 무한히 0으로 좁히는 것이 불가능해 생기는 한계입니다.



<img src="C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201217175527161.png" alt="image-20201217175527161" style="zoom: 67%;" />

수치 미분에는 오차가 포함됩니다. 이 오차를 줄이기 위해서 (x + h)와 (x - h)일 때의 함수 f의 차분을 계산하는 방법을 쓰기도 합니다.

[^차분]: 임의 두 점에서의 함수 값들의 차이

이 차분은 x를 중심으로 그 전후의 차분을 계산한다는 의미에서 **중심 차분** 혹은 **중앙 차분**이라 합니다. (한편, (x + h)와 x의 차분은 **전방차분**이라고 합니다.)

그럼 이상의 두 개선점을 적용해 수치 미분을 다시 구현해봅시다.

```python
def numerical_dif(f,x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
```



> 여기서 하는 것처럼 아주 작은 차분으로 미분하는 것을 **수치 미분**이라고 합니다. 한편, 수식을 전개해 미분하는 것은 **해석적**이라는 말을 이용하여 표현합니다.
>
> 쉽게 말해서 해석적 미분은 우리가 수학 시간에 배운 바로 그 미분이고, 수치미분은 이를 근사치로 계산하는 방법.



#### 3.2 수치 미분의 예

간단한 2차 함수를 파이썬으로 구현하고 그려보면 아래와 같습니다.

```python
def function_1(x):
    return 0.01*x**2 + 0.1*x
```

```python
import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1) # 0에서 20까지 0.1 간격의 배열 x를 만든다
y = function_1(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y)
plt.show()
```

![img](data:image/png;base64,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)

그럼 x = 5일 때와 10일 때 이 함수의 미분을 계산해봅시다.

```python
numerical_diff(function_1,5)
>>> 0.1999999999990898
numerical_diff(function_1,10)
>>> 0.2999999999986347
```



이렇게 계산한 미분 값이 x에 대한 f(x)의 변화량입니다. 즉 함수의 기울기에 해당합니다.

이제 앞에서구한 수치 미분 값을 기울기로 하는 직선을 그리면 아래와 같습니다.(각각 x=5, 10)



![img](data:image/png;base64,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)

![img](data:image/png;base64,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)



#### 3.3 편미분

![image-20201217182254339](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201217182254339.png)

이어서 변수가 2개인 식을 파이썬으로 구현해봅시다.



```python
def function_2(x):
    retunr x[0]**2 + x[1]**2
```

이 함수를 그래프로 그리면 아래와 같습니다.

<img src="C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201217182702877.png" alt="image-20201217182702877" style="zoom:67%;" />



변수가 2개인 식을 미분할 때는 어느 변수에 대한 미분이냐, 즉 x0과 x1 중 어느 변수에 대한 미분이냐를 구별해야 합니다. 덧붙여 이와 같이 변수가여럿인 함수에 대한 미분을 **편미분**이라고 합니다.



##### 3.3.1 문제 1 

x0 = 3, x1 = 4일 때, x0에 대한 편미분을 구하라.

```python
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

numerical_diff(function_tmp1, 3.0)
>>> 6.00000000000378
```

##### 3.3.2 문제 2

x0=3, x1 = 4일 때, x1에 대한 편미분을 구하라.

```python
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

numerical_diff(function_tmp2, 4.0)
>>> 7.999999999999119
```



이 문제들은 변수가 하나인 함수를 정의하고, 그 함수를 미분하는 형태로 구현하여 풀었습니다.

- 이처럼 편미분은 변수가 하나인 미분과 마찬가지로 특정 장소의 기울기를 구합니다.

- 단, 여러 변수 중 목표 변수 하나에 초점을 맞추고 다른 변수는 값을 고정합니다. 

- 앞의 예에서는 목표 변수를 제외한 나머지를 특정 값에 고정하기 위해서 새로운 함수를 정의했지요
- 그 새로 정의한 함수에 대해 그동안 사용한 수치 미분 함수를 적용하여 편미분을 구한 것입니다.



### 4. 기울기



기울기란? 

모든 변수의 편미분을 벡터로 정리한 것.

```python
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x,size):
        imp_val = x[idx]
        #f(x+h) 계산
        x[idx] = imp_val + h
        fxh1 = f(x)
        
        x[idx] = imp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad
```

- numerical_gradient(f, x) 함수의 동작방식은 변수가 하나일 때의 수치 미분과 거의 같습니다.
- np.zeros_like(x)는 x와 형상이 같고 그 원소가 모두 0인 배열을 만듭니다.
- numerical_gradient(f, x) 함수의 인수인 f는 함수이고 x는 넘파이 배열이므로 넘파이 배열x의 각 원소에 대해서 수치 미분을 합니다.



여기서 (3,4),(0,2),(3.0)에서의 기울기를 구해보겠습니다.

```python
numerical_gradient(function_2, np.array([3.0, 4.0]))
>>> array([6., 8.])
numerical_gradient(function_2, np.array([0.0, 2.0]))
>>> array([0., 4.])
numerical_gradient(function_2, np.array([3.0, 0.0]))
>>> array([6., 0.])
```

이처럼 각 점에서의 기울기를 계산할 수 있습니다.



<img src="C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201217185058793.png" alt="image-20201217185058793" style="zoom:67%;" />

- 기울기 그림은 방향을 가진 벡터(화살표)로 그려집니다. 기울기는 함수의 가장 낮은 장소를 가리키는 것 같습니다. 또 가장 낮은 곳에서 멀어질 수록 화살표의 크기가 커짐을 알 수 있습니다.

- 그림에서 기울기는 가장 낮은 장소를 가리키지만, 실제로 반드시 그렇다고는 할 수 없습니다. 사실 기울기는 각 지점에서 낮아지는 방향을 가리킵니다.

> **<u>더 정확히는 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향</u>**입니다. 이건 중요한 포인트니 확실히 기억해두세요.



#### 4.1 경사법(경사 하강법)

경사법이란?

- 기울기를 잘 이용해 함수의 최솟값(또는 가능한 한 작은 값)을 찾으려는 것
  - 여기서 주의할 점은 각 지점에서 함수의 값을 낮추는 방안을 제시하는 지표가 기울기라는 것
  - 그러나 기울기가 가리키는 곳에 정말 함수의 최솟값이 있는지, 즉 그쪽이 정말 맞는 방향인지 보장할수가 없습니다.
  - 실제로 복잡한 함수에서는 기울기가 가리키는 바얗ㅇ에 최솟값이 없는 경우가 대부분 입니다.
  - 기울어진 방향이 꼭 최솟값을 가리키는 것은 아니나, 그 방향으로 가야 함수의 값을 줄일 수 있습니다.
  - 그래서 최솟값이 되는 장소를 찾는 문제에서는 기울기 정보를 단서로 나아갈 방향을 정해야 합니다.

경사법

- 현 위치에서 기울어진 방향으로 일정 거리만큼 이동합니다. 그런 다음 이동한 곳에서도 마찬가지로 기울기를 구하고, 또 그 기울어진 방향으로 나아가기를 반복합니다. 이렇게 해서 함수의 값을 점차 줄이는 것이 **경사법**입니다.
- 경사법은 기계학습을 최적화하는 데 흔히 쓰는 방법.
- 특히 신경망 학습에는 경사법을 많이 사용합니다.



![image-20201218140014117](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201218140014117.png)

저기서 n처럼 보이는 것은 eta(에타), 갱신하는 양을 나타냅니다. 이를 **학습률**이라고 합니다. 한 번의 학습으로 얼마만큼 학습해야 할지, 즉 매개변수 값을 얼마나 갱신하느냐를 정하는 것이 학습률입니다.



- 저 식은 1회에 해당하는 갱신, 이 단계를 반복합니다.

- 변수의 값을 갱신하는 단계를 여러 번 반복하면서 서서히 함수의 값을 줄이는 것 입니다.

- 변수의 수가 늘어도 같은 식으로 갱신하게 됩니다.

- 학습률 값은 미리 특정 값으로 정햐두어야 하는데 너무 크거나 작으면 '좋은 장소'를 찾아갈 수 없습니다. 신경망 학습에서는 보통 이 학습률 값을 변경하면서 올바르게 학습하고 있는지를 확인하면서 진행합니다.



경사하강법은 아래와 같습니다.

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
```



인수 f는 최적화하려는 함수, init_x는 초깃값, lr은 learning rate를 의미하는 학습률, step_num은 경사법에 따른 반복 횟수를 뜻합니다. 함수의 기울기는 numerical_gradient(f, x)로 구하고, 그 기울기에 학습률을 곱한 값으로 갱신하는 처리를 step_num번 반복합니다.



다음 문제를 풀어봅시다.

문제 : 경사법으로 f(x0,x1) = x0** + x2**의 최솝값을 구하라.

```python
def function_2(x):
    return x[0]**2 + x[1]**2

init-x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr-0.1, step_num=100)

>>>array([-6.11110793e-10,  8.14814391e-10])
```

초깃값을 (-3.0, 4.0)으로 설정한 후 경사법을 사용해 최솟값 탐색을 시작합니다.

이 코드를 그림으로 그리면 아래와 같습니다.

```python
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
```

![img](data:image/png;base64,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)

최종 결과는 거의 0,0에 가까운 결과 입니다. 

여기서 학습률이 너무 크거나 작으면 안좋다고 한 부분을 실험해보면

```python
# 학습률이 너무 큰 예 : lr = 10.0
init_x = np.array([-3.0, 4.0])

gradient_descent(function_2, init_x=init_x, lr=10, step_num=100)
>>> array([-2.58983747e+13, -1.29524862e+12])

# 학습률이 너무 작은 예 : lr = 1e-10
init_x = np.array([-3.0, 4.0])

gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
>>> array([-2.99999994,  3.99999992])
```

학습률이 너무 크면 큰 값으로 발산해버리고 너무 작으면 거의 갱신되지 않은 채 끝나버립니다.

> 학습률 같은 매개변수를 **하이퍼파라미터**라고 합니다. 이는 가중치와 편향 같은 신경망의 매개변수와는 성질이 다른 매개변수입니다. 신경망의 가중치 매개변수는 훈련 데이터와 학습 알고리즘에 의해서 '자동'으로 획득되는 반면, 학습률 같은 하이퍼파라미터는 사람이 직접 설정해야하는 매개변수입니다.



#### 4.2 신경망에서의 기울기



신경망 학습에서도 기울기를 구해야합니다.

![image-20201218145044384](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201218145044384.png)



가중치가 W, 손실 함수가 L인 신경망을 생각해보면 위와 같습니다. 아래의 각 원소는 각각의 원소에 관한 편미분입니다. 

간단한 신경망을 예로 들어 기울기를 구한느 코드를 구현해보겠습니다.

```PYTHON
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)

```

simpleNet 클래스는 형상이 2*3인 가중치 매개변수 하나를 인스턴스 변수로 갖습니다. 매서드는 2개인데, 하나는 예측을 수행하는 predict(x)이고, 다른 하나는 손실 함수의 값을 구하는 loss(x,t)입니다. 여기에서 인수 x는 입력 데이터, t는 정답 레이블 입니다.

simpleNet으로 실험을 해보죠

```python
net = simpleNet()
print(net.W) # 가중치 매개변수
>>>
[[ 0.59621865  0.42721305 -1.69790677]
 [ 1.63149782  0.45348609  1.63496106]]

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
>>> [1.82607922 0.66446531 0.45272089]

np.argmax(p) # 최댓값의 인덱스
>>> 0

t = np.array([0, 0, 1]) # 정답 레이블
net.loss(x, t)
>>> 1.8220327888714216
```

이어서 기울기를 구해보죠. 지금까지처럼 numericalgradient(f,x)를 써서 구하면 됩니다.

```python
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
>>>
[[ 0.38308388  0.11989784 -0.50298172]
 [ 0.57462581  0.17984676 -0.75447257]]
```

numerical_gradient(f,x)의 인수 f는 함수, x는 함수 f의 인수 입니다. 그래서 여기에서는 net.W를 인수로 받아 손실 함수를 계산하는 새로운 함수 f를 정의했습니다. 그리고 이 새로 정의한 함수를 numerical_gradient(f, x)에 넘깁니다.

dW numerical_gradient(f,net.W)의 결과로, 그 형상은 2*3의 2차원 배열입니다. 



이 구현에서는 새로운 함수를 정의하는 데 "def f(x)"를 썼는데, 파이썬에서는 간단한 함수라면 람다 기법을 쓰면 더 편합니다



```python
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
```

신경망의 기울기를 구한 다음에는 경사법에 따라 가중치 매개변수를 갱신하기만 하면 됩니다.



### 5. 학습 알고리즘 구현하기



신경망 학습의 절차는 다음과 같습니다.



- 전체
  - 신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 '학습'이라고 합니다. 신경망 학습은 다음과 같이 4단계로 수행합니다.

- 1단계 - 미니배치
  - 훈련 데이터 중 일부를 무작위로 가져옵니다. 이렇게 선별한 데이터를 미니배치라 하며, 그 미니배치의 손실 함수 값을 줄이는 것이 목표입니다.

- 2단계 - 기울기 산출
  - 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구합니다. 기울기는 손실 함수의 값을 가장 작게 하는 방향을 제시합니다.

- 3단계 - 매개변수 갱신
  - 가중치 매개변수를 기울기 방향으로 아주 조금 갱신합니다.

- 4단계 - 반복
  - 1 ~ 3단계를 반복합니다.



이것이 신경망 학습이 이뤄지는 순서입니다. 이는 경사 하강법으로 매개변수를 갱신하는 방법이며, 이때 데이터를 미니배치로 무작위로 선정하기 때문에 **확률적 경사 하강법**이라고 부릅니다. 대부분의 딥러닝 프레임워크는 확률적 경사 하강법의 영어 머리 글자를 딴 SGD라는 함수로 이 기능응ㄹ 구현하고 있습니다.



#### 5.1 2층 신경망 클래스 구현하기

처음에는 2층 신경망을 하나의 클래스로 구현하는 것부터 시작합니다.

```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:
                       # 순서대로 입력층,은닉층,출력층의 뉴런 수
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x): # 예측(추론)을 수행한다 인수 x는 이미지 데이터
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t): # 손실함수의 값을 구한다. 
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t): # 정확도를 구한다
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t): #가중치 매개변수의 기울기를 구함
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t): #가중치 매개변수의 기울기를 구함. 위 함수의 성능 개선판.
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

```

위와 같이 구현할 수 있습니다. TwoLayerNet  클래스는 딕셔너리인 params와 grads를 인수턴스 변수로 갖습니다.



grads 변수에는 parmas 변수에 대응하는 각 매개변수의 기울기가 저장됩니다. 예를 들어 다음과 같이 numerical_gradient() 메서드를 사용해 기울기를 계싼하면 grads 변수에 기울기 정보가 저장됩니다.



```python
x = np.random.rand(100, 784) # 더미 입력 데이터 (100장 분량)
t = np.random.rand(100, 10) # 더미 정답 데이터 (100장 분량)

grads = net.numerical_gradient(x, t) # 기울기 계산

grads['W1'].shape # (784, 100)
grads['b1'].shape # (100,)
```



#### 5.3 시험 데이터로 평가하기



훈련 데이터 중 일부를 무작위로 꺼내고(미니배치), 그 미니배치에 대해서 경사법으로 매개변수를 갱신합니다. 그럼 TwoLayer 클래스와 MNIST 데이터셋을 사용하여 학습을 수행해봅시다.

> 에폭은 하나의 단위 입니다. 1에폭은 학습에서 훈련 데이터를 모두 소진했을 때의 횟수에 해당합니다. 

```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from common.mnist import load_mnist
from common.two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

```



![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYIAAAEKCAYAAAAfGVI8AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deXxU9b3/8ddntux7wiJhCYqKWhcMLlVc6lVBqkJbl1at9faKaPXaWr1i697e1sqtt7c/68JVqrVevWqtS0utS1HvvVYFLYqAGBCFYQ1JCIRkklm+vz9moCEEmGgmJ2Tez8djHjNnmTnvmcB85nzP+X6POecQEZHs5fM6gIiIeEuFQEQky6kQiIhkORUCEZEsp0IgIpLlVAhERLJcxgqBmc02sw1m9sEulpuZ/dLMlpnZ+2Y2LlNZRERk1zK5R/AQMHE3yycBY1K3acC9GcwiIiK7kLFC4Jx7HWjczSpnA79xSW8CpWY2NFN5RESkewEPtz0MWNVpOpyat7brimY2jeReAwUFBUceeOCBfRJQRGSgeOeddzY656q6W+ZlIbBu5nU73oVzbhYwC6C2ttbNnz8/k7lERAYcM/t0V8u8PGsoDAzvNF0NrPEoi4hI1vKyEDwHfDN19tAxQLNzbqdmIRERyayMNQ2Z2WPASUClmYWBW4AggHPuPmAOcAawDGgFLslUFhER2bWMFQLn3Nf3sNwB38nU9kVEJD3qWSwikuVUCEREspwKgYhIllMhEBHJcioEIiJZzsuexSIiez3nHPGEI7btFk8QjTtiiQSxuKMjnqA9mqAjnqAjlqA9Fqcjtu1x6j6eoD0a77ROotM6qfXjCf5h7GC+Mq6619+DCoGI9Budv1R3vE8k7+Op6ViUeDxKNGHECRCLxXAtG4jFosSiHUSj7cSjHWz2l7PFX0qifSslTQtJpJ4Xj0Zx8SifhPZjnW8wOZF6Dt7yBi4ehUQMi3fgS0T5n8CxfGzDGBZdydSOP+B3UfwuSpAoARfjl7Gv8F58FMf6FnFD4L8IESNIjByLEiLGtI5rWehGc7bvf/lx8NcYDkuNpGPAlI7b+cgN50L/S9wUeCS1nNRyx1m+/0djcCiXuKeZkJjHGyMez8jnrkIgki0SCUjEdryZQV5Z8gt4U5hoZCvRaJRotINoNEqHL5fW4tF0xBIE18zDRbakvmSTt82BSlYWjyMSTbD/ysfxRVsg1oGLtUO8g08C+/J63slEonH+qfHnBONt+BLJL1NfIsZrHMHDiTNwiRhP+X5AgPj2m98S/FfsS9wTn0IxLbyZcxV+4gSJ47Pkl+nM6Ln8Kj6FfdjIG7n/vNNbvi16Eb+OT2KMhXkp5192Wv7TwHf4IPc0DnWfcMnWf99pecfgkeSUjGVsW4zT1rxJ3B8ibkHiFiThC3HO6EpOLNmP6q1tlKzah4QvhPOHaPeHiPhDXL7fOKIloxi0OUBTuBW/z4ffb/jMh99n/Oe4U/GXDCV/fTFuRRF+vw+/+TBfshzM+eJkyCuDT/OgvpaDakf38j+KJEv269p7aNA52Wsl4hBrh3g7xKOpxx3ESmuSTQn1y4k3h4lF24l3RIi3txGLR1k/8kwi0TjFK/5EfsMiiLZBLAKxNtotl1dq/oX2WJwJK37BiM3v4k+0E0i0E0y0U+8fxI2Vv6A9luBHjddxSGzRDpE+YF++GvtXOuIJ/hC8gYN9O45L9kb8IL4RvRGAuaHvUeNbv8Pyl+LjuDR6LQDzci6nypoBiOEjRoBXgydwT8n3yA34+XnDdAIkiFsA5wvhfAE+LDuJeftcgN+Mr9Vdh/MFcBbA+YNgfsJVEwgPPZ2g62DcsrvBFwB/APxBzBdg06CjaBlUSzDRxrCVz+EPhAgEgviDIfyBEAw5BH/lfoQSbeSuX0AgGMT8IfD5wR+E4mGQX578W2ytB18wOd8fBH9O8t66Gx9z72Nm7zjnartdpkIgWSPWDlvWQVsjtDYQb2mgvaWBltFfpi2nAta+R+7yF0hE20nEo7hYB4l4lMX7X85mfwUVa1+lJvwMxGOQ2NaEEOXREbfRSDG1G5/lxKan/958kLr/VsmDNCfy+HbrbL4ee2anWKMjvyWBj38NPMgFgVd2WNbmQoxtfwiAu4L3MMX3f0QIESFIhBDrXTlTO24n6DeuCTzFF2wFUV+IqC+XmC+HTYEqnim+kJygjxPbXqYq0QC+AOZP3iI5lSyrOpVQwMf+zW+Q77bi8wfw+YP4A0ES+RVsrRpHKOCjvHkRIWIEQrmEcnIIhnIJ5ZcSKhtKbtBPKLYVXyCU/PL0+fviLyo9sLtCoKYh2fs4Bx0tJFo20rq5ns2hITRZCe0bPqZk6RO41gZ8bY0E2psIdWziicHf5X07kIOa/sI1m36y/WX8QD5wYXs777r9Ocf/KjODs+hwfmIEiOInSoCZS45kuRvGVN8irggsSf3a9RO3AAkL8H8fraclFKMkEWAE+xD3J5sNEhYk4Q8xtLSAIcECGttPZE77YNy2X5qBHCyQw/cGjyEYDJIT+Q4vxc7HF8jBgjn4Q/kEcvL4bWkNOUEfOb6jqQuGyA35yQn4yQv4OCDoY5nfR8DvIzl0186+sf3R0Xv4YA/Yw/LBu18cLN7D86W/0h6BZEYikfzl3bIecoqgdESySeOdhyEWwUXbiEcjdERa2TTsJNZXfZFI01pq3rwRF43gYhGIteOLt/NS2fm8lnMSJVvquH3jNeTQToD49k1d0zGdpxMncKQt5cnQ7TRRyCZXSCNFbPEV82joXNYUHMS+wQbGJxYSzyvH8svxFVQQLKzAl19GMBgiJ2DkBPzkBP3kBHx/v+/6OOAn6DdsgDQZSHbQHoH0no7W5Jd76pbYsp7Wgmrqh5xIU8tW9nt2KoG2enLbG/C5GAAvlZ3HwwXfJt7axGON1wPJMyYSzk+cEA+9tYVZcahiE78J1dFOkPZU00fcV8L7sTirC9uwYDFvFE/EgnkkcktxueVYQTknVx3KaWXVFOfUsjj3W5QU5FKZF2RUTgC/z/jSDm9gSl9/YiL9ngpBtnMueeAxmAdA/KOX2LLhE1ob19GxeQPWsp71OSN5seoSmlqj3PbhZAoTW7Y/3Qe8Ev8iV0eTe5b3BvPY7MZSTwkbXSmtORWs79iX1mCM4sIyZpQ/R25ePnl5+RTm5VCcF+Tg3ACzcwMU5wbx5Z7FoNwAxXlBCkJ+zLp+kU/us49GJFuoEAxEHVuTZ0BsbUjeuzgcOJloPEHbiz/GrZoHW+vxRxrJ7WhkZc4Yvl/8b6zdFOHXkWsY61tJKbDF5bHBlbI44Xj045WU5Qd5MPdicnJyieVXQeEg/EVDyCkZxF0FuZTlhyjNf5Kx+SHK8kMU5Qbw+dR8ItLfqRDs7Vo2wIbFMPoktrbHiD5yDqXhv+ywSr2visn+HOpb2vmZfz77+8I0uBIa3QE0+0tpYiQFoQATxlTy19BdLC0poaxqKFVlJZQXhDg/P8i3gtvOAjmlz9+iiGSWCsHepmE5LHsFF36b+KdvEdi8kgQ+zin7b/62LsrJdgRjbBANFNMaLMNXUEWgZAgnlw9iSEkurvRumkvyGF6Sy/iSXIpyAjroKZLlVAj6sy3rIfw2hOfRUTudRVvyiPzPYxxbN5N6ypkf3493ExNY7D+A/PwCrjy5ksNH1DKiPJ8hJXkU5ujPKyJ7pm+K/mbjMnj1p8RXvo1/80oAYgS47PUC5kYPoYx9GV0yi+pRYzhyVDlTR5YxY3BR6jxyEZGeUyHw2oYluD9ey7qaKczNO51P6j7i0uV/YV4s+Wv/fdsfhh7GoSMHc+7IMsaNLGNwca7XqUVkAFEh8FL9UjoenMyW9hg/rTuM5xILqcgPsqLmSY4cWcbpI8u4dlgJuUF11xeRzFEh8MrGOtofnMzmSIwfltzB6SeewDUjyxhZka+DtyLSp1QIvBBppv3BybS0tXNjyR3cOf1rlOaHvE4lIllKhcAD/xeO8lLLZNaWjuOOy85RERART6kQ9KWmT3l/6Ud8+48xRpZP5b8uPZqyAhUBEfGWCkFf2bSKyINnULklQk3p/Txy6dFUFOZ4nUpEBJ183heaVxN58Aw6tjTyo4IZ/GbaBCpVBESkn9AeQaZtXkvkgTOIbq5nRv7t3Db9IqqKVAREpP/QHkGGrX/x58Q3r2NG/q3ccvk3GaTOYCLSz2iPIIPeW7WJixd+icPyv8DPpp+nHsEi0i+pEGTC1o00P3UV310xhaKCKn467VyGlKgIiEj/pKah3ra1gciDk8lZ8RJjQ/U8dukx7FOa53UqEZFdUiHoTa2NtM0+ExqXc33wh9ww/R+pLsv3OpWIyG6paai3tDURmX0W/oal/EvwBr4//TKGl6sIiEj/p0LQS5avayK6cQv3B67nu5ddzogKFQER2TtktGnIzCaa2VIzW2ZmM7pZXmJmz5vZe2a2yMwuyWSejGhvYdnaRs57dDnfCtzJVZddwajKAq9TiYikLWN7BGbmB34FnAqEgXlm9pxzbnGn1b4DLHbOnWlmVcBSM3vUOdeRqVy9qr2FtoemsHxdCPzX8NtpxzG6qtDrVCIiPZLJPYKjgGXOuY9TX+yPA2d3WccBRZYcgL8QaARiGczUezq20vbwVwmufYe/2DE8Pu1o9hukIiAie59MFoJhwKpO0+HUvM7uBsYCa4CFwNXOuUTXFzKzaWY238zm19fXZypv+qIR2h4+h9Cat7nRrubbl32P/QYVeZ1KROQzyWQh6O4yW67L9OnAAmAf4HDgbjMr3ulJzs1yztU652qrqqp6P2kPNb//B/JW/x8/tml867Jr2H+wioCI7L0yWQjCwPBO09Ukf/l3dgnwtEtaBqwADsxgpl7xbmMuj8VOZvLXv8OBQ3aqWyIie5VMFoJ5wBgzqzGzEHA+8FyXdVYCpwCY2WDgAODjDGbqFe/Z/twQu5QvjO7a0iUisvfJ2FlDzrmYmV0J/BnwA7Odc4vMbHpq+X3Aj4CHzGwhyaak651zGzOVqbc01K9nSFGQnIDf6ygiIp9bRjuUOefmAHO6zLuv0+M1wGmZzJAJ05dfziT/KPbC6CIiO9FYQz3lHJWx9UQKhnqdRESkV6gQ9FBs8zpy6MAVj/A6iohIr1Ah6KGG8DIAgpU1HicREekdKgQ9tGltshAUDhntcRIRkd6hQtBDn/hGcFf0a1RV7+91FBGRXqFC0EOLYtXcnfgKgyvLvY4iItIrdD2CHupY9yEHFCUIBVRDRWRgUCHooW9+egMnBmuAqV5HERHpFfpZ2xOJBBXxDbQVVHudRESk16gQ9EBH8zpyiOJK1IdARAYOFYIeaFz9EQChilHeBhER6UUqBD3QvCY5MGrR0P08TiIi0ntUCHrgo9BYrotOo7JahUBEBg4Vgh5Y2l7O0+5khlSUeR1FRKTX6PTRHgit/ivHFBkBv+qniAwcKgQ9cM7qn1EbHANc7HUUEZFeo5+26UokqIxvIFKoy1OKyMCiQpCmSNNqgsRAfQhEZIBRIUhTQ7gOgJyqUd4GERHpZSoEadq8bjkARUN06qiIDCwqBGlalD+eCzpuoHK4rkMgIgOLCkGalm/N5W07lMFlxV5HERHpVTp9NE0Vn/6JiUVB/D7zOoqISK/SHkGavrz+Pr5uL3kdQ0Sk16kQpCMRpzJRT3uhrkMgIgOPCkEaIo1hAsShVH0IRGTgUSFIw8ZUH4JQZY3HSUREep8KQRo2r0teh6Bk6GiPk4iI9D4VgjT8rfBETmmfSdXwA7yOIiLS61QI0rByc4JV/uFUlRR6HUVEpNepEKShZsVjfKPwXXzqQyAiA5A6lKXhpIbHGZF7kNcxREQyQnsEexKPUZHYqD4EIjJgqRDsQWvDKoLqQyAiA1hGC4GZTTSzpWa2zMxm7GKdk8xsgZktMrPXMpnns9i4/ToEOnVURAamjB0jMDM/8CvgVCAMzDOz55xzizutUwrcA0x0zq00s0GZyvNZbd6wEoCSoft6nEREJDMyuUdwFLDMOfexc64DeBw4u8s63wCeds6tBHDObchgns9kftEpjI3MZtCIA72OIiKSEZksBMOAVZ2mw6l5ne0PlJnZq2b2jpl9s7sXMrNpZjbfzObX19dnKG73wk1tuGA+lUW5fbpdEZG+ksnTR7s76d51s/0jgVOAPOCvZvamc+6jHZ7k3CxgFkBtbW3X18iow5ffS1FBPmaT+nKzIiJ9Jq09AjP7nZlNNrOe7EGEgeGdpquBNd2s84JzbqtzbiPwOnBYD7aRceObX2C8v87rGCIiGZPuF/u9JNvz68zsDjNLp8F8HjDGzGrMLAScDzzXZZ1ngQlmFjCzfOBoYEmamTIvHtV1CERkwEurEDjnXnbOXQCMAz4BXjKzN8zsEjML7uI5MeBK4M8kv9yfcM4tMrPpZjY9tc4S4AXgfeBt4AHn3Aef9031li31n+LHYWUjvY4iIpIxaR8jMLMK4ELgIuBvwKPA8cDFwEndPcc5NweY02XefV2mZwIzexK6rzSEl1EE5FaqD4GIDFxpFQIzexo4EHgEONM5tza16L/NbH6mwnmtqXEjJa5QfQhEZEBLd4/gbufcX7pb4Jyr7cU8/crf8o9javss3h2lPgQiMnCle7B4bKoXMABmVmZmV2QoU7+xqqmV/JCfsvxuD4OIiAwI6RaCS51zm7ZNOOeagEszE6n/OKHuDmbkPYuZrkMgIgNXuoXAZ52+DVPjCIUyE6n/OHjLX9kv0O9GvRAR6VXpFoI/A0+Y2Slm9iXgMZKnfQ5YLtZORWIjHUXD97yyiMheLN2DxdcDlwGXkxw64kXggUyF6g9a1n9KkakPgYgMfGkVAudcgmTv4nszG6f/2Li6jiIgr2qU11FERDIq3X4EY4CfAgcB24fhdM4N2J5WG7e00ZYYSck++3sdRUQko9I9RvBrknsDMeBk4DckO5cNWO+FxnFGx08ZMkKFQEQGtnQLQZ5z7hXAnHOfOuduBb6UuVjeW9XYSlFOgOK8TI7ULSLivXS/5SKpIajrzOxKYDXQ7y4r2ZvOWno943JKMTvd6ygiIhmV7h7Bd4F84J9JXkjmQpKDzQ1YI9qWUBmKeh1DRCTj9rhHkOo8dq5z7jqgBbgk46k85qIRKhKNRIt0HQIRGfj2uEfgnIsDR1oWjbPQvP4TfOpDICJZIt1jBH8DnjWzJ4Gt22Y6557OSCqPNYbrKAXyB9V4HUVEJOPSLQTlQAM7ninkgAFZCNa1+VgRP4Lhww7wOoqISMal27N4wB8X6Gyh70B+Gr2O94cP2P5yIiLbpduz+Nck9wB24Jz7x15P1A+sbtxKSV6Q4lxdh0BEBr50m4b+0OlxLjAVWNP7cfqHCz68nFOCZcBpXkcREcm4dJuGftd52sweA17OSKJ+oKJjDRsLNfy0iGSHdDuUdTUGGNGbQfoLF22j0qkPgYhkj3SPEWxhx2ME60heo2DAaVqzgnLApz4EIpIl0m0aKsp0kP6icU0d5UDBYPUhEJHskFbTkJlNNbOSTtOlZjYlc7G8E44W8UjsHyipPsjrKCIifSLdYwS3OOeat0045zYBt2QmkrcWJ0ZwU+wfGTpsQB4CERHZSbqFoLv1BuRA/Q3166nK91OQMyDfnojITtL9tptvZncBvyJ50Pgq4J2MpfLQ+XXf50x/DjDR6ygiIn0i3T2Cq4AO4L+BJ4A24DuZCuWl8ug6tubt43UMEZE+k+5ZQ1uBGRnO4rlEeysVrolYsTqTiUj2SPesoZfMrLTTdJmZ/TlzsbzRuHY5AH71IRCRLJJu01Bl6kwhAJxzTQzAaxY3rV4GQP4gjToqItkj3UKQMLPt51Oa2Si6GY10b/epG8Sd0fMoHXmw11FERPpMumcN/RD4XzN7LTV9AjAtM5G882HHIO6Jn81VQ4Z5HUVEpM+ke7D4BTOrJfnlvwB4luSZQwNKZN1SDihoIy/k9zqKiEifSfdg8T8BrwDfT90eAW5N43kTzWypmS0zs12edWRm480sbmZfSy92ZkxZcTs/8/3KywgiIn0u3WMEVwPjgU+dcycDRwD1u3uCmflJdkCbBBwEfN3MdhrAJ7XezwDPz0Iqj65ja776EIhIdkm3EESccxEAM8txzn0I7OnK7kcBy5xzHzvnOoDHgbO7We8q4HfAhjSzZEQ80kI5zcSKNcaQiGSXdAtBONWP4BngJTN7lj1fqnIYsKrza6TmbWdmw0he9vK+3b2QmU0zs/lmNr++frc7Ip9Z45rkqaP+cvUhEJHsku7B4qmph7ea2VygBHhhD0+z7l6qy/QvgOudc3Gz7lbfvv1ZwCyA2trajJy22rh6OVVAgfoQiEiW6fEQm8651/a8FpDcA+g8VkM1O+9F1AKPp4pAJXCGmcWcc8/0NNfntdxfw70dV3D1yC/09aZFRDz1Wa9ZnI55wBgzqzGzEHA+8FznFZxzNc65Uc65UcBTwBVeFAGAutYinkkcz9DBg73YvIiIZzI26L5zLmZmV5I8G8gPzHbOLTKz6anluz0u0NcCq9/iuMI2coPqQyAi2SWjV19xzs0B5nSZ120BcM59K5NZ9mRi+D84zlcIXOplDBGRPpfJpqG9SrIPgYaWEJHso0IAxNo2U8ZmXYdARLKSCgGwMZzsQxAsH+VtEBERD6gQAE2pzmT5g/f1OImISN9TIQCWhg7m/I4bKa85zOsoIiJ9ToUAWNES5C13EEMqK7yOIiLS51QIgJJP/8zZhUsIBfRxiEj20TcfcNK6h/im7WnoJBGRgUmFAKiIraNN1yEQkSyV9YUg2rqJElqIl6gPgYhkp6wvBBtXLQcgUDHK2yAiIh7J+kKwaU0dAIWDdR0CEclOWV8IFuaP54T2f6es5givo4iIeCLrC8Gq5hirbQhDK0q9jiIi4omsLwTDPn6CiwveJODP+o9CRLJURq9HsDc4euPT7O9Xj2IRyV5Z/zO4IraetoJqr2OIiHgmqwtBe0sjxWwlrusQiEgWy+pCsO06BCH1IRCRLJbVhaBp3UpAfQhEJLtldSF4P+8oDog8RNl+tV5HERHxTFYXgnBTK3FfDkNKC72OIiLimawuBAcue4BrCl7A7zOvo4iIeCar+xF8YdPLVAcGeR1DRMRT2btH4BxVsfW05asPgYhkt6wtBJEtjRTSSkLXIRCRLJe1hWDDquTw06HKUd4GERHxWNYeI2hs2ECeK6ZwyH5eRxER8VTW7hEsDB3G+Pb7qNhvvNdRREQ8lbWFINzUSsjvY1BRjtdRREQ8lbVNQ0d99HNG50fx+SZ5HUVExFNZWwj23TyP0uBQr2OIiHguO5uGnKMqvo5IwTCvk4iIeC4rC0Frcz0FRHClI7yOIiLiuYwWAjObaGZLzWyZmc3oZvkFZvZ+6vaGmR2WyTzb1IeTfQiCug6BiEjmjhGYmR/4FXAqEAbmmdlzzrnFnVZbAZzonGsys0nALODoTGXapn5TC82JGoqGHZDpTYmI9HuZ3CM4CljmnPvYOdcBPA6c3XkF59wbzrmm1OSbQJ8M/LPYfwBndfwrlaMP74vNiYj0a5ksBMOAVZ2mw6l5u/Jt4E/dLTCzaWY238zm19fXf+5gqxpbyQn4qCpUHwIRkUyePtrdIP+u2xXNTiZZCI7vbrlzbhbJZiNqa2u7fY2eOOXDmzkqN0qyNUpEJLtlco8gDHQe2rMaWNN1JTM7FHgAONs515DBPNvt07qU8mC0LzYlItLvZbIQzAPGmFmNmYWA84HnOq9gZiOAp4GLnHMfZTDL3zlHVXy9+hCIiKRkrGnIORczsyuBPwN+YLZzbpGZTU8tvw+4GagA7jEzgJhzLqNXkm9pWkch7biykZncjIjIXiOjQ0w45+YAc7rMu6/T438C/imTGbraGK6jEMhRHwIRESALxxpa25JgabyWEdUHeR1FRHYhGo0SDoeJRCJeR9nr5ObmUl1dTTAYTPs5WVcIPnQjuS16De+MOtjrKCKyC+FwmKKiIkaNGkWq2VjS4JyjoaGBcDhMTU1N2s/LurGGVjVsJS/op7wg5HUUEdmFSCRCRUWFikAPmRkVFRU93pPKuj2CqUuu4YxQB2YTvY4iIruhIvDZfJbPLev2CEra10BOgdcxRET6jewqBM5RlVhPe0GfDGkkInupTZs2cc8993ym555xxhls2rSplxNlVlYVgs0Na8mjA9N1CERkN3ZXCOLx+G6fO2fOHEpLSzMRK2Oy6hjBxvBHFAM5laO8jiIiabrt+UUsXrO5V1/zoH2KueXMXZ85OGPGDJYvX87hhx/OqaeeyuTJk7ntttsYOnQoCxYsYPHixUyZMoVVq1YRiUS4+uqrmTZtGgCjRo1i/vz5tLS0MGnSJI4//njeeOMNhg0bxrPPPkteXt4O23r++ef58Y9/TEdHBxUVFTz66KMMHjyYlpYWrrrqKubPn4+Zccstt/DVr36VF154gR/84AfE43EqKyt55ZVXPvfnkVWFINyWw2ux0zl2xKFeRxGRfuyOO+7ggw8+YMGCBQC8+uqrvP3223zwwQfbT8ucPXs25eXltLW1MX78eL761a9SUVGxw+vU1dXx2GOP8Z//+Z+ce+65/O53v+PCCy/cYZ3jjz+eN998EzPjgQce4M477+TnP/85P/rRjygpKWHhwoUANDU1UV9fz6WXXsrrr79OTU0NjY2NvfJ+s6oQfBQbzI9jF7NgpC5II7K32N0v97501FFH7XBu/i9/+Ut+//vfA7Bq1Srq6up2KgQ1NTUcfnjyuidHHnkkn3zyyU6vGw6HOe+881i7di0dHR3bt/Hyyy/z+OOPb1+vrKyM559/nhNOOGH7OuXl5b3y3rLqGEFD/TpKc6AkL/0edyIiAAUFfz/b8NVXX+Xll1/mr3/9K++99x5HHHFEt+fu5+T8/Zonfr+fWCy20zpXXXUVV155JQsXLuT+++/f/jrOuZ1OBe1uXm/IqkIwue4mngjcrPOTRWS3ioqK2LJlyy6XNzc3U1ZWRn5+Ph9++CFvvvnmZ95Wc3Mzw4YlR0N++OGHt88/7bTTuPvuu7dPNzU1ceyxx/Laa6+xYsUKgF5rGsqqQlDSvpbNOft4HUNE+rmKigqOO+44DjnkEK677rqdlk+cODkHslUAAAs7SURBVJFYLMahhx7KTTfdxDHHHPOZt3XrrbdyzjnnMGHCBCorK7fPv/HGG2lqauKQQw7hsMMOY+7cuVRVVTFr1iy+8pWvcNhhh3Heeed95u12Zs597gt+9ana2lo3f/78Hj/PJRK03zaIdwafw3FX3JuBZCLSW5YsWcLYsWO9jrHX6u7zM7N3djXMf9bsETTXrybXoqA+BCIiO8iaQrBx9TIAcgelPyKfiEg2yJpCsDJazE+iX6dw5BFeRxER6Veyph/BmDFj2XD2D6geoYPFIiKdZU0hGF6ez/lH6fiAiEhXWdM0JCIi3VMhEBHp4vMMQw3wi1/8gtbW1l5MlFkqBCIiXWRbIciaYwQishf79eSd5x08BY66FDpa4dFzdl5++DfgiAtgawM88c0dl13yx91urusw1DNnzmTmzJk88cQTtLe3M3XqVG677Ta2bt3KueeeSzgcJh6Pc9NNN7F+/XrWrFnDySefTGVlJXPnzt3htW+//Xaef/552tra+OIXv8j999+PmbFs2TKmT59OfX09fr+fJ598kn333Zc777yTRx55BJ/Px6RJk7jjjjt6+untkQqBiEgXXYehfvHFF6mrq+Ptt9/GOcdZZ53F66+/Tn19Pfvssw9//GOysDQ3N1NSUsJdd93F3LlzdxgyYpsrr7ySm2++GYCLLrqIP/zhD5x55plccMEFzJgxg6lTpxKJREgkEvzpT3/imWee4a233iI/P7/XxhbqSoVARPq/3f2CD+XvfnlBxR73APbkxRdf5MUXX+SII5L9kFpaWqirq2PChAlce+21XH/99Xz5y19mwoQJe3ytuXPncuedd9La2kpjYyMHH3wwJ510EqtXr2bq1KkA5ObmAsmhqC+55BLy8/OB3ht2uisVAhGRPXDOccMNN3DZZZfttOydd95hzpw53HDDDZx22mnbf+13JxKJcMUVVzB//nyGDx/OrbfeSiQSYVdjvmVq2OmudLBYRKSLrsNQn3766cyePZuWlhYAVq9ezYYNG1izZg35+flceOGFXHvttbz77rvdPn+bbdcaqKyspKWlhaeeegqA4uJiqqureeaZZwBob2+ntbWV0047jdmzZ28/8KymIRGRPtJ5GOpJkyYxc+ZMlixZwrHHHgtAYWEhv/3tb1m2bBnXXXcdPp+PYDDIvfcmRzaeNm0akyZNYujQoTscLC4tLeXSSy/lC1/4AqNGjWL8+PHblz3yyCNcdtll3HzzzQSDQZ588kkmTpzIggULqK2tJRQKccYZZ/CTn/yk199v1gxDLSJ7Dw1D/floGGoREekRFQIRkSynQiAi/dLe1mzdX3yWz02FQET6ndzcXBoaGlQMesg5R0NDw/Z+COnSWUMi0u9UV1cTDoepr6/3OspeJzc3l+rq6h49R4VARPqdYDBITY0uK9tXMto0ZGYTzWypmS0zsxndLDcz+2Vq+ftmNi6TeUREZGcZKwRm5gd+BUwCDgK+bmYHdVltEjAmdZsG3JupPCIi0r1M7hEcBSxzzn3snOsAHgfO7rLO2cBvXNKbQKmZDc1gJhER6SKTxwiGAas6TYeBo9NYZxiwtvNKZjaN5B4DQIuZLf2MmSqBjZ/xuZnUX3NB/82mXD2jXD0zEHON3NWCTBaC7obM63ouWDrr4JybBcz63IHM5u+qi7WX+msu6L/ZlKtnlKtnsi1XJpuGwsDwTtPVwJrPsI6IiGRQJgvBPGCMmdWYWQg4H3iuyzrPAd9MnT10DNDsnFvb9YVERCRzMtY05JyLmdmVwJ8BPzDbObfIzKanlt8HzAHOAJYBrcAlmcqT8rmblzKkv+aC/ptNuXpGuXomq3LtdcNQi4hI79JYQyIiWU6FQEQky2VNIdjTcBdeMLPhZjbXzJaY2SIzu9rrTJ2Zmd/M/mZmf/A6yzZmVmpmT5nZh6nP7VivMwGY2fdSf8MPzOwxM+vZ8I+9l2O2mW0wsw86zSs3s5fMrC51X9ZPcs1M/R3fN7Pfm1lpf8jVadm1ZubMrLKvc+0um5ldlfouW2Rmd/bGtrKiEKQ53IUXYsD3nXNjgWOA7/STXNtcDSzxOkQX/wG84Jw7EDiMfpDPzIYB/wzUOucOIXlyxPkexXkImNhl3gzgFefcGOCV1HRfe4idc70EHOKcOxT4CLihr0PRfS7MbDhwKrCyrwN18hBdspnZySRHZDjUOXcw8G+9saGsKASkN9xFn3POrXXOvZt6vIXkl9owb1MlmVk1MBl4wOss25hZMXAC8CCAc67DObfJ21TbBYA8MwsA+XjUH8Y59zrQ2GX22cDDqccPA1P6NBTd53LOveici6Um3yTZj8jzXCn/DvwL3XRw7Su7yHY5cIdzrj21zobe2Fa2FIJdDWXRb5jZKOAI4C1vk2z3C5L/ERJeB+lkNFAP/DrVZPWAmRV4Hco5t5rkL7OVJIdHaXbOvehtqh0M3tY/J3U/yOM83flH4E9ehwAws7OA1c6597zO0o39gQlm9paZvWZm43vjRbOlEKQ1lIVXzKwQ+B3wXefc5n6Q58vABufcO15n6SIAjAPudc4dAWzFm2aOHaTa3M8GaoB9gAIzu9DbVHsPM/shyWbSR/tBlnzgh8DNXmfZhQBQRrIp+TrgCTPr7vutR7KlEPTboSzMLEiyCDzqnHva6zwpxwFnmdknJJvRvmRmv/U2EpD8O4adc9v2mp4iWRi89g/ACudcvXMuCjwNfNHjTJ2t3zaqb+q+V5oTeoOZXQx8GbjA9Y9OTfuSLOjvpf79VwPvmtkQT1P9XRh4OjVi89sk99g/98HsbCkE6Qx30edSlfxBYIlz7i6v82zjnLvBOVftnBtF8rP6i3PO81+4zrl1wCozOyA16xRgsYeRtlkJHGNm+am/6Sn0g4PYnTwHXJx6fDHwrIdZtjOzicD1wFnOuVav8wA45xY65wY550al/v2HgXGpf3v9wTPAlwDMbH8gRC+MkpoVhSB1QGrbcBdLgCecc4u8TQUkf3lfRPIX94LU7QyvQ/VzVwGPmtn7wOHATzzOQ2oP5SngXWAhyf9XngxRYGaPAX8FDjCzsJl9G7gDONXM6kieCXNHP8l1N1AEvJT6t39fP8nVL+wi22xgdOqU0seBi3tjT0pDTIiIZLms2CMQEZFdUyEQEclyKgQiIllOhUBEJMupEIiIZDkVApEMM7OT+tMIriJdqRCIiGQ5FQKRFDO70MzeTnVuuj91PYYWM/u5mb1rZq+YWVVq3cPN7M1OY+mXpebvZ2Yvm9l7qefsm3r5wk7XUXh02/gwZnaHmS1OvU6vDCks0lMqBCKAmY0FzgOOc84dDsSBC4AC4F3n3DjgNeCW1FN+A1yfGkt/Yaf5jwK/cs4dRnK8obWp+UcA3yV5PYzRwHFmVg5MBQ5Ovc6PM/suRbqnQiCSdApwJDDPzBakpkeTHNTrv1Pr/BY43sxKgFLn3Gup+Q8DJ5hZETDMOfd7AOdcpNMYOm8758LOuQSwABgFbAYiwANm9hWgX4y3I9lHhUAkyYCHnXOHp24HOOdu7Wa93Y3JsrvhgNs7PY4DgdQYWEeRHH12CvBCDzOL9AoVApGkV4Cvmdkg2H6d35Ek/498LbXON4D/dc41A01mNiE1/yLgtdS1JMJmNiX1Gjmp8e27lboORYlzbg7JZqPDM/HGRPYk4HUAkf7AObfYzG4EXjQzHxAFvkPy4jcHm9k7QDPJ4wiQHM75vtQX/cfAJan5FwH3m9ntqdc4ZzebLQKeteSF7g34Xi+/LZG0aPRRkd0wsxbnXKHXOUQySU1DIiJZTnsEIiJZTnsEIiJZToVARCTLqRCIiGQ5FQIRkSynQiAikuX+P5I5VAKlfZajAAAAAElFTkSuQmCC)



이 예에서는 1에폭마다 모든 후련 데이터와 시험 데이터에 대한 정확도를 계산하고 그 결과를 기록합니다. 정확도를 1 에폭마다 계산하는 이유는 for 문 안에서 매번 계산하기에는 시간이 오래 걸리고, 또 그렇게까지 자주 기록할 필요도 없기 때문이죠.



### 6. 정리

- 이번 장에서는 신경망 학습에 대해서 설명했습니다. 가장 먼저 신경망이 학습을 수행할 수 있도록 손실 함수라는 '지표'를 도입했습니다.

- 이 손실 함수를 기준으로 그 값이 가장 작아지는 가중치 매개변수 값을 찾아내는 것이 신경망 학습의 목표입니다.
-  가능한 한 작은 손실 함수의 값을 찾는 수법으로 경사법을 소개했습니다. 경사법은 함수의 기울기를 이용하는 방법입니다.



배운내용

1. 기계학습에서 사용하는 데이터셋은 훈련 데이터와 시험 데이터로 나뉜다.
2. 훈련 데이터로 학습한 모델의 범용 능력을 시험 데이터로 평가한다.
3. 신경망 학습은 손실 함수를 지표로, 손실 함수의 값이 작아지는 방향으로 가중치 매개변수를 갱신한다.
4. 가중치 매개변수를 갱신할 때는 가중치 매개변수의 기울기를 이용하고, 기울어진 방향으로 가중치의 값을 갱신하는 작업을 반복한다.
5. 아주 작은 값을 주었을 때의 차분으로 미분하는 것을 수치 미분이라고 한다.
6. 수치 미분을 이용해 가중치 매개변수의 기울기를 구할 수 있다.
7. 수치 미분을 이용한 계산에는 시간이 걸리지만, 그 구현은 간단하다. (다음 장에서 구현하는 오차역전파법은 기울기를 고속으로 구할 수 있다.)





