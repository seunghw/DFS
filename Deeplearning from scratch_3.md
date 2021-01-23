# Deeplearning from scratch

[TOC]

## CHAPTER 3 신경망



- 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습하는 능력이 신경망의 중요한 성질입니다.
- 이번 장에서는 신경망의 개요를 설명하고 신경망이 입력 데이터가 무엇인지 식별하는 처리 과정을 자세히 알아봅니다.





### 1. 퍼셉트론에서 신경망으로





#### 	1.1 신경망의 예

- 가장 왼쪽 줄을 **입력층**, 맨 오른쪽 줄을 **출력층**, 중간 줄을 **은닉층**이라고 합니다.

- 은닉층의 뉴런은 사람 눈에는 보이지 않습니다. 

- 0층 입력층, 1층이 은닉층, 2층이 출력층이 됩니다.



![image-20201215145811861](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215145811861.png)

> 가중치를 갖는 층은 2개이기 때문에 2층 신경망이라고도 합니다.



#### 1.2 퍼셉트론 복습



![img](https://t1.daumcdn.net/cfile/tistory/99BDCE4D5B98A1022C)



우선 퍼셉트론을 복습해봅시다.

x1과 x2라는 두 신호가 입력받아 y를 출력하는 퍼셉트론입니다. 이 퍼셉트론을 수식으로 나타내면 아래와 같습니다.

![image-20201214160201876](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201214160201876.png)

여기서 b는 **편향**을 나타내는 매개변수로, 뉴런이 얼마나 쉽게 활성화되느냐를 제어합니다.

한편 w1과 w2는 각 신호의 **가중치**를 나타내는 매개변수로, 각 신호의 영향력을 제어합니다.

이 때 편향값을 명시한다면 아래처럼 나타낼 수 있습니다.

![image-20201215151404861](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215151404861.png)

x1,x2,1이라는 3개의 신호가 뉴런에 입력되어, 각 신호에 가중치를 곱한 후, 다음 뉴런에 전달됩니다.<br>다음 뉴런에서는 이 신호들의 값을 더하여, 그 합이 0을 넘으면 1을 출력하고 그렇지 않으면 0을 출력합니다.



#### 1.3 활성화 함수의 등장



- 입력 신호의 총합을 출력 신호로 변환하는 함수를 일반적으로 **활성화 함수**라 합니다. 

```mathematica
(1) a = b + w1*x1 + w2*x2 
(2) y = h(a)
```

(1) 은 가중치가 달리 ㄴ입력 신호와 편향의 총합을 계산하고, 이를 a라 합니다.

(2) 는 a를 함수 h()에 넣어 y를 출력하는 흐름입니다.



![image-20201215152306813](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215152306813.png)

활성화 함수의 처리과정을 나타낸 것입니다.

가중치 신호를 조합한 결과가 a라는 노드가 되고, 활성화 함수 h()를 통과하여 y라는 노드로 변환되는 과정이 나타나 있습니다.<br>또한 신경망의 동작을 더 명확히 드러내고자 할 때는 오른쪽 그림처럼 활성화 처리 과정을 명시합니다.

> 이 책에서는 뉴런 = 노드 같은 말 입니다.



### 2. 활성화 함수

```mathematica
y = h(b + w1x1 + w2x2) 

h(x) = 0 (x <= 0) or h(x) = 1 (x > 0)
```



위와 같은 활성화 함수는 임계값을 경계로 출력이 바뀌는데, 이런 함수를 **계단함수**라고 합니다.

그래서 "퍼셉트론에서는 활성화 함수로 계단 함수를 이용한다" 라고 할 수 있습니다.



#### 2.1 시그모이드 함수

![image-20201215153842822](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215153842822.png)

- 신경망에서 자주 이용하는 함수입니다.
- e는 자연상수로 2.7182...의 값을 갖는 실수입니다.
- 쉽게 말해서 입력을 주면 출력하는 변환기 역할, 특정 값을 출력합니다.
- 신경망에서는 활성화 함수로 시그모이드 함수를 이용하여 신호를 변환하고, 그 변환된 신호를 다음 뉴런에 전달합니다.



#### 2.2 계단 함수 구현하기

계단 함수를 구현하면 아래와 같습니다.

```python
(1)
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

(2)    
def step_function(x):
    y = x > 0
    return y.astype(np.int)
```

(1) 인수 x는 실수(부동소수점)만 받아들입니다. 즉, step_function(3.0) 은 되지만 넘파이 배열을 인수로 넣을 수는 없습니다. 그래서 우리는 앞으로를 위해서 (2) 처럼 생각할 수 있습니다.



```python
import numpy as np
x = np.array([-1.0, 1.0, 2.0])
x
>>> array([-1.,  1.,  2.])
y = x > 0
y
>>>array([False,  True,  True]) 
```

넘파이 배열에 부등호 연산을 수행하면 배열의 원소 각각에 부등호 연산을 수행한 bool 배열이 생성됩니다. 배열 x의 원소 각각이 0보다 크면 True로, 0 이하면 False로 변환한 새로운 배열 y가 생성됩니다.



여기서 우리가 원하는 계단 함수는 int형을 출력하는 함수기 때문에 int형으로 바꾸면 아래와 같습니다.

```python 
y = y.astype(np.int)
y
>>> array([0, 1, 1])
```

이처럼 넘파이 배열의 자료형을 변환할 때는 astype() 메서드를 이용합니다.

#### 2.3 계단 함수의 그래프



이제 matplotlib 라이브러리를 이용하여 계단 함수를 그래프로 그립니다.



```python
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)# [-5.0, -4.9, -4.8, ... 4.9]
y = step_function(x) # 계단함수를 실행하여 결과값을 다시 배열로 만들어줌
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show() # 출력
```



- np.arange(-5.0, 5.0, 0.1)은 -5.0에서 5.0 전까지 0.1 간격의 넘파이 배열을 생성합니다.

- step_function()은 인수로 받은 넘파이 배열의 원소 각각을 인수로 계단 함수 실행해, 그 결과를 다시 배열로 만들어 돌려줍니다.

  

![img](data:image/png;base64,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)

이렇게 계단함수는 0을 경계로 출력이 0에서 1로 바뀝니다.



#### 2.4 시그모이드함수 구현하기

시그모이드 함수는 파이썬으로 다음과 같이 할 수있습니다.

```python
def sigomoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)
>>> array([0.26894142, 0.73105858, 0.88079708])
```

여기서  인수 x가 넘파이 배열이어도 올바른 결과가 나온다는 정도만 기억해두면 됩니다.

이 함수가 넘파이 배열도 처리할 수 있는 이유는 바로 넘파이의 브로드캐스트에 있습니다.

브로드캐스트란 넘파이 배열과 스칼라값의 연산을 넘파이 배열의 원소 각각과 스칼라값의 연산을 바꿔 수행하는 것 입니다.



시그모이드 함수를 그래프로 그리면 다음과 같습니다.

```python
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()
```

![img](data:image/png;base64,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)

> 시그모이드는 S자 모양이라는 뜻입니다. 자주 등장하니까 '시그모이드 = S자 모양'이란 점만 확실히 기억해주세요.



#### 2.5 시그모이드 함수와 계단 함수 비교



![image-20201215163159999](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215163159999.png)



시그모이드와 계단함수를 비교해봅시다.

- 차이점
  - 우선 매끄러움의 차이가 가장 먼저 눈에 들어올 것 입니다. 이 매끈함이 신경망 학습에서 아주 중요한 역할을 하게 됩니다.
  - 비유하자면 계단함수는 시시오도시, 시그모이드 함수는 물레방아에 가깝죠

[^시시오도시]: 대나무통으로 만든 일본 전통 정원에서 볼 수 있는 장식.

- 공통점
  - 두 함수의 공통점은 같은 모양. 즉, 입력이 커지면(중요하면) 큰 값을 출력하고 입력이 작으면(중요하지 않으면) 작은 값을 출력합니다. 
  - 입력이 아무리 작거나 커도 0에서 1 사이라는 점



#### 2.6 비선형 함수

저 둘의 또 다른 중요한 공통점은 둘 다 비선형 함수 입니다.

신경망에서는 활성화 함수로 비선형 함수를 사용해야 합니다. 선형 함수를 이용하면 신경망의 층을 깊게 하는 의미가 없어지기 때문입니다.



선형함수의 문제점

1. 층을 아무리 깊게 해도 '은닉층이 없는 네트워크'로도 똑같은 기능을 할 수 있다.

> 선형함수인 h(x)=cx를 활성화함수로 사용한 3층 네트워크를 떠올려 보세요. 이를 식으로 나타내면 y(x)=h(h(h(x)))가 됩니다. 이는 실은 y(x)=ax와 똑같은 식입니다. a=c3이라고만 하면 끝이죠. 즉, 은닉층이 없는 네트워크로 표현할 수 있습니다. 뉴럴네트워크에서 층을 쌓는 혜택을 얻고 싶다면 활성화함수로는 반드시 비선형 함수를 사용해야 합니다.
>



#### 2.7 ReLU 함수

시그모이드 함수는 신경망 분야에서 오래전부터 이용해왔으나, 최근에는 ReLU함수를 주로 이용합니다.

ReLU는 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하이면 0을 출력하는 함수 입니다.

```python 
def relu(x):
    return np.maximum(0,x)# maximum은 두 입력 중 큰 값을 선택해 반환하는 함수
```



![img](data:image/png;base64,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)



### 3. 다차원 배열의 계산

넘파이의 다차원 배열을 사용한 계산법을 숙달하면 신경망을 효율적으로 구현할 수 있습니다.





#### 3.1 다차원 배열

n차원으로 나열하는 것을 통틀어 다차원 배열이라고 합니다.

아래는 1차원 배열입니다.

```python
import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
>>> [1 2 3 4]
np.ndim(A) # 배열 차원의 수 확인하는 함수
>>> 1
A.shape # 배열의 형상 확인하는 함수
>>> (4,)
A.shape[0] # 튜플을
>>> 4
```

> A.shape이 튜플을 반환하는 것에 주의!! 1차원 배열이라도 다차원 배열일 때와 통일된 형태로 결과를 반환하기 위함입니다.



```python
B = np.array([[1,2],[3,4],[5,6]])
print(B)
>>>
[[1 2]
 [3 4]
 [5 6]]
np.ndim(B)
>>> 2
B.shape
>>> (3, 2)
```

![image-20201215170636059](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215170636059.png)



#### 3.2 행렬의 곱

![image-20201215171258176](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215171258176.png)



행렬의 곱은 위와 같이 계산합니다.

파이썬으로 구현하면 다음과 같습니다.

```python
a = np.array([[1,2],[3,4]])
a.shape
>>> (2, 2)
b = np.array([[5,6],[7,8]])
b.shape
>>> (2, 2)
np.dot(a, b)
>>>
array([[19, 22],
       [43, 50]])
```

- np.dot()은 입력이 1차원 배열이면 벡터를, 2차원 배열이면 행렬 곱을 계산합니다.
- 행렬의 곱에서는 피연산자의 순서가 다르면 다른 결과가 나타나는 것을 주의해야 합니다.



![image-20201215172705792](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215172705792.png)

위와같이 행렬의 곱에서는 대응하는 차원의 원소 수를 일치시켜야 합니다.



#### 3.3 신경망에서의 행렬 곱

![image-20201215172900917](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215172900917.png)

행렬의 곱으로 신경망의 계산을 수행합니다.

이 신경망은 편향과 활성화 함수를 생략하고 가중치만 갖습니다.

이 구현에서도 X, W, Y의 형상을 주의해서 보세요. 특히 X와 W의 대응하는 차원의 원소 수가 같아야 한다는 걸 잊지 말아야 합니다.

```python
X = np.array([1,2])
X.shape
>>> (2,)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
>>>
[[1 3 5]
 [2 4 6]]
W.shape
>>> (2, 3)
Y = np.dot(X, W)
print(Y)
>>> [ 5 11 17]
```

다차원 배열의 스칼라곱을 구해주는 np.dot 함수를 사용하면 이처럼 단번에 Y값을 구할 수 있습니다.



### 4. 3층 신경망 구현하기

![image-20201215173725974](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215173725974.png)



- 3층 신경망 : 입력층은 2개, 첫 번째 은닉층(1층)은 3개, 두 번째 은닉층(2층)은 2개, 출력층(3층)은 2개의 뉴런으로 구성된다.



#### 4.1 표기법

![image-20201215191518214](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215191518214.png)



- 가중치 오른쪽 아래의 인덱스 번호는 '다음 층 번호, 앞 층 번호' 순서로 작성합니다.

#### 4.2 각층의 신호 전달 구현하기



![](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215191925411.png)



- 편향은 오른쪽 아래 인덱스가 하나밖에 없다는 것에 주의하세요. 이는 앞 층의 편향 뉴런(뉴런1)이 하나뿐이기 때문입니다.

- 세번째 처럼 행렬의 곱을 이용하면 간소화가 가능합니다.

  



```python
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
>>> (2, 3)
print(X.shape)
>>> (2,)
print(B1.shape)
>>> (3,)
A1 = np.dot(X, W1) + B1
```

이 계산은 앞 절에서 한 계산과 같습니다. W1ㅇ느 2*3 행렬, X는 원소가 2개인 1차원 배열입니다.

여기서도 역시 W1과 X의 대응하는 차원의 원소 수가 일치하고 있습니다.

이어서 1층의 활성화 함수에서의 처리를 살펴보겠습니다. 이 활성화 함수의 처리를 그림으로 그리면 아래와 같습니다.

![image-20201215193019462](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215193019462.png)

은닉층에서 가중치의 합(가중 신호와 편향의 총합)을 a로 표기하고 활성화 함수 h()로 변환된 신호를 z로 표기합니다. 여기에서는 활성화 함수로 시그모이드 함수를 사용합니다.

이를 파이썬으로 구현하면 아래와 같습니다.

```python
Z1 = sigmoid(A1)

print(A1) # [0.3 0.7 1.1]
print(Z1) # [0.57444252 0.66818777 0.75026011]
```

이 sigmoid() 함수는 앞에서 정의한 함수입니다. 이 함수는 넘파이 배열을 반환합니다.

이어서 1층에서 2층으로 가는 과정과 그 구현을 살펴보죠

![image-20201215195207811](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215195207811.png)



```python
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape) # (3,)
print(W2.shape) # (3, 2)
print(B2.shape) # (2,)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
```

이구현은 1층의 출력 Z1이 2층의 입력이 된다는 점을 제외하면 조금 전의 구현과 똑같습니다.<br>

마지막으로 2층에서 출력층으로 신호 전달입니다. 다 똑같지만 딱 하나, 활성화 함수만 지금까지의 은닉층과 다릅니다.



![image-20201215195809195](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201215195809195.png)

```python
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) # 혹은 Y = A3
```

여기에서는 항등함수인  identity_function()을 정의하고, 이를 출력층의 활성화 함수로 이용했습니다. 항등 함수는 입력을 그대로 출력하는 함수입니다. 위 그림에서는 출력층의 활성화 함수를 ㅁ()로 표시하여 은닉층의 활성화 함수h()와는 다름을 명시했습니다. (시그마라고 읽습니다)



> 출력층의 활성화 함수는 풀고자 하는 문제의 성질에 맞게 정합니다. 예를 들어 회귀에는 항등 함수를,2클래스 분류에는 시그모이드 함수를, 다중 클래스 분류에는 소프트맥스 함수를 사용하는 것이 일반적 입니다.



#### 4.3 구현 정리

지금까지의 구현을 정리하겠습니다. 신경망 구현의 관례에 따라 가중치만 W1과 같이 대문자로 쓰고, 그 외 편향과 중간 결과 등은 모두 소문자로 썼습니다.

```python
def init_network(): 
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) +b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) +b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [0.31682708 0.69627909]
```



- init_network() 함수 : 가중치와 편향을 초기화하고 이들을 딕셔너리 변수인 network에 저장합니다.
-  이 딕셔너리 변수 network에는 각 층에 필요한 매개변수(가중치와 편향)를 저장합니다. 
-  forward()함수는 입력 신호를 출력으로 변환하는 처리 과정을 모두 구현하고 있습니다.

### 5. 출력층 설계하기



#### 5.1 항등 함수와 소프트맥스 함수 구현하기

**항등함수**는 입력을 그대로 출력합니다. 입력 = 출력

항등 함수에 의한 변환은 은닉층에서의 활성화 함수와 마찬가지로 화살표를 그립니다.

<img src="C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201216155226500.png" alt="image-20201216155226500" style="zoom: 50%;" />

한편, 분류에서 사용하는 **소프트맥스 함수**의 식은 다음과 같습니다.



![image-20201216155414473](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201216155414473.png)

exp는 지수 함수입니다. n은 출력층의 뉴런 수, yk는 그중 k번째 출력임을 뜻합니다.

소프트맥스 함수의 분자는 입력 신호 ak의 지수 함수, 분모는 모든 입력 신호의 지수 함수의 ㅎ바으로 구성됩니다.

```python
import numpy as np

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a) # 지수 함수
print(exp_a)
>>> [ 1.34985881 18.17414537 54.59815003]
sum_exp_a = np.sum(exp_a)
print(sum_exp_a) 
>>> 74.1221542101633
y = exp_a / sum_exp_a
print(y)
>>> [0.01821127 0.24519181 0.73659691]
```

이제 이 논리 흐름을 파이썬 함수로 정의하겠습니다.

```python
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
```



#### 5.2 소프트맥스 함수 구현시 주의점



앞서 구현한 softmax()함수는 컴퓨터로 계산할 때는 결함이 있습니다. 바로 오버플로 문제입니다.

소프트맥스는 지수함수를 사용하는데 지수함수는 아주 쉽게 큰 값을 내기때문에 이 값들로 나눗셈을 하면 결과 수치가 '불안정'해집니다.

[^오버플로]: 표현할 수 있는 수의 범위가 한정되어 너무 큰 값은 표현할 수 없다는 문제가 발생



이 문제를 해결하도록 소프트맥스 함수 구현을 개선하면 아래와 같습니다.

![image-20201216160641117](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201216160641117.png)

- 첫 번째 변형에서는 c라는 임의의 정수를 분자와 분모 양쪽에 곱함.

- C를 지수 함수 exp() 안으로 옮겨 logC로 만듭니다.
- 마지막으로 logC를 C'라는 새로운 기호로 바꿉니다.
- 오버플로를 막기 위해 입력 신호 중 최댓값을 이용하는 것이 일반적입니다.



```python
a = np.array([1010, 1000, 990])
np.exp(a) / np.sum(np.exp(a)) #소프트맥스 함수의 계산
>>> array([nan, nan, nan]) # 제대로 계산이 안됨

c = np.max(a)
a - c
>>> array([  0, -10, -20])

np.exp(a - c) / np.sum(np.exp(a - c))
>>> array([9.99954600e-01, 4.53978686e-05, 2.06106005e-09])
```



이렇게 아무런 조치 없이 계산하면 nan이 출력됩니다. 하지만  입력 신호 중 최댓값을 빼주면 올바르게 계산이 가능합니다. 이를 바탕으로 소프트맥스함수를 다시 구현하면 아래와 같습니다.

```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
    
```

#### 5.3 소프트맥스 함수의 특징



```python
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
>>> [0.01821127 0.24519181 0.73659691]
np.sum(y)
>>> 1.0
```



소프트맥스 함수의 중요한 성질

1. 소프트맥스 함수의 출력은 0에서 1.0 사이의 실수입니다. 

2. 소프트맥스 함수의 출력의 총합은 1



이 성질 덕분에 소프트맥스 함수의 출력을 '확률'로 해석할 수 있습니다.

위 예에서 y[0]은 1.8% y[1]의 확률은 24.5%, y[2]의 확률은 73.7%로 해석이 가능하여 문제를 확률적으로 대응할 수 있게 되는 것 입니다.



여기서 주의점은 소프트맥스 함수를 적용해도 각 원소의 대소 관계는 변하지 않습니다.

이는 지수 함수 y = exp(x)가 단조 증가 함수이기 때문입니다.

예를 들어 a에서 가장 큰 원소는 2번째, y에서 가장 큰 원소도 2번째.



신경망으로 분류할 때 출력층의 소프트맥스 함수를 생략해도 됩니다.<br>현업에서도 지수 함수 계단에 드는 자원 낭비를 줄이고자 출력층의 소프트맥스 함수는 생략하는 것이 일반적입니다.

> 기계학습의 문제 풀이는 학습과 추론의 두 단계를 거쳐 이뤄집니다.
>
> 학습 단계 에서 모델을 학습하고 추론 단계에서 앞서 학습한 모델로 미지의 데이터에 대해서 추론을 수행합니다. 방금 설명한 대로 추론 단계에서는 출력층의 소프트맥스 함수를 생략하는 것이 일반적입니다. 한편, 신경망을 학습시킬 대는 출력층에서 소프트맥스 함수를 사용합니다.



#### 5.4 출력층의 뉴런 수 정하기



출력층의 뉴런 수는 풀려는 문제에 맞게 정해야합니다.

분류에서는 분류하고 싶은 클래스 수로 설정하는 것이 일반적입니다. 예를 들어 숫자 0부터 9 중 하나로 분류하는 문제라면 출력층의 뉴런을 10개로 설정합니다.





### 6. 손글씨 숫자 인식



신경망의 구조를 배웠으니 실전 예인 손글씨 숫자 분류로 적용해봅시다.

이번 과정에서는 이미 학습된 매개변수를 사용하여 학습 과정은 생략하고, 추론 과정만 구현할 겁니다. 이 추론 과정을 신경망의 **순전파**라고도 합니다.



#### 6.1 MNIST 데이터셋



이번 예에서 사용하는 데이터셋은 MNIST라는 손글씨 숫자 이미지 집합니다.

기계학습 분야에서 아주 유명한 데이터셋으로, 다양한 곳에서 이용하고 있습니다.

MNIST 데이터셋은 0부터 9까지 숫자 이미지로 구성됩니다.



<img src="C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201216163932989.png" alt="image-20201216163932989" style="zoom:67%;" />

```python
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)

```



- load_mnist 함수는 읽은 MNIST 데이터를 "(훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)" 형식으로 반환합니다. 인수로는 normalize, flatten, one_hot_label 세 가지를 설정할 수 있습니다. 세 인수 모두 bool값입니다. 
- normalize : 입력 이미지의 픽셀 값을 0.0 ~ 1.0 사이의 값으로 정규화할지를 정합니다. false로 설정하면 입력 이미지의 픽셀은 원래 값 그대로 0 ~ 255 사이의 값을 유지합니다.
- flatten :  입력 이미지를 편탄하게, 즉 1차원 배열로 만들지를 정합니다. False로 설정하면 입력 이미지를 1 * 28 * 28 의 3차원 배열로, True로 설정하면 784개의 원소로 이루어진 1차원 배열로 저장합니다.
-  one_hot_label : **원-핫 인코딩** 형태로 저장할지를 정합니다.

[^원-핫 인코딩]: 정답을 뜻하는 원소만 1이고 나머지는 모두 0인 배열 one_hot_label이 False면 7이나 2와 같이 숫자 형태의 레이블을 저장하고, True일 때는 레이블을 원-핫 인코딩하여 저장



![image-20201216173007760](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201216173007760.png)



여기서 주의 사항으로 fatten=True로 설정해 읽어 들인 이미지는 1차원 넘파이 배열로 저장되어 있다는 것 입니다.

그래서 이미지를 다시 원래 형상인 28*28로 변형해야합니다. reshaple()메서드에 원하는 형상을 인수로 지정하면 넘파이 배열의 형상을 바꿀 수 있습니다. 또한 넘파이로 저장된 이미지 데이터를 PIL용 데이터객체로 변환해야 하며, 이 변환은 Image.fromarray()가 수행합니다.



#### 6.2 신경망의 추론 처리



- 이 MNIST 데이터셋을 가지고 신경망을 구현할 차례입니다. 입력층 뉴런을 784개, 출력층 뉴런을 10개로 구성합니다.

- 입력층 뉴런이 784개인 이유는 28*28 = 784.

- 출력층 뉴런이 10개인 이유는 문제가 0에서 9까지의 숫자 구분이기 때문

한 편, 은닉층은 총 두 개로, 첫 번째 은닉층에는 50개의 뉴런을, 두 번째 은닉층에는 100개의 뉴런을 배치할 것 입니다. 50과 100은 임의로 정한 값입니다.

```python
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
```

init_network()에서는 pickle 파일인 sample_weight.pkl에 저장된 '학습된 가중치 매개변수'를 읽습니다. 이 파일에는 가중치와 편향 매개변수가 딕셔너리 변수로 저장되어 있습니다.

이제 이 세 함수로 신경망에 의한 추론을 수행해보고, **정확도**도 평가해봅시다.



```python
x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

```



1. 가장 먼저 MNIST 데이터셋을 얻고 네트워크를 생성

2. for 문을 돌며 x에 저장된 이미지 데이터를 1장씩 꺼내 predict() 함수로 분류

   predict() : 각 레이블의 확률을 넘파이 배열로 반환합니다. ex[0.1, 0.3, 0.2] 일 때 이미지가 숫자 0일 확률이 0.1, 1일 확률이 0.3 ... 이런식으로 해석합니다.

3. np.argmax() 함수로 이 배열에서 값이 가장 큰(확률이 가장 높은) 원소의 인덱스를 구합니다. <br>이것이 바로 예측 결과죠
4. 마지막으로 신경망이 예측한 답변과 정답 레이블을 비교하여 맞힌 숫자(accuracy_cnt)를 세고, 이를 전체 이미지 숫자로 나눠 정확도를 구합니다.



이 코드를 실행하면 "Accuracy: 0.9352" 라고 출력합니다. 올바르게 분류한 비율이 93.52%라는 뜻입니다. 점점 이 정확도를 높여갈 것 입니다.



또한 이 예제에서는 load_mnist 함수의 인수인 normalize를 True로 설정했습니다.

이처럼 데이터를 특정 범위로 변환하는 처리를 **정규화**라 하고, 신경망의 입력 데이터에 특정 변환을 가하는 것을 **전처리**라 합니다.여기에서는 입력 이미지 데이터에 대한 전처리 작업으로 정규화를 수행한 셈입니다.



#### 6.3 배치 처리



입력 데이터와 가중치 매개변수의 '형상'에 주의해서 조금 전의 구현을 다시 살펴봅시다.



```python
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']

x.shape
>>> (10000, 784)
x[0].shape
>>> (784,)
W1.shape
>>> (784, 50)
W2.shape
>>> (50, 100)
W3.shape
>>> (100, 10)
```



이 결과에서 다차원 배열의 대응하는 차원의 원소 수가 일치함을 확인할 수 있습니다.(편향 생략)

최종 결과로는 원소가 10개인 1차원 배열 y가 출력되는 점도 확인합시다.



![image-20201216182841392](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201216182841392.png)

전체적으로 보면 원소 784개로 구성된 1차원 배열이 입력되어 마지막에는 원소가 10개인 1차원 배열이 출력되는 흐름입니다. 이는 이미지 데이터를 1장만 입력했을 때의 처리 흐름입니다.



그럼 이미지 100개를 묶어 predict() 함수에 한 번에 넘기는 것이죠. x의 형상을 100 * 784로 바꿔서 100장 분량의 데이터를 하나의 입력 데이터로 표현하면 될 겁니다.



![image-20201216183029472](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201216183029472.png)

이처럼 하나로 묶은 입력 데이터를 **배치**라 합니다. 배치가 곧 묶음이란 의미죠. 



> 배치 처리는 컴퓨터로 계산할 때 큰 이점을 줍니다. 크게 두가지 이유가 있는데, 하나는 수치 계산 라이브러리 대부분이 큰 배열을 효율적으로 처리할 수 있도록 고도로 최적화되어 있기 때문입니다. 그리고 커다란 신경망에서는 데이터 전송이 병목으로 작용하는 경우가 자주 있는데, 배치 처리를 함으로써 버스에 주는 부하를 줄인다는 것이 두 번째 이유입니다. (정확히는 느린I/O를 통해 데이터를 읽는 횟수가 줄어, 빠른 CPU나 GPU로 순수 계산을 수행하는 비율이 높아집니다.)

이제 배치 처리를 구현해봅시다.

```python 
x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

```

우선 range() 함수입니다. range()함수는 range(start,end)처럼 인수를 2개 지정해 호출하면 start에서 end-1까지의 정수로 이루어진 리스트를 반환합니다. 또 range(start, end, step)처럼 인수를 3개 지정하면 start에서 end-1까지 step 간격으로 증가하는 리스트를 반환합니다.



```python
list(range(0, 10))
>>> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
list(range(0, 10, 3))
>>> [0, 3, 6, 9]
```

이 range() 함수가 반환하는 리스트를 바탕으로 x[i:i+batch_size] 에서 입력 데이터를 묶습니다. x[i:i+batch_size]은 입력 데이터의 i번째부터 i+batch_size번째까지의 데이터를 묶는다는 의미죠.

이 예에서는 batch_size가 100이므로 1장씩 묶어 꺼냅니다.

그리고 앞에서 나온 argmax()는 최댓값의 인덱스를 가져옵니다. 여기서 axis=1이라는 인수를 추가한 것에 주의합시다. 이는 100 * 10의 배열 중 1번째 차원을 구성하는 각 원소에서 최댓값의 인덱스를 찾도록한 것입니다.

```python
x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis=1)
print(y)
>>> [1 2 1 0]
```



마지막으로 배치 단위로 분류한 결과를 실제 답과 비교합니다. 이를 위해 연산자 == 를 사용해서 넘파이 배열끼리 비교하여 True/False로 구성된 bool 배열을 만들고, 이 결과 배열에서 True가 몇 개인지 셉니다.

```python
y = np.array([1, 2, 1, 0])
t = np.array([1, 2, 0, 0])
print(y==t)
>>> [ True  True False  True]
np.sum(y==t)
>>> 3
```



### 7. 정리



- 신경망에서는 활성화 함수로 시그모이드 함수와 ReLU 함수 같은 매끄럽게 변화하는 함수를 이용한다.
- 넘파이의 다차원 배열을 잘 사용하면 신경망을 효율적으로 구현할 수 있다.
- 기계학습 문제는 크게 회귀와 분류로 나눌 수 있다.
- 출력층의 활성화 함수로는 회귀에서는 주로 항등 함수를, 분류에서는 주로 소프트맥스 함수를 이용한다.
- 분류에서는 출력층의 뉴런 수를 분류하려는 클래스 수와 같게 설정한다.
- 입력 데이터를 묶은 것을 배치라 하며, 추론 처리를 이 배치 단위로 진행하면 결과를 훨씬 빠르게 얻을 수 있다.



