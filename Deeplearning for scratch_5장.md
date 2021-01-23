# Deeplearning from scratch

[TOC]



## CHAPTER 5 오차역전파법



앞장에서 신경망의 가중치 매개변수의 기울기는 수치 미분을 사용해 구했습니다.

단순하고 구현하기도 쉽지만 계산 시간이 오래 걸린다는 게 단점입니다.

이번 장에서는 가중치 매개변수의 기울기를 효율적으로 계산하는 오차역전파법을 배워보겠습니다.





### 1. 계산 그래프



- 계산그래프는 계산 과정을 그래프로 나타낸 것입니다.

- 이 그래프는 복수의 노드와 에지로 표현됩니다.



#### 1.1 계산 그래프로 풀기



그래프의 흐름은 다음과 같다.

1. 계산 그래프를 구성한다.
2. 그래프에서 계산을 왼쪽에서 오른쪽으로 진행한다.
3. 여기서 계산을 왼쪽에서 오른쪽으로 진행하는 단계를 **순전파**라고 합니다.
4. 반대반향으로 되어있는 것을 **역전파**라고 합니다. 역전파는 미분을 계산할 때 중요한 역할을 합니다.



#### 1.2 국소적 계산

계산 그래프의 특징은 국소적 계산을 전파함으로써 최종 결과를 얻는다는 점에 있습니다. 국소적이란 자신과 직접 관계된 작은 범위라는 뜻입니다.

결국 전체에서 어떤 일이 벌어지든 상관없이 자신과 관계된 정보만으로 결과를 출력할 수 있다는 것입니다. 이처럼 계산그래프는 국소적 계산에 집중합니다. 국소적인 계산은 단순하지만, 그 결과를 전달함으로써 전체를 구성하는 복잡한 계산을 해낼 수 있습니다.



#### 1.3 왜 계산 그래프로 푸는가?



계산 그래프의 이점

1. 국소적 계산. 전체가 아무리 복잡해도 각 노드에서는 단순한 계산에 집중항 ㅕ문제를 단순화할 수 있습니다.
2. 중간 계산 결과를 모두 보관할 수 있습니다.
3. 역전파를 통해 '미분' 을 효율적으로 계산할 수 있다는 점 (제일 중요)



![image-20201222175950310](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201222175950310.png)



그림과 같이 역전파는 순전파와는 반대 방향의 화살표(굵은 선)으로 그립니다.

이 전파는 국소적 미분을 전달하고 그 미분 값은 화살표의 아래에 적습니다.

이 예에서 역전파는 오른쪽에서 왼쪽으로 1 > 1.1 > 2.2 순으로 미분 값을 전달하고 사과 가격에 대한 지불 금액의 미분값은 2.2라고 할 수 있습니다.



이처럼 계산 그래프의 이점은 순전파와 역전파를 활용해서 각 변수의 미분을 효율적으로 구할 수 있다는 것입니다.



### 2. 연쇄법칙

국소적 미분을 전달하는 원리는 **연쇄법칙**에 따른 것 입니다. 이번 절에서는 연쇄법칙을 설명하고 그것이 계산 그래프 상의 역전파와 같다는 사실을 밝히겠습니다.



#### 2.1 연쇄법칙이란?

**합성함수**란 여러 함수로 구성된 함수입니다.



![image-20201222180533442](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201222180533442.png)

>  합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.



이것이 연쇄법칙의 원리입니다.



#### 2.3 연쇄법칙과 계산 그래프



![image-20201223150758546](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223150758546.png)

- 2 제곱 계산을 **2 노드로 나타내면 위처럼 그릴 수 있습니다.

- 그림과 같이 역전파의 계산 절차에서는 노드로 들어온 입력 신호에 그 노드의 국소적 미분을 곱한 후 다음 노드로 전달합니다.

- 맨 왼쪽 계산을 보시면 연쇄법칙을 적용하여 x에 대한 z의 미분이 됩니다. 즉, 역전파가  하는 일은 연쇄 법칙의 원리와 같다는 것입니다.

  

### 3. 역전파

덧셈과 곱셈을 통해서 역전파를 알아봅시다.



#### 3.1 덧셈 노드의 역전파



z = x + y라는 식을 대상으로 그 역전파를 살펴봅시다.



![image-20201223151358460](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223151358460.png)

이렇게 해석적으로 계산할 수 있습니다. 그래프를 그려보면 아래와 같습니다.



![image-20201223151425246](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223151425246.png)



덧셈 노드의 역전파는 1을 곱하기만 할 뿐이므로 입력된 값을 그대로 다음 노드로 보내게 됩니다.



![image-20201223151515205](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223151515205.png)

결국 이렇게 입력력 신호를 다음 노드로 출력할 뿐이므로 그림처럼 그대로 1.3을 다음 노드로 전달합니다.





#### 3.2 곱셈 노드의 역전파



이어서 z = xy라는 식을 미분해봅시다.

![image-20201223151621268](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223151621268.png)

계산그래프는 아래와 같습니다.



![image-20201223151638947](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223151638947.png)

곱셈 노드 역전파는 상류의 값에 순전파 때의 입력 신호들을 '서로 바꾼 값'을 곱해서 하류로 보냅니다.

서로 바꾼 값이란 순전파 때 x였다면 역전파에서는 y, 순전파 때 y였다면 역전파에서는 x로 바꾼다는 의미 입니다.

![image-20201223152248000](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223152248000.png)

이렇게 입력 신호를 바꾼 값을 곱하여 하나는 1.3 * 5 = 6.5, 1.3 * 10 = 13이 됩니다.

곱셈의 역전파는 순방향 입력 신호의 값이 필요합니다. 그래서 곱셈 노드를 구현할 때는 순전파의 입력 신호를 변수에 저장해 둡니다.



#### 3.3 단순한 계층 구현하기



곱셈 노드를 'MulLayer' 덧셈 노드를 'AddLayer'라는 이름으로 구현합니다.



##### 3.3.1 곱셈 계층

모든 계층은 forward()와 backward()라는 공통의 메서드를 갖도록 구현할 것 입니다. forward()는 순전파, backward()는 역전파를 처리합니다.



```python
# coding: utf-8


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

```



MulLayer를 사용하여 순전파를 다음과 같이 구현할 수 있습니다.

```python
# coding: utf-8
from seung.layer_naive import *


apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)

>>>
price: 220
dApple: 2.2
dApple_num: 110
dTax: 200
```

backward() 호출 순서는 forward() 때와는 반대 입니다. 또, backward()가 받는 인수는 '순전파의 출력에 대한 미분'임에 주의하세요. 가령 mul_apple_layer라는 곱셈 계층은 순전파 때는 apple_price를 출력합니다만, 역전파 때는 apple_price의 미분 값인 dapple_price를 인수로 받습니다.



##### 3.3.2 덧셈 계층



이어서 덧셈 계층을 구현합니다.

```python
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

```



덧셈 계층에서는 초기화가 필요 없으니 __init__()에서는 아무 일도 하지 않습니다.(pass가 가만히 있으라는 명령입니다.) 덧셈 계층의 forward()에서는 입력받은 두 인수 x,y를 더해서 반환합니다. backward()에서는 상류에서 내려온 미분을 그대로 하류로 흘립니다.

사과 귤을 코드로 나타내면 아래와 같습니다.



```python
# coding: utf-8
from layer_naive import *

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
price = mul_tax_layer.forward(all_price, tax)  # (4)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dOrange:", dorange)
print("dOrange_num:", int(dorange_num))
print("dTax:", dtax)

```

단순하게 보면 필요한 계층을 만들어 순전파 메서드인 forward()를 적절한 순서로 호출 합니다. 그런 다음 순전파와 반대 순서로 역전파 메서드인 backward()를 호출하면 원하는 미분이 나옵니다.



#### 3.4 활성화 함수 계층 구현하기



이제 계산 그래프를 신경망에 적용할 때가 왔습니다. 여기에서는 신경망을 구성하는 층(계층) 각각을 클래스 하나로 구현합니다.



##### 3.4.1 ReLU 계층

활성화 함수로 사용되는 ReLU의 수식과 미분은 다음과 같습니다.

![image-20201223155033455](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223155033455.png)

순전파 때의 입력인 x가 0보다 크면 역전파는 상류의 값을 그대로 하류로 흘립니다. 반면, 순전파 때 x가 0 이하면 역전파 때는 하류로 신호를 보내지 않습니다. 아래는 계산 그래프로 나타낸 것 입니다.

![image-20201223155150579](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223155150579.png)

이제 이 계층을 구현해 봅시다.



```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
```

Relu 클래스는 mask라는 인스턴스 변수를 가집니다. mask는 True/False로 구성된 넘파이 배열로, 순전파의 입력인 x의 원소 값이 0 이하인 인덱스는 True, 그 외는 False로 유지합니다.

```python
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
print(x)

>>>
[[ 1.  -0.5]
 [-2.   3. ]]

mask = (x <= 0)
print(mask)
>>>
[[False  True]
 [ True False]]
```

위와 같이 순전파 때의 입력 값이 0 이하면 역전파 때의 값은 0이 돼야 합니다. 그래서 역전파 때는 순전파 때 만들어둔 mask를 써서 mask의 원소가 True인 곳에는 상류에서 전파된 dout을 0으로 설정합니다.



##### 3.4.2 Sigmoid 계층

![image-20201223155939411](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223155939411.png)

시그모이드 함수는 다음 식을 의미합니다.

![image-20201223155959837](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223155959837.png)



위는 계산 그래프입니다.

이번엔 exp와 / 노드가 새롭게 등장했습니다. exp노드는 y = exp(x) 계산을 수행하고 / 노드는 y = 1/x 계산을 수행합니다.

위와같이 국소적 계산의 전파로 이뤄집니다. 한 단계씩 살펴보죠



**1단계**

/노드, 즉 y = 1/x를 미분하면 다음식이 됩니다.

![image-20201223160216315](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223160216315.png)

역전파 대는 상류에서 흘러온 값에 -y**(순전파의 출력을 제곱한 후 -를 붙임)을 곱해서 하류로 전달합니다. 계산그래프는 아래와 같습니다.

![image-20201223160328270](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223160328270.png)



**2단계**

+노드는 상류의 값을 여과 없이하루로 내보내는게 다입니다. 

![image-20201223160423581](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223160423581.png)



**3단계**

exp노드는 y = exp(x) 연산을 수행하며, 그 미분은 다음과 같습니다.



![image-20201223160454854](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223160454854.png)

계산 그래프에서는 상류의 값에 순전파 때의 출력( 이 예에서는 exp(-x))을 곱해 하류로 전파합니다.

![image-20201223160520255](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223160520255.png)



**4단계**

x 노드는 순전파 때의 값을 '서로 바꿔' 곱합니다. 이 예에서는 -1을 곱합니다.

![image-20201223161302924](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223161302924.png)

역전파의 최종 출력인 값이 하류 노드로 전파됩니다. 여기에서 최종값이 순전파의 입력 x와 출력 y만으로 계산할 수 있다는 것을 확인하셨나요? 이 중간 과정을 모두 묶어 아래처럼 노드하나로 대체할 수 있습니다.

![image-20201223161444564](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223161444564.png)

이렇게 중간 계산들을 생략할 수 있어 더욱 효율적인 계산이죠. 또 노드를 그룹화하여 계층의 세세한 내용을 노출하지 않고 **입/출력에만 집중할 수 있다**는 것이 포인트 입니다.

정리하면 아래와 같습니다.

![image-20201223161623778](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223161623778.png)

이처럼 sigmoid 계층의 역전파는 순전파의 출력(y)만으로 계산할 수 있습니다.

```python
class Sigmoid:
    def __init__(self):     # 초기화
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
        
# 순전파의 출력을 인스턴스 변수 out에 보관했다가, 역전파 계산 때 그 값을 사용합니다.
    def backword(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx
```



#### 3.5 Affine/Softmax 계층 구현하기



##### 3.5.1 Affine 계층

신경망의 순전파 때 수행하는 행렬의 곱은 기하학에서는 **어파인 변환**(affine transformation)이라고 합니다. 그래서 이 책에서는 어파인 변환을 수행하는 처리를 Affine 계층이라는 이름으로 구현합니다.



**신경망 순전파 복습**

신경망의 순전파에 가중치 신호의 총합을 계산하기 때문에 행렬의 곱(np.dot)을 사용했습니다. 그러면 뉴런의 가중치 합은 Y = np.dot(X, W) + B처럼 계산합니다. 그리고 이 Y를 활성화 함수로 변환해 다음 층으로 전파하는 것이 신경망 순전파의 흐름이었습니다. 행렬의 곱 계산은 대응하는 **차원의 원소 수를 일치**시키는 게 핵심입니다.



![image-20201223163057450](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223163057450.png)

X, W, B가 행렬이라는 점에 주의하세요. 지금까지 계산에는 노드 사이에 스칼라값이 흘렀지만 이 예에서는 행렬이 흐르고 있습니다.

역전파에 대해 생각하며 전개하면 아래와 같습니다.

![image-20201223163222380](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223163222380.png)

Wt는 전치행렬을 뜻하는데 W의 (i,j) 위치의 원소를 (j,i)로 바꾼 것을 뜻합니다.

![image-20201223163500907](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223163500907.png)

W의 형상이 (2,3)이었다면 전치 행렬 Wt의 형상은 (3,2)가 됩니다.

위 식을 바탕으로 계산 그래프의 역전파를 구해봅시다.

![image-20201223163638444](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223163638444.png)

계산 그래프에서는 각 변수의 형상에 주의해서 살펴봅시다. 

![image-20201223164046761](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223164046761.png)

행렬의 형상에 주의하는 이유는 행렬의 곱에서는 대응하는 차원의 원소 수를 일치시켜야하는데 이를 위해서는 위의 식을 동원해야 할 수도 있기 때문입니다.

![image-20201223164112679](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223164112679.png)

##### 3.5.2 배치용 Affine 계층



지금까지 설명한 Affine 계층은 이볅 데이터로 X하나만 고려한 것이었습니다. 이번 절에서는 데이터 N개를 묶어 순전파하는 경우, 즉 배치용 Affine 계층을 생각해보겠습니다.(묶은 데이터를 배치라고 부릅니다.)



![image-20201223164309030](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223164309030.png)

기존과 다른 부분은 입력인 X의 형상이 (N, 2)가 된 것  뿐입니다. 편향을 더할 때도 주의해야합니다. 순전파 때의 편향 덧셈은 X * W에 대한 편향이 각 데이터에 더해집니다.

이제 Affine 구현을 해봅시다.

```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx

```



##### 3.5.3 Softmax-with-Loss 계층

마지막으로 출력층에서 사용하는 소프트맥스 함수에 관해 설명하겠습니다. 앞에서 말했듯 이 소프트맥스 함수는 입력 값을 정규화 하여 출력합니다. 예를 들어 손글시 숫자 인식에서의 softmax 계층의 출력은 아래와 같습니다.

![image-20201223164827416](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223164827416.png)

그림과 같이 softmax 계층은 입력 값을 정규화(출력의 합이 1이 되도록 변형)하여 출력합니다. 



이제 소프트맥스 계층을 구현할 텐데, 손실 함수인 교차 엔트로피 오차도 포함하여 'Softmax-with-Loss 계층'이라는 이름으로 구현합니다.

그래프는 아래와 같습니다.

![image-20201223165149393](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223165149393.png)

좀 복잡하니 간소화하면 

![image-20201223165226826](C:\Users\sjn60\AppData\Roaming\Typora\typora-user-images\image-20201223165226826.png)



여기에서 주목할 것은 역전파의 결과입니다. 신경망의 **역전파에서는 이 차이인 오차가 앞 계층에 전해지는 것** 입니다. 이는 신경망 학습의 **중요한 성질**입니다.

 

 신경망 학습의 목적은 신경망의 출력(Softmax의 출력)이 **정답 레이블과 가까워지도록 가중치 매개변수의 값을 조정**하는 것이었습니다. 그래서 신경망의 출력과 정답 레이블의 오차를 효율적으로 앞 계층에 전달해야 합니다. 앞의 (y1−t1, y2−t2, y3−t3)라는 결과는 신경망의 **현재 출력과 정답 레이블의 오차를 있는 그대로** 나타내는 것입니다.

 

> **'소프트맥스 함수'의 손실 함수로 '교차 엔트로피 오차'를 사용하는 이유**
> 역전파의 결과가 (y1−t1, y2−t2, y3−t3)로 말끔히 떨어지기 때문입니다. 교차 엔트로피 오차라는 함수가 그렇게 설계되었기 때문입니다.
>
> **'항등 함수'의 손실 함수로 '오차 제곱합'을 사용하는 이유**
> 역전파의 결과가 (y1−t1, y2−t2, y3−t3)로 말끔히 떨어지기 때문입니다.

 정답 레이블이 (0, 1, 0)일 때 Softmax 계층이 (0.3, 0.2, 0.5)를 출력했다고 해봅시다. 이 경우 Softmax 계층의 역전파는 (0.3, -0.8, 0.5)라는 커다란 오차를 전파합니다. 결과적으로 Softmax 계층의 앞 계층들은 그 큰 오차로부터 큰 깨달음을 얻게 됩니다.

그럼 Softmax-with-Loss 계층을 구현한 코드를 보겠습니다.

```python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
```

#### 

#### 3.6 오차역전파법 구현하기



##### 3.6.1 신경망 학습의 전체 그림

다음은 신경망 학습의 순서이다.



**전제**

​	신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련 데이터에 적응하도록 조정하는 과정을 "학습"이라고 합니다. 신경망 학습은 다음과 같이 4단계로 수행합니다.



**1단계 - 미니배치**

​	훈련 데이터 중 일부를 무작위로 가져옵니다. 이렇게 선별한 데이터를 미니배치라고 하며, 그 미니배치의 손실함수 값을 줄이는 것이 목표입니다.



**2단계 - 기울기 산출**

​	미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구합니다. 기울기는 **손실 함수의 값을 가장 작게 하는 방향**을 제시합니다.



**3단계 - 매개변수 갱신**

​	가중치 매개변수를 기울기 방향으로 아주 조금 갱신합니다.



**4단계 - 반복**

​	1 ~ 3단계를 반복합니다.



지금까지 설명한 오차역전파법이 등장하는 단계는 두 번째인 기울기 산출입니다.

오차역전파법을 이용하면 느린 수치 미분과 달리 기울기를 효율적이고 빠르게 구할 수 있습니다.



##### 3.6.2 오차역전파법을 적용한 신경망 구현

```python
import sys, os
sys.path.append(os.pardir) # 부모디렉토리를 가져옴
import numpy as np
from seung.layers import *
from seung.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {} # params 딕셔너리 변수로, 신경망의 매개변수를 보관. 
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict() # layers 순서가 있는 딕셔너리 변수로, 신경망의 계층을 보관
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss() # 신경망의 마지막 계층
        
    def predict(self, x): # 예측(추론)을 수행한다.
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t): # 손실함수의 값을 구한다. 인수 x는 이미지 데이터, t는 정답 레이블 
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t): # 정확도를 구한다.
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t): #가중치 매개변수의 기울기를 수치 미분 방식으로 구한다.
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t): # 가중치 매개변수의 기울기를 오차역전파법으로 구한다.
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

```



이 구현에서는 신경망의 계층을 OrderedDict에 보관하는 점이 중요합니다. OrderedDict은 순서가 있는 딕셔너리입니다. 그래서 순전파 때는 추가한 순서대로 각 계층의 forward() 메서드를 호출하기만 하면 처리가 완료됩니다. 마찬가지로 역전파 때는 계층을 반대 순서로 호출하면 됩니다. Affine 계층과 ReLU 계층이 각자의 내부에서 순전파와 역전파를 제대로 처리하고 있으니 그냥 올바른 순서대로 연결한 다음 순서대로 호출해주면 끝입니다.



##### 3.6.3 오차역전파법으로 구한 기울기 검증



기울기 구하는 방법

- 수치 미분
- 해석적으로 수식을 풀어 구하기(오차역전파법)

해석적으로 푸는 방법이 훨씬 효율적.

수치미분은 오차역전파법을 정확히 구현했는지 확인하는 역할을 할 수 있습니다.

수치 미분의 이점은 구현하기 쉬워서 버그가 숨어 있기 어렵지만, 오차역전파법은 구현하기 복잡해서 종종 실수가 생깁니다. 그래서 수치 미분의 결과와 오차역전파법의 결과를 비교하여 오차역전파법을 제대로 구현했는지 검증하곤 합니다. 



이처럼 두 방식으로 구한 기울기가 일치함을 확인하는 작업을 **기울기 확인**이라고 합니다.

```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from seung.mnist import load_mnist
from seung.two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))

>>>
W1:4.934589503036241e-10
b1:2.6014773364684993e-09
W2:7.187884432268386e-09
b2:1.3969405174424355e-07
```



MNIST 데이터셋을 읽습니다. 훈련 데이터 일부를 수치 미분으로 구한 기울기와 오차역전파법으로 구한 기울기의 오차를 확인합니다. 여기에서는 각 가중치 매개변수의 차이의 절댓값을 구하고, 이를 평균한 값이 오차가 됩니다.



> 수치 미분과 오차역전파법의 결과 오차가 0이 되는 일은 드뭅니다. 컴퓨터가 할 수 있는 계산의 정밀도가 유한하기 때문입니다. 만약 나온다면 잘못 구현했다고 의심해야합니다!



##### 3.6.4 오차역전파법을 사용한 학습 구현하기

마지막으로 오차역전파법을 사용한 신경망 학습을 구현합니다.

지금까지와 다른 부분은 기울기를 오차역전파법으로 구한다는 점뿐입니다.

```python
# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from seung.mnist import load_mnist
from seung.two_layer_net import TwoLayerNet

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

train acc, test acc | 0.0914, 0.0894 train acc, test acc | 0.9020833333333333, 0.9057 train acc, test acc | 0.9227833333333333, 0.9248 train acc, test acc | 0.9354833333333333, 0.9348 train acc, test acc | 0.94495, 0.9438 train acc, test acc | 0.9521333333333334, 0.9492 train acc, test acc | 0.9583666666666667, 0.9545 train acc, test acc | 0.9630666666666666, 0.9585 train acc, test acc | 0.9664166666666667, 0.9614 train acc, test acc | 0.96875, 0.9643 train acc, test acc | 0.97075, 0.9643 train acc, test acc | 0.97275, 0.9675 train acc, test acc | 0.9752333333333333, 0.969 train acc, test acc | 0.97635, 0.9681 train acc, test acc | 0.9785, 0.9703 train acc, test acc | 0.9782333333333333, 0.9698 train acc, test acc | 0.9804, 0.9722



![img](data:image/png;base64,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)

### 4. 정리



- 이번 장에서는 계산과정을 시각적으로 보여주는 방법인 계산 그래프를 배웠습니다. 

- 계산 그래프를 이용하여 신경망의 동작과 오차역전파법을 설명하고, 그 처리 과정을 계층이라는 단위로 구현했습니다. 
  - 예를 들어 ReLU 계층, Softmax-with-Loss 계층, Affine 계층, Softmax 계층 등입니다. 모든 게층에서 forward와 bakcward라는 메서드를 구현합니다. 
  - 전자는 순전파, 후자는 역전파로 가중치 매개변수의 기울기를 효율적으로 구할 수 있습니다. 

- 이처럼 동작을 모듈화한 덕분에, 신경망의 계층을 자유롭게 조합하여 원하는 신경망을 만들 수 있습니다.



**배운 내용**

- 계산그래프를 이용하여 계산과정을 시각적으로 파악가능
- 게산 그래프의 노드는 국소적 계산으로 구성된다. 국소적 계산을 조합해 전체 계산을 완성한다.
- 계산 그래프의 순전파는 통상의 계산을 수행한다. 한편, 계산 그래프의 역전파로는 각 노드의 미분을 구할 수 있다.
- 신경망의 구성 요소를 계층으로 구현하여 기울기를 효율적으로 계산할 수 있다.(오차역전파법)
- 수치 미분과 오차역전파법의 결과를 비교하면 오차역전파법의 구현에 잘못이 있는지 없는지 확인할 수 있다(기울기 확인)