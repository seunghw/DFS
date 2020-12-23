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
>  역전파의 결과가 (y1−t1, y2−t2, y3−t3)로 말끔히 떨어지기 때문입니다. 교차 엔트로피 오차라는 함수가 그렇게 설계되었기 때문입니다.
>
> **'항등 함수'의 손실 함수로 '오차 제곱합'을 사용하는 이유**
>  역전파의 결과가 (y1−t1, y2−t2, y3−t3)로 말끔히 떨어지기 때문입니다.

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

