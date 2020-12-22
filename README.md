# DFS_5

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
2.  그래프에서 계산을 왼쪽에서 오른쪽으로 진행한다.
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