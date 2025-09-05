# 학습 정리

## 1️⃣ 강의 복습 내용

- 이번주 동안 배운 내용 : PyTorch 기초, Tensor 생성과 조작, Tensor 연산 및 심화, 선형 회귀, 이진 분류
<br>
<br>
- **특히 궁금했던 부분들** : tensor에서 부동 소수점 계산 방법, stack()함수에서 생성되는 차원축에 대한 결합 방식의 차이, expand()와 repeat()의 동작방식 차이, sklearn에서 fit_transform()과 transform()의 차이


## 2️⃣ 과제 수행 과정 / 결과물 정리

- 과제 1,2,3 풀면서 든 생각은 확실히 **많은 양치기**가 텐서에 대한 학습곡선이 단축되는 것을 느꼈다.
- 과제 3번에서, boolean 비교문 쓰는거보고 충격. 이렇게 0과 1을 나눌수도 있구나.
- 과제 2,3번에서 Tensor가 GPU에 올라갈시, cpu에 있는 tensor와 연산이 안되는 것을 머리로만 이해했었으나, 직접 풀어보면서 손으로 깨달았다. 또한, 모델의 train과 eval과정에 대해서 다시 외울수 있을 정도로 학습해야함을 생각함(그동안 아는건 아는게 아니었다)
<br>
<br>
- 심화 과제 :**Cross-Entropy 구현에서 실패**, 주말동안 다시 찬찬히 훝어볼 예정
- 위클리 미션 : torch.isin, torch.clip 등 아직 다양한 torch함수들에 대해서 얕은 지식만을 가지고 있다. 다시 정확히 코드로 수정해보며, 리뷰할 예정

- **위클리 미션하면서 궁금 했던 것이 있었으나, 끝내 알지 못했던 것** : torch.tensor가, sep(list같은)내에서 텐서들이 있을 때, 만약 각 텐서들의 요소가 1개씩 있으면 에러가 뜨지 않지만, 2개 이상일 시, 아래와 같은 Value 에러 발생
  - 처음에는 부덕이가 torch.tensor내에서, seq안의 Tensor들이 있을 시, 자동으로 stack연산을 적용하여 묶어 준다고 했는데, 애초에 이 연산 자체가 stack에서 동작하지 않아 tensor로 돌린 것. 따라서 말의 앞뒤가 맞지 않아 다시 오류를 보고 생각해보니, list(tensor) -> list(Python scalars) -> tensor(tensor) 이런 동작 방식 인 것 같은데, 이거는 tensor 내의 seq를 받는 parameter인 data가 처리하는 'array-like' 내부로직에 있다는 생각이 든다. 아직 이 부분을 확인해보지 못해 따로 남긴다. (나중에 찾아볼 예정)
```python
# 과제하면서 궁금 했던 것이 있었으나, 끝내 알지 못했던 것
import torch

# ValueError: only one element tensors can be converted to Python scalars
a1 = torch.tensor([1,2], dtype=torch.float32) # 텐서shape(2,1) 한개의 요소를 가진 텐서들이 아니다
b1 = torch.tensor([2,3], dtype=torch.float32)

# ValueError: only one element tensors can be converted to Python scalars
a2 = torch.tensor([[1],[2]], dtype=torch.float32) # 텐서shape(1,2) 이거는 한 개의 요소가 아니다.
b2 = torch.tensor([[2],[3]], dtype=torch.float32)

# good
a3 = torch.tensor([[1]], dtype=torch.float32) # 텐서shape(1,1) 이거는 한 개의 요소
b3 = torch.tensor([[2]], dtype=torch.float32)

d = []
d.append(a3)
d.append(b3)
print(type(d))
print(d[0].dtype)
print(d[0].shape)
c= torch.tensor(d) # <- 1, 2는 요소가 여러개인 텐서들, 3은 요소가 한 개인 텐서들이 된다. 따라서 1 2는 안됨
print(c.dtype)
print(c[0].dtype)
# 근데 그러면 동작과정이 list(tensor) -> list(Python scalars) -> tensor(tensor)인건가?
```

## 3️⃣ 피어세션 정리

- 1일차(월) : 그라운드 룰 정하기
- 1일차(화) : 코딩 테스트 대비 알고리즘 스터디
- 1일차(수) : 코딩 테스트 대비 알고리즘 스터디
- 1일차(목) : 강의 리뷰 
- 1일차(금) : 위클리 미션 리뷰, 팀 회고록 정리

- 좋은 점 : 다들 착하고 열심히하신다. 팀운 Good.
- 아쉬운 점 : 조금 더 열심히 공부해서 얘기하고 싶은데 내 의지가 조금 부족하다, 다음주엔 조금 더 미리미리 정리해볼 것

- 주간 회고 내용 : 목요일에 core질문 리뷰 추가, 팀 회고록 작성


## 4️⃣ 학습 회고 - 5F

<!-- ## KPT

**KPT**(Keep, Problem, Try)는 이름에서 알 수 있듯 3가지 관점에서 업무를 돌아보고, 다음 액션 아이템을 도출해내는 데 도움이 되는 회고 템플릿이다.

**Keep** (프로젝트에서 만족했고, 앞으로의 업무에서 지속하고 싶은 부분)
**Problem** (프로젝트에서 부정적인 요소로 작용했거나 아쉬웠던 점)
**Try** (Problem에 대한 해결 방식으로 다음 프로젝트에서 시도해볼 점)
KPT에서 가장 중요한 부분은 **Try**이다. 이번주 아쉬웠던 점을 Try를 통해 어떻게 보완할 수 있을지 정리해보면서 구체적인 실천 방안을 세울 수 있다. -->



**5F**는 다음 다섯 개의 키워드에 따라 순서대로 회고를 진행하는 방식이며, 개인이 한 활동을 회고하는 데 유용하다. 어떤 일이 있었고 무엇을 느꼈는지를 시간 순서대로 정리하는 데 도움이 되는 방식이다.

- Fact (사실: 일어난 일에 대한 객관적인 기록)
- Feeling (느낌: 상황에 대한 감정적인 반응)
- Finding (배운 점: 경험을 통해 배울 수 있었던 것)
- Future action (향후 행동: 향후 할 수 있는 개선된 행동)
- Feedback (피드백: 앞서 정한 향후 행동을 실천해본 뒤, 받은 피드백)

### Fact
- 모든 강의 수강 완료, 심화과제 제외 과제 1,2,3 수행 완료, 위클리 미션 진행도 7/8
    - 강의 내용 : PyTorch 기초, Tensor 생성과 조작, Tensor 연산 및 심화, 선형 회귀, 이진 분류
- 피어세션 : 알고리즘 스터디 2회 완료, 강의 리뷰 완료, 위클리 리뷰 및 팀 회고록 작성

### Feeling & Finding & Future action
- 우선 생각보다 오전과 오후 시간에 집중이 잘 되지 않았다. 익숙한 공간에 있어서 그런지 몸이 쉽게 늘어지는 것 같다. 그러나 나가기에도 애매한 시간대여서 **우선 다음주까지는 더 집중한다는 생각으로 하기**
- **잠을 2시 전에는 자야한다.** 그래야 최소 7시간 반을 잘 수 있다. 이 부분이 제일 중요한 것 같다. 지금동안은 2-3시 사이로 자서, 많이 피곤했다.
- **호기심 기르기.** 내가 일찍 공부를 끝내야 시야가 넓어지고, 공부할 때도 여유가 생겨 궁금한 점이 더 생긴다. 또한, 남의 코드에서 궁금한게 생기면 꼭 물어보기

### Feedback
- 2주차부터 1주차 피드백 예정