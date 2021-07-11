# Dialogue State Tracking

- [Task](#Task)
- [Dataset](#dataset)
  - [Train dataset](#train-dataset)
    - [Preprocessing](#preprocessing)
    - [Relation classes](#relation-classes)
  - [Test dataset](#test-dataset)
- [Result](#result)
- [How to use](#how-to-use)
  - [Install requirements](#install-requirements)
  - [Train](#train)
  - [Inference](#inference)

---

## Task
목적 지향형 대화에서는 유저가 미리 정의된 시나리오 안에서 특정 목적을 수행하기 위해 대화를 진행한다는 가정을 하고 있습니다. 이 "미리 정의 된 시나리오"는 보통 특정 Knowledge Base (KB)의 Entity를 검색하거나 예약과 같은 새로운 Instance를 추가하는 행위로 나타날 수 있습니다. 이러한 검색/예약과 같은 태스크를 수행하기 위해 시스템은 유저의 목적(Goal)을 파악해야만 하고, 보통 이 Goal은 (Slot, Value) 페어의 집합으로 표현될 수 있습니다. Slot은 Goal을 수행하기 위해 파악해야 하는 정보의 타입이고, Value는 이 Slot에 속할 수 있는 값입니다. 예컨데, 숙소를 예약하는 시나리오의 경우 "숙소의 종류", "숙소의 가격대"가 Slot의 타입이 될 수 있고, 이에 속할 수 있는 Value로 각각 ("호텔", "모텔", "에어비앤비", ...), ("저렴", "적당", "비싼", ...) 등을 가질 수 있습니다.

대화 상태 추적(Dialogue State Tracking)은 목적 지향형 대화(Task-Oriented Dialogue)의 중요한 하위 테스크 중 하나입니다. 유저와의 대화에서 미리 시나리오에 의해 정의된 정보인 Slot과 매 턴마다 그에 속할 수 있는 Value의 집합인, 대화 상태 (Dialogue State)를 매 턴마다 추론하는 테스크입니다. 대화 상태는 아래 그림과 같이 미리 정의된 J(45)개의 Slot S마다 현재 턴까지 의도된 Value를 추론하여 (S, V)와 같은 페어의 집합(B)으로 표현될 수 있습니다. ( 이 때, 현재까지 의도되지 않은 정보(Slot)는 "none"이라는 특별한 Value를 가지게 되고, 아래 B에서 생략되어 있습니다.)

![DST](https://user-images.githubusercontent.com/77161691/125183603-c974c200-e252-11eb-9320-33b11d66a1e5.png)

모든 대화는 (시스템 발화, 유저 발화)를 하나의 턴으로 봅니다. 하나의 대화에서 모든 유저 발화마다 Dialogue State가 추론되어야 합니다. 예컨데 위 대화에서 두번째 턴의 인풋/아웃풋은 아래와 같습니다.

* input: ["안녕하세요.", "네. 안녕하세요. 무엇을 도와드릴까요?", "서울 중앙에 위치한 호텔을 찾고 있습니다. 외국인 친구도 함께 갈 예정이라서 원활하게 인터넷을 사용할 수 있는 곳이 었으면 좋겠어요."]

* output: ["숙소-지역-서울 중앙", "숙소-인터넷 가능-yes"]

본 프로젝트(2021.04.12 - 2021.04.25)에서는 유저와의 대화에서 미리 시나리오에 의해 정의된 정보인 `Slot`과 매 턴마다 그에 속할 수 있는 `Value`의 집합인 `Dialogue State(대화 상태)`를 매 턴마다 추론하는 모델을 학습시켰습니다. 그리고 시나리오에 대해 매 턴마다 알맞은 Dialogue State를 예측했습니다.

Base model로는 [TRADE](https://github.com/jasonwu0731/trade-dst), [SUMBT](https://github.com/SKTBrain/SUMBT) 를 사용했습니다.

## Dataset
### Train dataset
9000개의 `tsv` 문장 데이터 사용

![Train dataset](https://user-images.githubusercontent.com/77161691/115668486-1fd41200-a382-11eb-950e-ad1d1340f769.png)

* column 1: `데이터가 수집된 정보`
* column 2: `sentence`
* column 3: `entity 1`
* column 4: `entity 1의 시작 지점`
* column 5: `entity 1의 끝 지점`
* column 6: `entity 2`
* column 7: `entity 2의 시작 지점`
* column 8: `entity 2의 끝 지점`
* column 9: `relation`

#### Preprocessing
Raw data의 `sentence`, `entity1`, `entity2`, `relation` column 사용 (`load_data.py`)

#### Relation classes
42개의 class 설정
```
with open('./label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)

{'관계_없음': 0, '인물:배우자': 1, '인물:직업/직함': 2, '단체:모회사': 3, '인물:소속단체': 4, '인물:동료': 5, '단체:별칭': 6, '인물:출신성분/국적': 7, '인물:부모님': 8, '단체:본사_국가': 9,
'단체:구성원': 10, '인물:기타_친족': 11, '단체:창립자': 12, '단체:주주': 13, '인물:사망_일시': 14, '단체:상위_단체': 15, '단체:본사_주(도)': 16, '단체:제작': 17, '인물:사망_원인': 18, '인물:출생_도시': 19,
'단체:본사_도시': 20, '인물:자녀': 21, '인물:제작': 22, '단체:하위_단체': 23, '인물:별칭': 24, '인물:형제/자매/남매': 25, '인물:출생_국가': 26, '인물:출생_일시': 27, '단체:구성원_수': 28, '단체:자회사': 29,
'인물:거주_주(도)': 30, '단체:해산일': 31, '인물:거주_도시': 32, '단체:창립일': 33, '인물:종교': 34, '인물:거주_국가': 35, '인물:용의자': 36, '인물:사망_도시': 37, '단체:정치/종교성향': 38, '인물:학교': 39,
'인물:사망_국가': 40, '인물:나이': 41} 
```

### Test dataset
1000개의 `tsv` 문장 데이터 사용

![Test dataset](https://user-images.githubusercontent.com/77161691/115676003-1f3f7980-a38a-11eb-9f3d-23b772b19fd2.png)

* column 1-8: train dataset과 동일
* column 9: 'blind' relation -> will be predicted

## Training Details
| Model | Batch Size | epochs | LR | Train Time |
| :--- | ---: | ---: | ---: | ---: |
| `XLM-Roberta-large` | 18 | 10 | 3e-5 | 34h |
| `XLM-Roberta-base` | 18 | 6 | 1e-5 | 12h |
| `BERT-base-multilingual-cased`| 18 | 6 | 5e-5 | 20h |

- **P40**을 이용하여 학습하였습니다.

## Result
| Model | Accuracy |             
| --- | ---: |
| [XLM-Roberta-large](https://huggingface.co/xlm-roberta-large) | 79.50 |
| [XLM-Roberta-base](https://huggingface.co/xlm-roberta-base) | 60.10 |
| [BERT-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)| 71.70 |

## How to use
### Install requirements
```
pip install -r requirements.txt
```

### Train
```
python train.py
```
Please refer to `train.py` for train arguments. (Optional)

### Inference
```
python inference.py
```
Please refer to `inference.py` for inference arguments. (Optional)

### Evaluation
```
python evaluation.py
```
Please refer to `evaluation.py` for evaluation arguments. (Optional)

---
#### Docs
- https://github.com/SKTBrain/KoBERT#using-with-pytorch
- https://github.com/monologg/KoBERT-Transformers
- https://github.com/monologg/KoBERT-NER
- https://github.com/eagle705/pytorch-bert-crf-ner

#### License
CC-BY-SA
