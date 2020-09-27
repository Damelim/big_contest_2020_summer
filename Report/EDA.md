```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```

# 1. 데이터 불러오기

- 개인투수, 개인타자, 팀투수, 팀타자, 팀 데이터만 사용

### 개인투수, 개인타자의 성적을 예측하여 이를 aggregate하는 것을 팀 투수, 타자 성적을 바로 예측하는 것보다 중요시하였다

#### 개인별 접근을 우선시 한 이유 : 팀 성과, 성적은 연도마다 변동이 비교적 심하다 + 팀별 데이터를 쓰면 row의 손실 (즉, 정보의 손실)이 매우 크다 

#### 개인별 접근이 가능하다 판단한 이유 : 야구는 여타 스포츠보다 동료의 성적이 개인 성적에 미치는 영향이 작은 스포츠이다!

[2], [3]번 코드는 개인투수, 개인타자 시트의 데이터를 연도순으로 불러오는 코드이다


```python
# 개인투수
private_pitcher_2016 = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2016.csv', engine='python', encoding="CP949")
private_pitcher_2017 = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2017.csv', engine='python', encoding="CP949")
private_pitcher_2018 = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2018.csv', engine='python', encoding="CP949")
private_pitcher_2019 = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2019.csv', engine='python', encoding="CP949")
private_pitcher_2020 = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_개인투수_2020.csv', engine='python', encoding="CP949")
```


```python
# 개인타자
private_batter_2016 = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2016.csv', engine='python', encoding="CP949")
private_batter_2017 = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2017.csv', engine='python', encoding="CP949")
private_batter_2018 = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2018.csv', engine='python', encoding="CP949")
private_batter_2019 = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2019.csv', engine='python', encoding="CP949")
private_batter_2020 = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_개인타자_2020.csv', engine='python', encoding="CP949")
```

[4]는 연도별로 흩어져 있는 데이터프레임을 투수, 타자 각각 합산하는 과정이다.

투수, 타자 모두 연도별로 변수명, 형태가 동일하므로 merge에서 충돌은 없다


```python
private_pitcher = pd.concat([private_pitcher_2016, private_pitcher_2017, private_pitcher_2018, private_pitcher_2019, private_pitcher_2020], ignore_index=True)
private_batter = pd.concat([private_batter_2016, private_batter_2017, private_batter_2018, private_batter_2019, private_batter_2020], ignore_index=True)
```

개인별 정보는 선수별 타율, 방어율을 예측하는 데 직접적으로 쓰인다.

그러나 팀 정보도 초기에 배제하지 않았다.

팀 정보는 방어율, 타율을 예측하는데 직접 쓰지는 않지만 승률 계산 (모델링 과정에서 피타고리안 승률 : 팀 실점, 득점으로 보정 승률을 구한다)에 직접적으로 쓰이게 된다.

팀 실점, 득점은 연도마다의 변동이 매우 심하다 (순위 변동이 심한 것처럼). 그러므로 관심 시기의 승률을 예측할 때 굳이 2016~19 데이터를 써서 SMOOTHING 효과를 야기하는 것보다 2020년 데이터를 쓰는 것이 현명하다.

[5],[6]은 팀투수, 팀타자 데이터를 불러오는 과정이다. 




```python
# 팀투수
team_pitcher = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_팀투수_2020.csv', engine='python', encoding="CP949")
```


```python
# 팀타자
team_batter = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_팀타자_2020.csv', engine='python', encoding="CP949")
```

팀들의 기본적인 데이터는 reference만으로 쓴다 16~부터 10구단 체제가 계속되었으므로 2020년만 가져와도 문제가 없다. 

넥센 -> 키움으로 한글 팀명이 변경되었지만 코드는 그대로이다.

[7]은 팀 정보를 가져오는 방식이다


```python
# 팀
team = pd.read_csv('data/2020빅콘테스트_스포츠투아이_제공데이터_팀_2020.csv', engine='python', encoding="CP949")
```

# 2. NA 확인

[8]은 개인투수 데이터의 NA값을 확인하는 과정이다. 모든 컬럼에 NA가 없다


```python
print(private_pitcher.shape)
print(private_pitcher.isna().sum())
print(np.sum(private_pitcher=="?", axis=1).value_counts())
```

    (27804, 38)
    G_ID          0
    GDAY_DS       0
    T_ID          0
    VS_T_ID       0
    HEADER_NO     0
    TB_SC         0
    P_ID          0
    START_CK      0
    RELIEF_CK     0
    CG_CK         0
    QUIT_CK       0
    WLS           0
    HOLD          0
    INN2          0
    BF            0
    PA            0
    AB            0
    HIT           0
    H2            0
    H3            0
    HR            0
    SB            0
    CS            0
    SH            0
    SF            0
    BB            0
    IB            0
    HP            0
    KK            0
    GD            0
    WP            0
    BK            0
    ERR           0
    R             0
    ER            0
    P_WHIP_RT     0
    P2_WHIP_RT    0
    CB_WHIP_RT    0
    dtype: int64
    0    27804
    dtype: int64
    

[9]는 개인타자의 NA값들 확인 과정이다. 역시 NA가 모든 열에서 없다


```python
print(private_batter.shape)
print(private_batter.isna().sum())
print(np.sum(private_batter=="?", axis=1).value_counts())
```

    (81102, 31)
    G_ID            0
    GDAY_DS         0
    T_ID            0
    VS_T_ID         0
    HEADER_NO       0
    TB_SC           0
    P_ID            0
    START_CK        0
    BAT_ORDER_NO    0
    PA              0
    AB              0
    RBI             0
    RUN             0
    HIT             0
    H2              0
    H3              0
    HR              0
    SB              0
    CS              0
    SH              0
    SF              0
    BB              0
    IB              0
    HP              0
    KK              0
    GD              0
    ERR             0
    LOB             0
    P_HRA_RT        0
    P_AB_CN         0
    P_HIT_CN        0
    dtype: int64
    0    81102
    dtype: int64
    

마찬가지로 [10],[11]에서 팀 투수 팀 타자 모두 NA값이 없음을 알 수 있다.


```python
print(team_pitcher.shape)
print(team_pitcher.isna().sum())
print(np.sum(team_pitcher=="?", axis=1).value_counts())
```

    (640, 34)
    G_ID          0
    GDAY_DS       0
    T_ID          0
    VS_T_ID       0
    HEADER_NO     0
    TB_SC         0
    CG_CK         0
    WLS           0
    HOLD          0
    INN2          0
    BF            0
    PA            0
    AB            0
    HIT           0
    H2            0
    H3            0
    HR            0
    SB            0
    CS            0
    SH            0
    SF            0
    BB            0
    IB            0
    HP            0
    KK            0
    GD            0
    WP            0
    BK            0
    ERR           0
    R             0
    ER            0
    P_WHIP_RT     0
    P2_WHIP_RT    0
    CB_WHIP_RT    0
    dtype: int64
    0    640
    dtype: int64
    


```python
print(team_batter.shape)
print(team_batter.isna().sum())
print(np.sum(team_batter=="?", axis=1).value_counts())
```

    (640, 28)
    G_ID         0
    GDAY_DS      0
    T_ID         0
    VS_T_ID      0
    HEADER_NO    0
    TB_SC        0
    PA           0
    AB           0
    RBI          0
    RUN          0
    HIT          0
    H2           0
    H3           0
    HR           0
    SB           0
    CS           0
    SH           0
    SF           0
    BB           0
    IB           0
    HP           0
    KK           0
    GD           0
    ERR          0
    LOB          0
    P_HRA_RT     0
    P_AB_CN      0
    P_HIT_CN     0
    dtype: int64
    0    640
    dtype: int64
    

# 3. 새 변수 생성

예측에 사용할 독립변수 후보들 (출루율, 도루 관련, 타석 수 등등), 혹은 구하고자 하는 타율, 방어율 등의 변수를 계산 , 추가하는 과정이다

### 1) 타율(AVG):  HIT / AB
- 안타 / 타수


```python
private_batter['AVG'] = private_batter['HIT'] / private_batter['AB']
private_batter['AVG'] = private_batter['AVG'].fillna(0)
```

### 2) 방어율(ERA): (ER * 9) / (INN2 / 3)

- (총 자책점 * 9 ) / 총 던진 이닝수 

INN2에 3을 나누어야 이닝이 나옴을 감안하였다


```python
private_pitcher['ERA'] = (private_pitcher['ER']*9) / (private_pitcher['INN2']/3)
private_pitcher['ERA'] = private_pitcher['ERA'].fillna(0)
```

### 3) 도루 시도 횟수(SB_trial): SB + CS

도루 성공 횟수 (SB), 실패 횟수 (CS)만 나왔는데 이 두 컬럼들보다 도루 시도 횟수, 도루 성공율의 정보를 활용하는 것이 합리적이라 판단

[14],[15]는 그 계산과정


```python
private_batter['SB_trial'] =  private_batter['SB'] + private_batter['CS']
```

### 4) 도루 성공율(SB_SR) : SB / (SB + CS)


```python
private_pitcher['SB_SR'] = private_pitcher['SB'] / (private_pitcher['SB']+private_pitcher['CS'])
private_pitcher['SB_SR'] = private_pitcher['SB_SR'].fillna(0)
```

### 5) 타석 수 - 타수(PA - AB)

: 단순히 타석 수가 아니라 투수 허용 사사구 수 (BB, IB, HP 모두)의 정보를 담은 변수 추가


```python
private_pitcher['PA-AB'] = private_pitcher['PA'] - private_pitcher['AB']
private_batter['PA-AB'] = private_batter['PA'] - private_batter['AB']
```

### 6) 희생타 + 희생플라이(SH + SF)

희생타, 희생플라이는 공통적으로 자신을 희생하여 동료를 진루시킨다. 어떤 방식으로 진루시키는지는 무관하다 판단


```python
private_batter['SH+SF'] = private_batter['SH'] + private_batter['SF']
```

### 7) BABIP
- 인플레이 타구(방망이에 공이 맞았을 때)의 안타 비율(타자, 투수 모두에게 사용 가능한 지표)

- 배팅 실력을 시사하는 2차지표

[18] : 투수

[19] : 타자


```python
private_pitcher['BABIP'] = (private_pitcher['HIT']-private_pitcher['HR']) / (private_pitcher['AB']-private_pitcher['KK']-private_pitcher['HR']+private_pitcher['SF'])
private_pitcher['BABIP'] = private_pitcher['BABIP'].fillna(0)
```


```python
private_batter['BABIP'] = (private_batter['HIT']-private_batter['HR']) / (private_batter['AB']-private_batter['KK']-private_batter['HR']+private_batter['SF'])
private_batter['BABIP'] = private_batter['BABIP'].fillna(0)
```

### 8) KK9

- 9이닝 당 삼진개수. 원래 삼진은 이닝과 관련성이 짙으므로 던진 삼진 능력 파악하기 위해 추가


```python
private_pitcher['KK9'] = (private_pitcher['KK'] / private_pitcher['INN2']) * 27
private_pitcher['KK9'] = private_pitcher['KK9'].fillna(0)
```

### 9) BB9
- 9이닝 당 볼넷 개수 , 역시 볼넷은 이닝과 관련성이 짙으므로 이닝과 관계 없는 제구력 파악


```python
private_pitcher['BB9'] = ((private_pitcher['BB']+private_pitcher['HP']) / private_pitcher['INN2']) * 27
private_pitcher['BB9'] = private_pitcher['BB9'].fillna(0)
```

### 10) 장타 허용(SLG): H2 + H3 + HR


```python
private_pitcher['SLG'] = private_pitcher['H2'] + private_pitcher['H3'] + private_pitcher['HR']
```

### 11) 단타 허용(H1)


```python
private_pitcher['H1'] = private_pitcher['HIT'] - private_pitcher['SLG']
```

### 계산과정 도중 분모가 0인 계산이 필수불가결하게 발생. 이들은 0으로 대체하였다


```python
private_pitcher = private_pitcher.replace(np.inf, 0)
private_batter = private_batter.replace(np.inf, 0)
```

# 4. Feature Selecting 

### 개인 투수 변수 1차 제거

#### 지금까지는 변수 추가만 하였으므로 쓰지 않기로 한 변수들을 불포함한다.

- PA 대신 (PA – AB) 사용: 단순히 타석 수보다 투수 허용 사사구 수 (BB, IB, HP 총합) 이 연관성 있음
- SB, CS 대신 SB_trial (도루 시도 횟수),  SB_SR (도루 성공률) 사용한다.


```python
private_pitcher = private_pitcher[['GDAY_DS', 'T_ID', 'P_ID', 'TB_SC', 'INN2', 'BF', 'PA-AB', 'AB', 'HIT', 'H1', 'H2', 'H3', 'HR', 'SB_SR', 'KK', 'WP', 'ER', 'ERA', 'SLG', 'BABIP', 'KK9', 'BB9']]
```


```python
private_pitcher.columns
```




    Index(['GDAY_DS', 'T_ID', 'P_ID', 'TB_SC', 'INN2', 'BF', 'PA-AB', 'AB', 'HIT',
           'H1', 'H2', 'H3', 'HR', 'SB_SR', 'KK', 'WP', 'ER', 'ERA', 'SLG',
           'BABIP', 'KK9', 'BB9'],
          dtype='object')



### 개인 타자 변수 1차 제거

#### 타자도 역시 변수 추가만 하였으므로 쓰지 않는 변수 불포함한다.


- H1, H2, H3, H4 등 타율의 분자, 즉 안타의 구성요소를 제거: 장타율을 예측하는 게 아니기 때문이다
- SB, CS 대신 SB_trial(도루 시도 횟수) 사용
- BB, IB, HP 대신 이들을 모두 고려하는 (PA – AB) 사용한다.
- SH, SF의 차이는 중요하지 않다! 자신의 희생으로 동료 선수의 진루를 돕는 것: SH + SF 변수를 사용.


```python
private_batter = private_batter[['GDAY_DS', 'T_ID', 'P_ID', 'TB_SC', 'PA-AB', 'AB', 'RUN', 'RBI', 'HIT', 'SH+SF', 'KK', 'AVG', 'SB_trial', 'BABIP']]
```


```python
private_batter.columns
```




    Index(['GDAY_DS', 'T_ID', 'P_ID', 'TB_SC', 'PA-AB', 'AB', 'RUN', 'RBI', 'HIT',
           'SH+SF', 'KK', 'AVG', 'SB_trial', 'BABIP'],
          dtype='object')



### X 변수 간 correlation 확인

#### 개인 투수, 개인 타자 모두 변수 제거 및 추가를 두 단계로 시행한다.

#### 첫째 과정이 [28]까지 배경지식을 사용한 과정이라면

#### 둘째 과정은 X변수들끼리 상관관계가 큰 변수들을 조정하는 방식이다.



[29]는 개인투수 변수들의 상관관계 heatmap이다. 이를 보니 (선형) 상관성이 큰 변수들이 존재하여 조정이 필요하다.

SLG(장타), KK(단순 삼진 개수), HIT(타자 데이터에서는 안타이나 투수 데이터에서는 허용 안타이다), AB(타수), BF(투구 수) 제거


```python
temp = private_pitcher.drop(['GDAY_DS', 'ER', 'ERA'], axis=1)
figure,ax1 = plt.subplots()
figure.set_size_inches(10,10)

sns.heatmap(temp.corr(),annot=True,cmap='YlGnBu')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x258b60ce438>




![png](output_53_1.png)


 - SLG(장타), KK(삼진), HIT(피안타), AB(타수), BF(투구 수) 제거


```python
private_pitcher = private_pitcher.drop(['SLG', 'KK', 'HIT', 'AB', 'BF'], axis=1)
```

#### [31]은 타자의 변수들 간 선형 상관관계를 시각화한다.

타자 데이터에서는 변수들끼리 선형 상관성이 낮아 추가적으로 변수 제거를 할 필요성이 없었다.


```python
temp = private_batter.drop(['GDAY_DS', 'HIT', 'AVG'],axis=1)
figure,ax1 = plt.subplots()
figure.set_size_inches(10,10)

sns.heatmap(temp.corr(),annot=True,cmap='YlGnBu')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x258b61365c0>




![png](output_57_1.png)


### 최종 변수


```python
private_pitcher.columns
```




    Index(['GDAY_DS', 'T_ID', 'P_ID', 'TB_SC', 'INN2', 'PA-AB', 'H1', 'H2', 'H3',
           'HR', 'SB_SR', 'WP', 'ER', 'ERA', 'BABIP', 'KK9', 'BB9'],
          dtype='object')




```python
private_batter.columns
```




    Index(['GDAY_DS', 'T_ID', 'P_ID', 'TB_SC', 'PA-AB', 'AB', 'RUN', 'RBI', 'HIT',
           'SH+SF', 'KK', 'AVG', 'SB_trial', 'BABIP'],
          dtype='object')



# 5. 데이터 내보내기


```python
private_pitcher.to_csv('data/private_pitcher.csv',index=False)
```


```python
private_batter.to_csv('data/private_batter.csv',index=False)
```


```python
team_pitcher.to_csv('data/pitcher_T.csv',index=False)
```


```python
team_batter.to_csv('data/batter_T.csv',index=False)
```
