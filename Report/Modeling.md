```python
import numpy as np
import pandas as pd
import datetime
```

## 1. 데이터프레임 변형
- 선수별로 row에서 x값은 그대로 사용하되, y값은 그 선수가 출전한 다음 경기의 값으로 채워줌
- test data는 선수별 가장 최근 경기의 x값으로 구성

### 기존 데이터프레임

- 전처리가 끝난 데이터를 불러온다
- datetime 패키지를 이용해서 GDAY_DS변수를 연도-월-일 형식으로 변환


```python
batter = pd.read_csv('data/private_batter.csv')
batter = batter.sort_values(by=['P_ID','GDAY_DS'])
batter['GDAY_DS'] = pd.to_datetime(batter['GDAY_DS'],format='%Y%m%d')
```


```python
pitcher = pd.read_csv('data/private_pitcher.csv')
pitcher = pitcher.sort_values(by=['P_ID','GDAY_DS'])
pitcher['GDAY_DS'] = pd.to_datetime(pitcher['GDAY_DS'],format='%Y%m%d')
```

### 타자 데이터프레임 변형

##### train 데이터
- 해당 날짜의 x값은 그대로 사용하되, y값은 다음 경기의 값으로 변형
- 예를들어 A선수의 2020.08.23 경기 기록에서 y값에 해당하는 AB HIT 변수를 삭제하고 A선수가 출전한 다음 경기인 2020.08.24 경기의 AB HIT의 값을 채워주는 방식


```python
df=pd.DataFrame()
for i in batter.P_ID.unique():
    temp = batter[batter['P_ID']==i].drop('AVG',axis=1)
    temp = temp.reset_index(drop='index')
    future_y = temp[['AB','HIT']]
    x = temp.drop(['AB','HIT'],axis=1)
    x = x.shift(periods=1,axis=0)
    new = pd.concat([x,future_y],axis=1) #x값은 그대로 y값은 future_y 즉 다음 경기 기록으로 채워주는 방식
    new = new.drop(0)
    df = pd.concat([df,new])
```


```python
df = df.reset_index(drop='index')
```


```python
df.to_csv('data/batter_train.csv',index=False)
```

##### test data
- 각 선수별 가장 마지막 경기 기록의 x값만 select해서 test set 구성
- 선수별 마지막 경기가 2020년인 경우에만 test set에 포함


```python
final_x = pd.DataFrame()
for i in batter.P_ID.unique():
    temp = batter[batter['P_ID']==i].drop(['AVG','AB','HIT'],axis=1).tail(1) #가장 마지막 경기 tail(가장 최신 경기기록)
    final_x = pd.concat([final_x,temp])
```


```python
final_x = final_x[final_x['GDAY_DS'].dt.year>=2020] 
final_x = final_x.reset_index(drop='index')
final_x.to_csv('data/batter_test.csv',index=False)
```

### 투수 데이터프레임 변형
- 타자 데이터프레임 구성 방식과 동일하므로 설명 생략

- train 데이터


```python
df=pd.DataFrame()
for i in pitcher.P_ID.unique():
    temp = pitcher[pitcher['P_ID']==i].drop('ERA',axis=1)
    temp = temp.reset_index(drop='index')
    future_y = temp[['INN2','ER']]
    x = temp.drop(['INN2','ER'],axis=1)
    x = x.shift(periods=1,axis=0)
    new = pd.concat([x,future_y],axis=1)
    new = new.drop(0)
    df = pd.concat([df,new])
```


```python
df = df.reset_index(drop='index')
df.to_csv('data/pitcher_train.csv',index=False)
```

- test x 


```python
final_x = pd.DataFrame()
for i in pitcher.P_ID.unique():
    temp = pitcher[pitcher['P_ID']==i].drop(['INN2','ER','ERA'],axis=1).tail(1)
    final_x = pd.concat([final_x,temp])
```


```python
final_x = final_x[final_x['GDAY_DS'].dt.year>=2020]
```


```python
final_x = final_x.reset_index(drop='index')
final_x.to_csv('data/pitcher_test.csv',index=False)
```

## 2 모델링
- 선수별 가장 마지막 경기의 X값을 통해 Y값을 예측한다

#### ERA 예측 방식(Y: INN2 ER)
- ERA를 예측하기 위해 필요한 변수는 INN2와 ER로 투수 데이터 프레임을 이용했다
- LGBM Xgboost RandomForest 3가지 모델을 사용하였고, gridsearch-cv를 통해 best parameter를 찾았다
- 그 결과 LGBM을 사용할 때 MSE값이 가장 낮게 나왔다
- ERA를 예측하기 위해 최종적으로 LGBM을 사용하였기에,  본 글에서는 grid-search와 다른 모델 코드는 생략 (생략된 내용은 PPT에서 확인 가능)

#### AVG 예측 방식(Y:HIT AB)
- AVG를 예측하기 위해 필요한 변수는 HIT와 AB로 타자 데이터 프레임을 이용했다
- LGBM Xgboost RandomForest 3가지 모델을 사용하였고, gridsearch-cv를 통해 best parameter를 찾았다
- 그 결과 Xgboost을 사용할 때 MSE값이 가장 낮게 나왔다
- AVG를 예측하기 위해 최종적으로 Xgboost을 사용하였기에,  본 글에서는 grid-search와 다른 모델 코드는 생략 (생략된 내용은 PPT에서 확인 가능)

#### 승률 예측 방식(피타고리안 승률)
- 2020년 팀 데이터에서 각 팀별 득점과 실점을 통해 각 팀 별 승률을 예측해주었다

## 2-1 투수: LGBM


```python
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
```


```python
pitcher_train = pd.read_csv("data/pitcher_train.csv")
pitcher_test = pd.read_csv("data/pitcher_test.csv")
```


```python
pitcher_team = pitcher_test['T_ID']
```


```python
pitcher_train = pitcher_train.drop(columns = ['GDAY_DS','T_ID','P_ID'])
pitcher_test = pitcher_test.drop(columns=['GDAY_DS','P_ID','T_ID'])
```


```python
cat_features = ['TB_SC']
pitcher_train[cat_features] = pitcher_train[cat_features].astype('category')
pitcher_test[cat_features] = pitcher_test[cat_features].astype('category')
```


```python
pitcher_train = pd.get_dummies(pitcher_train)
pitcher_test = pd.get_dummies(pitcher_test)
```

- ER 예측


```python
X = pitcher_train.drop(columns = ['ER','INN2'])
y = pitcher_train['ER']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 2020)
```


```python
lgb1 = LGBMRegressor(boosting_type='gbdt', num_boost_round=2000, learning_rate=0.01,
                    lambda_l1 = 1.5,
                    lambda_l2 = 1,
                    min_data_in_leaf = 400,
                    num_leaves = 30,
                    reg_alpha = 0.1)
```


```python
lgb1.fit(X_train,y_train)
y_pred = lgb1.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mse
```




    2.0221188524575013




```python
ER = lgb1.predict(pitcher_test)
```

- INN2예측


```python
X = pitcher_train.drop(columns = ['INN2','ER'])
y = pitcher_train['INN2']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 2020)
```


```python
lgb2 = LGBMRegressor(boosting_type='gbdt', num_boost_round=2000, learning_rate=0.01,
                    lambda_l1 = 1.5,
                    lambda_l2 = 0,
                    min_data_in_leaf = 300,
                    num_leaves = 50,
                    reg_alpha = 0.1)
```


```python
lgb2.fit(X_train,y_train)
y_pred = lgb2.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mse
```




    12.50284231058004




```python
INN2 = lgb2.predict(pitcher_test)
```


```python
pitcher_test['INN2'] = INN2
pitcher_test['ER'] = np.round(ER,2)
pitcher_test['T_ID'] = pitcher_team
```

## 3.타자: XGB



```python
batter_train = pd.read_csv("data/batter_train.csv")
batter_test = pd.read_csv("data/batter_test.csv")
```


```python
batter_team = batter_test['T_ID']
```


```python
batter_train = batter_train.drop(columns=['GDAY_DS','T_ID','P_ID'])
batter_test = batter_test.drop(['GDAY_DS','T_ID','P_ID'],axis=1)
```


```python
cat_features = ['TB_SC']
batter_train[cat_features] = batter_train[cat_features].astype('category')
batter_test[cat_features] = batter_test[cat_features].astype('category')
```


```python
batter_train = pd.get_dummies(batter_train)
batter_test = pd.get_dummies(batter_test)
```

- AB


```python
X= batter_train.drop(columns=['HIT','AB'])
y= batter_train['AB']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 2020)
```


```python
xgb1 = XGBRegressor(colsample_bytree= 1, gamma= 2, learning_rate= 0.01, n_estimators= 500, subsample= 0.5)
```


```python
xgb1.fit(X_train, y_train)
y_pred = xgb1.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mse
```




    2.040222122413691




```python
AB = xgb1.predict(batter_test)
```

- HIT


```python
X= batter_train.drop(columns=['HIT','AB'])
y= batter_train['HIT']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 2020)
```


```python
xgb2 = XGBRegressor(colsample_bytree= 0.7, gamma= 2, learning_rate= 0.01, n_estimators= 500, subsample= 0.5)
```


```python
xgb2.fit(X_train, y_train)
y_pred = xgb2.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mse
```




    0.7912551088376826




```python
HIT =  xgb2.predict(batter_test)
```


```python
batter_test['AB'] = AB
batter_test['HIT'] = HIT
batter_test['T_ID'] = batter_team
```

## 4.최종적인 Y값 : AVG, ERA, 승률


```python
#avg
hit = batter_test['HIT'].groupby(batter_test['T_ID']).sum()
ab = batter_test['AB'].groupby(batter_test['T_ID']).sum()
AVG = hit/ab
```


```python
#era
inn2 = pitcher_test['INN2'].groupby(pitcher_test['T_ID']).sum()
er = pitcher_test['ER'].groupby(pitcher_test['T_ID']).sum()/3
ERA = er*9/inn2
```


```python
#승률
batter_T = pd.read_csv('data/batter_T.csv')
pitcher_T = pd.read_csv('data/pitcher_T.csv')
```


```python
run = batter_T['RUN'].groupby(batter_T['T_ID']).sum()
R = pitcher_T['R'].groupby(pitcher_T['T_ID']).sum()
WR = (run**2)/((run**2)+(R**2))
```


```python
df = pd.DataFrame({'타율': AVG,
                          '방어율': ERA,
                          '승률': WR})
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>타율</th>
      <th>방어율</th>
      <th>승률</th>
    </tr>
    <tr>
      <th>T_ID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>HH</th>
      <td>0.272515</td>
      <td>0.551424</td>
      <td>0.250240</td>
    </tr>
    <tr>
      <th>HT</th>
      <td>0.281469</td>
      <td>0.580171</td>
      <td>0.534899</td>
    </tr>
    <tr>
      <th>KT</th>
      <td>0.271968</td>
      <td>0.504492</td>
      <td>0.512446</td>
    </tr>
    <tr>
      <th>LG</th>
      <td>0.280787</td>
      <td>0.534595</td>
      <td>0.529560</td>
    </tr>
    <tr>
      <th>LT</th>
      <td>0.276635</td>
      <td>0.516658</td>
      <td>0.477718</td>
    </tr>
    <tr>
      <th>NC</th>
      <td>0.282384</td>
      <td>0.534549</td>
      <td>0.651110</td>
    </tr>
    <tr>
      <th>OB</th>
      <td>0.274859</td>
      <td>0.545387</td>
      <td>0.557373</td>
    </tr>
    <tr>
      <th>SK</th>
      <td>0.276308</td>
      <td>0.546736</td>
      <td>0.353222</td>
    </tr>
    <tr>
      <th>SS</th>
      <td>0.276568</td>
      <td>0.542690</td>
      <td>0.546796</td>
    </tr>
    <tr>
      <th>WO</th>
      <td>0.276960</td>
      <td>0.532718</td>
      <td>0.548047</td>
    </tr>
  </tbody>
</table>
</div>


