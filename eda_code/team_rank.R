library(tidyverse)
library(lubridate)
library(plotly)
library(dplyr)
Sys.setlocale("LC_ALL","korean")
# KBO Team rank from 14~19

year = 2015:2019

HH = c(6,7,8,3,9)
HT = c(7,5,1,5,7)
KT = c(10,10,10,9,6)
LG = c(9,4,6,8,4)
LT = c(8,8,3,7,10)
NC = c(3,2,4,10,5)
OB = c(1,1,2,2,1)
SK = c(5,6,5,1,3)
SS = c(2,9,9,6,8)
WO = c(4,3,7,4,2)




fig = plot_ly(x = ~year)

fig = fig %>% add_lines(y = ~HH, name= '한화',line = list(shape = 'linear'))

fig = fig %>% add_lines(y = ~HT, name= '기아',line = list(shape = 'linear'))


fig = fig %>% add_lines(y = ~KT, name= 'KT',line = list(shape = 'linear'))
fig = fig %>% add_lines(y = ~LG, name= 'LG',line = list(shape = 'linear'))
fig = fig %>% add_lines(y = ~LT, name= '롯데',line = list(shape = 'linear'))
fig = fig %>% add_lines(y = ~NC, name= 'NC',line = list(shape = 'linear'))
fig = fig %>% add_lines(y = ~OB, name= '두산',line = list(shape = 'linear'))
fig = fig %>% add_lines(y = ~SK, name= 'SK',line = list(shape = 'linear'))
fig = fig %>% add_lines(y = ~SS, name= '삼성',line = list(shape = 'linear'))
fig = fig %>% add_lines(y = ~WO, name= '넥센,키움',line = list(shape = 'linear'))
fig = fig %>% layout(title = "Team Rank by Year", paper_bgcolor='rgb(250,250,250)')

 f1 <- list(
   family = "Arial, sans-serif",
   size = 20,
   color = "black"
 )
 f2 <- list(
   family = "Old Standard TT, serif",
   size = 20,
   color = "black"
 )
 a <- list(
   dtick  = 1,
   ticklen = 10,
   tickwidth = 4,
   title = "rank",
   titlefont = f1,
   showticklabels = TRUE,
   tickangle = 20,
   tickfont = f2,
   exponentformat = "E"
 )
 b <- list(
   
   ticklen = 10,
   tickwidth = 4,
   title = "year",
   titlefont = f1,
   showticklabels = TRUE,
   tickangle = 20,
   tickfont = f2,
   exponentformat = "E"



 )
 



fig <- fig %>% layout(yaxis = a, xaxis = b, showlegend = TRUE)

fig <- fig %>% layout(yaxis = list(autorange = "reversed"))

fig


