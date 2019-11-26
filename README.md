
# Rishabh Agrawal
# Assignment 4

# Libraries


```python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
import sklearn.linear_model as Lm
from sklearn.linear_model import LinearRegression as Lr
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import pycountry_convert as pc
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
```

# Problem 1


```python
p1 = pd.read_table("Files/Sdf16_1a.txt")
```

    C:\Users\risha\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:3020: DtypeWarning: Columns (0,3) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    


```python
totalFed = p1['TFEDREV'].sum()
print("Total Federal Budget = ", totalFed)
print("Amount to cut i.e. 15% of Total Budget = ", 0.15 * totalFed)
print("Total amount left for the new Budget = ", 0.85 * totalFed)
```

    Total Federal Budget =  55602739138
    Amount to cut i.e. 15% of Total Budget =  8340410870.7
    Total amount left for the new Budget =  47262328267.299995
    


```python
def makeEqualBudget(df, cutPercent, total):
    amountCut = (cutPercent/100) * total
    ans = [0]*len(df)
    newsum = 0
    for i in range(len(df)):
        if (df['AmountLeft'][i]  > 0):
            newsum += df['AmountLeft'][i]
    cutPercent = amountCut/newsum
    
    for i in range(len(df)):
        if (amountCut == 0):
            break
        if (df['AmountLeft'][i] > 0):
            ans[i] = cutPercent * df['AmountLeft'][i]
            amountCut -= ans[i] 

    return amountCut, ans
            
```


```python
p4 = p1
p4 = p4.fillna(0)
p4['AmountLeft'] = p1['TOTALREV'] - p1['TOTALEXP']
amountLeft, p4['Federal Amount Cut'] = makeEqualBudget(p4, 15, totalFed)
p4['Got Budget Cut'] = (p4['Federal Amount Cut'] > 0)
p4['Cut Proportion From Total'] = p4['Federal Amount Cut'] / p1['TOTALREV']
p41 = p4[['LEAID','NAME', 'STNAME','Federal Amount Cut', 'Cut Proportion From Total', 'TOTALREV', 'Got Budget Cut']].sort_values('Cut Proportion From Total', ascending=False)
p41.fillna(0, inplace = True)
```


```python
plt.figure(figsize=(10,5))
plt.hist(p41['Cut Proportion From Total'],  bins = 20)
plt.title("Histogram of Proportion of Budget Cuts for Each District")
plt.xlabel("Proportion")
plt.ylabel('Count')
```




    Text(0,0.5,'Count')




![png](/Images/output_8_1.png)


We can see that majority of the budget cuts are proportionally in the low range i.e. less than 0.15


```python
p41.head(25)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LEAID</th>
      <th>NAME</th>
      <th>STNAME</th>
      <th>Federal Amount Cut</th>
      <th>Cut Proportion From Total</th>
      <th>TOTALREV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15896</th>
      <td>4800191</td>
      <td>NORTHWEST PREPARATORY</td>
      <td>Texas</td>
      <td>5.908030e+02</td>
      <td>0.295402</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>15838</th>
      <td>4800091</td>
      <td>CHILDREN FIRST ACADEMY OF DALLAS</td>
      <td>Texas</td>
      <td>4.342402e+04</td>
      <td>0.295402</td>
      <td>147000</td>
    </tr>
    <tr>
      <th>10134</th>
      <td>3400059</td>
      <td>Galloway Community Charter School</td>
      <td>New Jersey</td>
      <td>8.359863e+04</td>
      <td>0.295402</td>
      <td>283000</td>
    </tr>
    <tr>
      <th>10136</th>
      <td>3400062</td>
      <td>Educational Information and Resource Center</td>
      <td>New Jersey</td>
      <td>1.713329e+04</td>
      <td>0.295402</td>
      <td>58000</td>
    </tr>
    <tr>
      <th>10142</th>
      <td>3400075</td>
      <td>Central Jersey Arts Charter School</td>
      <td>New Jersey</td>
      <td>1.010273e+05</td>
      <td>0.295402</td>
      <td>342000</td>
    </tr>
    <tr>
      <th>345</th>
      <td>400206</td>
      <td>Desert Springs Academy</td>
      <td>Arizona</td>
      <td>1.562674e+05</td>
      <td>0.295402</td>
      <td>529000</td>
    </tr>
    <tr>
      <th>12944</th>
      <td>3901570</td>
      <td>Citizens Academy Southeast</td>
      <td>Ohio</td>
      <td>4.428069e+05</td>
      <td>0.295402</td>
      <td>1499000</td>
    </tr>
    <tr>
      <th>15791</th>
      <td>4800023</td>
      <td>IGNITE PUBLIC SCHOOLS AND COMMUNITY SERVICE CE...</td>
      <td>Texas</td>
      <td>2.954015e+02</td>
      <td>0.295402</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>15956</th>
      <td>4800291</td>
      <td>WINDHAM SCHOOL DISTRICT</td>
      <td>Texas</td>
      <td>9.461710e+05</td>
      <td>0.295402</td>
      <td>3203000</td>
    </tr>
    <tr>
      <th>12352</th>
      <td>3800395</td>
      <td>ROUGHRIDER AREA CAREER &amp; TECHNICAL CENTER</td>
      <td>North Dakota</td>
      <td>1.078216e+05</td>
      <td>0.295402</td>
      <td>365000</td>
    </tr>
    <tr>
      <th>2878</th>
      <td>09D0001</td>
      <td>COMMITTEE FOR SHARED SERVICES</td>
      <td>Connecticut</td>
      <td>1.296813e+05</td>
      <td>0.295402</td>
      <td>439000</td>
    </tr>
    <tr>
      <th>15935</th>
      <td>4800261</td>
      <td>HARRIS COUNTY DEPT OF ED</td>
      <td>Texas</td>
      <td>5.408802e+05</td>
      <td>0.295402</td>
      <td>1831000</td>
    </tr>
    <tr>
      <th>6128</th>
      <td>2307110</td>
      <td>Kingsbury Plt Public Schools</td>
      <td>Maine</td>
      <td>2.954015e+02</td>
      <td>0.295402</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>6047</th>
      <td>2300058</td>
      <td>West Forks Plt Public Schools</td>
      <td>Maine</td>
      <td>8.566644e+03</td>
      <td>0.285555</td>
      <td>30000</td>
    </tr>
    <tr>
      <th>17152</th>
      <td>4900189</td>
      <td>ATHLOS ACADEMY OF UTAH</td>
      <td>Utah</td>
      <td>3.308497e+04</td>
      <td>0.266814</td>
      <td>124000</td>
    </tr>
    <tr>
      <th>2811</th>
      <td>0903512</td>
      <td>EDUCATION CONNECTION</td>
      <td>Connecticut</td>
      <td>3.593559e+06</td>
      <td>0.260554</td>
      <td>13792000</td>
    </tr>
    <tr>
      <th>7676</th>
      <td>2680995</td>
      <td>Wayne RESA</td>
      <td>Michigan</td>
      <td>9.173901e+07</td>
      <td>0.258474</td>
      <td>354925000</td>
    </tr>
    <tr>
      <th>17289</th>
      <td>5003630</td>
      <td>Duxbury School District</td>
      <td>Vermont</td>
      <td>5.317227e+03</td>
      <td>0.241692</td>
      <td>22000</td>
    </tr>
    <tr>
      <th>2747</th>
      <td>0901360</td>
      <td>EASTERN CONNECTICUT REGIONAL</td>
      <td>Connecticut</td>
      <td>5.980994e+06</td>
      <td>0.241169</td>
      <td>24800000</td>
    </tr>
    <tr>
      <th>2732</th>
      <td>0900910</td>
      <td>COOPERATIVE EDUCATIONAL SERVI</td>
      <td>Connecticut</td>
      <td>7.767583e+06</td>
      <td>0.238841</td>
      <td>32522000</td>
    </tr>
    <tr>
      <th>1424</th>
      <td>601391</td>
      <td>Southern Placer Schools Transportation Authority</td>
      <td>California</td>
      <td>1.033905e+04</td>
      <td>0.229757</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>7340</th>
      <td>2619140</td>
      <td>School District of the City of Inkster</td>
      <td>Michigan</td>
      <td>5.497422e+05</td>
      <td>0.226231</td>
      <td>2430000</td>
    </tr>
    <tr>
      <th>6093</th>
      <td>2304830</td>
      <td>Damariscotta Public Schools</td>
      <td>Maine</td>
      <td>4.174023e+05</td>
      <td>0.224651</td>
      <td>1858000</td>
    </tr>
    <tr>
      <th>2808</th>
      <td>0903450</td>
      <td>LEARN</td>
      <td>Connecticut</td>
      <td>7.611906e+06</td>
      <td>0.202471</td>
      <td>37595000</td>
    </tr>
    <tr>
      <th>7665</th>
      <td>2680880</td>
      <td>Oakland Schools</td>
      <td>Michigan</td>
      <td>5.547670e+07</td>
      <td>0.193697</td>
      <td>286410000</td>
    </tr>
  </tbody>
</table>
</div>



These are the top 25 districts which are affected the most by the budget cuts because they lost around 30-20% of there total budget becaus eof the budget cuts

# Problem 2


```python
p2 = pd.read_csv("Files/ccd_lea_052_1516_w_1a_011717.csv")
p2['LEAID'] = p2['LEAID'].astype(object)
p41['LEAID'] = p41['LEAID'].astype(object)
```


```python
p2merged = pd.merge(left = p2, right = p41, left_on = "LEA_NAME", right_on = "NAME")
p2merged = p2merged[p2merged['TOTAL'] > 0]
```


```python
races = {
    'AM' : 'American Indian/Alaska Native',
    'AS' : 'Asian',
    'HI' : 'Hispanic',
    'BL' : 'Black',
    'WH' : 'White',
    'HP' : 'Hawaiian Native / Pacific Islander',
    'TR' : 'Two or More Races'
}
```


```python
for i in races:
    p2merged['p-'+i] = abs(p2merged[i]/p2merged['TOTAL']).fillna(0)
```


```python
for i in races:
    fig, axs = plt.subplots(ncols=2,figsize=(15,5))
    axs[0].hist(p2merged[p2merged['Got Budget Cut'] == True]['p-'+i])
    axs[1].hist(p2merged[p2merged['Got Budget Cut'] == False]['p-'+i], color = 'Red')
    axs[0].set_title(races[i] + ': Budget Cut')
    axs[1].set_title(races[i] + ': No Budget Cut')
    plt.show()
```


![png](/Images/output_17_0.png)



![png](/Images/output_17_1.png)



![png](/Images/output_17_2.png)



![png](/Images/output_17_3.png)



![png](/Images/output_17_4.png)



![png](/Images/output_17_5.png)



![png](/Images/output_17_6.png)


From the plots of the respective races we can see that there isnt really that big of a difference between histograms of Budget Cut and No Budget Cuts except Hispanic, Black and White races.
It can be clearly seen that my methods affects these 3 races much more than the other races. White race gets the most distinct plots by my budget cut as districts with less ratio of white people are more affected because they are having more budget cuts. As the plots for all the other races look almost the same we can say that none of the races were affected by the budget cut, except white race which was affected a little. It is evident from the plots that the count of each race is significantly higher than the one for no budget cut, but that is just because there are less districts with no budget cuts and mmore district with more budget cuts. Some bias in my assumptions would be that I ignored the districts with missing values, so that might produce some kind of bias in my assumption, but that would need to be explored more.

# Problem 3


```python
p3 = pd.read_csv("Files/ccd_lea_002089_1516_w_1a_011717.csv")
```


```python
p3 = p3[p3['SPECED'] > 0]
p3merged = pd.merge(left = p3, right = p41, left_on = "LEA_NAME", right_on = "NAME")
p3merged = pd.merge(left = p3merged, right = p2, left_on = "LEA_NAME", right_on = "LEA_NAME")
p3merged = p3merged[p3merged['TOTAL'] > 0]
```


```python
p3merged['p-disability'] = (p3merged['SPECED'] / p3merged['TOTAL']).fillna(0.5)
p3merged = p3merged[p3merged['p-disability'] < 1.0]
```


```python
plt.hist(p3merged['p-disability'], bins = 30)
plt.title('Proportion of Children with Disability for Each District')
```




    Text(0.5,1,'Proportion of Children with Disability for Each District')




![png](/Images/output_23_1.png)


We can see that the proportion of the children with disability is significantly less and stays mostly below 40% for each district.


```python
fig, axs = plt.subplots(ncols=2,figsize=(15,5))
axs[0].hist(p3merged[p3merged['Got Budget Cut'] == True]['p-disability'])
axs[1].hist(p3merged[p3merged['Got Budget Cut'] == False]['p-disability'], color = 'Red')
axs[0].set_title('Children With Disability' + ': Budget Cut')
axs[1].set_title('Children With Disability'+ ': No Budget Cut')
```




    Text(0.5,1,'Children With Disability: No Budget Cut')




![png](/Images/output_25_1.png)


As we see that the method that I chose to do the budget cut, didnt really made a difference for children with disability as the plot for budget cut and no budget cut are pretty similar. One reason for bias that can be there in this model is that the i removed all the missing values so that might have produced some kind of bias in the data as treating them by replacing them with median/mean or some other analysis might have yielded a different result. But since the count of the missing data was less, I went ahead with removing it as I thought it wont severely affect the analysis that I made.

# Problem 4

I choose to critique the the HW3 of this github repository : https://github.com/alefiya-naseem/DataVizHw3

They chose to do the budget cut on schools who were making more money then they were spending. Actually there approach is very similar to mine where they only chose the schools whose revenue was more then there expenditure, as taking only the excess money away wont harm the current functioning of that school. Then rather than cutting equal amount of money from all the districts with excess funds, they decided to cut funding proportionally to the money they have. The districts with more money get more money cut off then the district with less money.
I think this is a very justified way of cutting the money because the district with less money will have more money remaining for them this way, whereas cutting 15% from all would have been fair but wouldnt have been justified which is what I did.
So I think this way handles the problem very well, not affecting the workings of exisitng districts budgets. Only thing this would impact is that in an event, where a big district goes into debt, they wont have enough funds to recover as it has been already cut, whereas smaller districts might still recover.

# Problem 5

I choose to summarize and comment on the Map Reduce + Hadoop Lecture by Prof. Jan Vitek. 
He clearly described that what the Map Reduce is used for in today's industry and how it affect the market. He clearly told the pros and cons of using Map Reduce and talked about how it is implemented. From fault tolerance to status monitoring there are various programs implemented in order to successfully run Map Reduce without errors or failed system. The phenomenon of picking of slack of some faulted CPU's is a brilliant idea which avoids the whole system going down. He talked about hadoop which is a way to implement distributed computing. He then talked about various milestone achieved by Hadoop. The pros of it would be that it is various easy to implement, large amount of datas can be handled to do efficient and fast computing but cons would be that complex architecture or procedure are very hard to implement. So you need to decide for the type of task if you need or if Map Reduce will do the job or not. All in all he summarised how or what problems are being solved today using it and how tech giants are using it in order to further expand their industry and how its getting popular with tech companies of all sizes.
