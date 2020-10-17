

```python
import pandas as pd
import numpy as np 
import sklearn as skl 
import matplotlib.pyplot as plt
```


```python
#Import Data 
df=pd.read_csv('https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv',low_memory=False)
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
      <th>SEVERITYCODE</th>
      <th>X</th>
      <th>Y</th>
      <th>OBJECTID</th>
      <th>INCKEY</th>
      <th>COLDETKEY</th>
      <th>REPORTNO</th>
      <th>STATUS</th>
      <th>ADDRTYPE</th>
      <th>INTKEY</th>
      <th>...</th>
      <th>ROADCOND</th>
      <th>LIGHTCOND</th>
      <th>PEDROWNOTGRNT</th>
      <th>SDOTCOLNUM</th>
      <th>SPEEDING</th>
      <th>ST_COLCODE</th>
      <th>ST_COLDESC</th>
      <th>SEGLANEKEY</th>
      <th>CROSSWALKKEY</th>
      <th>HITPARKEDCAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>-122.323148</td>
      <td>47.703140</td>
      <td>1</td>
      <td>1307</td>
      <td>1307</td>
      <td>3502005</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>37475.0</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-122.347294</td>
      <td>47.647172</td>
      <td>2</td>
      <td>52200</td>
      <td>52200</td>
      <td>2607959</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>6354039.0</td>
      <td>NaN</td>
      <td>11</td>
      <td>From same direction - both going straight - bo...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-122.334540</td>
      <td>47.607871</td>
      <td>3</td>
      <td>26700</td>
      <td>26700</td>
      <td>1482393</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>4323031.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-122.334803</td>
      <td>47.604803</td>
      <td>4</td>
      <td>1144</td>
      <td>1144</td>
      <td>3503937</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>From same direction - all others</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>-122.306426</td>
      <td>47.545739</td>
      <td>5</td>
      <td>17700</td>
      <td>17700</td>
      <td>1807429</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>34387.0</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>4028032.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-122.387598</td>
      <td>47.690575</td>
      <td>6</td>
      <td>320840</td>
      <td>322340</td>
      <td>E919477</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>36974.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>-122.338485</td>
      <td>47.618534</td>
      <td>7</td>
      <td>83300</td>
      <td>83300</td>
      <td>3282542</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29510.0</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>8344002.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>9</td>
      <td>330897</td>
      <td>332397</td>
      <td>EA30304</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29745.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>Vehicle Strikes Pedalcyclist</td>
      <td>6855</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>-122.335930</td>
      <td>47.611904</td>
      <td>10</td>
      <td>63400</td>
      <td>63400</td>
      <td>2071243</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>6166014.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>-122.384700</td>
      <td>47.528475</td>
      <td>12</td>
      <td>58600</td>
      <td>58600</td>
      <td>2072105</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>34679.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>6079001.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>48900</td>
      <td>48900</td>
      <td>2024040</td>
      <td>Matched</td>
      <td>Alley</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>6145029.0</td>
      <td>NaN</td>
      <td>22</td>
      <td>One car leaving driveway access</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>-122.333831</td>
      <td>47.547371</td>
      <td>15</td>
      <td>38800</td>
      <td>38800</td>
      <td>C654800</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>33194.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>5223041.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>-122.356273</td>
      <td>47.571375</td>
      <td>16</td>
      <td>2771</td>
      <td>2771</td>
      <td>1211870</td>
      <td>Unmatched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>3137016.0</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>-122.323966</td>
      <td>47.606374</td>
      <td>17</td>
      <td>32800</td>
      <td>32800</td>
      <td>2128498</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - No Street Lights</td>
      <td>NaN</td>
      <td>5356027.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2</td>
      <td>-122.317414</td>
      <td>47.664028</td>
      <td>19</td>
      <td>1212</td>
      <td>1212</td>
      <td>3507861</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30</td>
      <td>From opposite direction - all others</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>-122.337663</td>
      <td>47.617510</td>
      <td>20</td>
      <td>330878</td>
      <td>332378</td>
      <td>3838086</td>
      <td>Unmatched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>-122.344539</td>
      <td>47.692012</td>
      <td>21</td>
      <td>46300</td>
      <td>46300</td>
      <td>2023080</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>37365.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>5182022.0</td>
      <td>NaN</td>
      <td>28</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>23000</td>
      <td>23000</td>
      <td>537838</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>4016025.0</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>-122.328270</td>
      <td>47.571420</td>
      <td>24</td>
      <td>330833</td>
      <td>332333</td>
      <td>EA29752</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>-122.383802</td>
      <td>47.583715</td>
      <td>25</td>
      <td>97100</td>
      <td>97100</td>
      <td>2894590</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>9152035.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>-122.292403</td>
      <td>47.732847</td>
      <td>26</td>
      <td>1347</td>
      <td>1347</td>
      <td>3608880</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>-122.313786</td>
      <td>47.708535</td>
      <td>28</td>
      <td>1323</td>
      <td>1323</td>
      <td>3502831</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>36505.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>-122.318169</td>
      <td>47.615837</td>
      <td>29</td>
      <td>80000</td>
      <td>80000</td>
      <td>2882620</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>8200013.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>-122.337486</td>
      <td>47.589746</td>
      <td>31</td>
      <td>28700</td>
      <td>28700</td>
      <td>1213894</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>4256026.0</td>
      <td>NaN</td>
      <td>51</td>
      <td>Other object</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>-122.279658</td>
      <td>47.553405</td>
      <td>33</td>
      <td>1268</td>
      <td>1268</td>
      <td>3672152</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>33499.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Y</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>-122.312857</td>
      <td>47.599218</td>
      <td>34</td>
      <td>320932</td>
      <td>322432</td>
      <td>E926429</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>Vehicle Strikes Pedalcyclist</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>-122.330730</td>
      <td>47.615450</td>
      <td>35</td>
      <td>113300</td>
      <td>113300</td>
      <td>3346338</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29865.0</td>
      <td>...</td>
      <td>Wet</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>10317016.0</td>
      <td>NaN</td>
      <td>28</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>-122.385859</td>
      <td>47.581191</td>
      <td>36</td>
      <td>64700</td>
      <td>64700</td>
      <td>2798260</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>7204015.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>-122.304990</td>
      <td>47.611474</td>
      <td>37</td>
      <td>1083</td>
      <td>1083</td>
      <td>3605909</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>From same direction - all others</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38</td>
      <td>61500</td>
      <td>61500</td>
      <td>2607270</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>6320040.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>1</td>
      <td>-122.345863</td>
      <td>47.612991</td>
      <td>219510</td>
      <td>307577</td>
      <td>308857</td>
      <td>3811871</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20</td>
      <td>One car leaving parked position</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>1</td>
      <td>-122.330298</td>
      <td>47.603233</td>
      <td>219511</td>
      <td>312117</td>
      <td>313537</td>
      <td>E885893</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>30479.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>2</td>
      <td>-122.328079</td>
      <td>47.604161</td>
      <td>219512</td>
      <td>307692</td>
      <td>308972</td>
      <td>E851047</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29973.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
      <td>From same direction - one right turn - one str...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>1</td>
      <td>-122.320756</td>
      <td>47.608656</td>
      <td>219513</td>
      <td>312289</td>
      <td>313709</td>
      <td>E872044</td>
      <td>Unmatched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>2</td>
      <td>-122.355449</td>
      <td>47.704720</td>
      <td>219514</td>
      <td>308575</td>
      <td>309855</td>
      <td>E872078</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Vehicle going straight hits pedestrian</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>1</td>
      <td>-122.357624</td>
      <td>47.635956</td>
      <td>219515</td>
      <td>307792</td>
      <td>309072</td>
      <td>3578593</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>2</td>
      <td>-122.302382</td>
      <td>47.626759</td>
      <td>219516</td>
      <td>307985</td>
      <td>309265</td>
      <td>E869075</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29328.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>2</td>
      <td>-122.355298</td>
      <td>47.684382</td>
      <td>219517</td>
      <td>310417</td>
      <td>311717</td>
      <td>3751221</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>1</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>219519</td>
      <td>309594</td>
      <td>310874</td>
      <td>E881227</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29745.0</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Vehicle going straight hits pedestrian</td>
      <td>0</td>
      <td>522257</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>1</td>
      <td>-122.335527</td>
      <td>47.617434</td>
      <td>219520</td>
      <td>308698</td>
      <td>309978</td>
      <td>3812450</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>2</td>
      <td>-122.322097</td>
      <td>47.649615</td>
      <td>219521</td>
      <td>308342</td>
      <td>309622</td>
      <td>3693206</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>1</td>
      <td>-122.312679</td>
      <td>47.719414</td>
      <td>219522</td>
      <td>311117</td>
      <td>312437</td>
      <td>E881587</td>
      <td>Unmatched</td>
      <td>Intersection</td>
      <td>36427.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>2</td>
      <td>-122.380016</td>
      <td>47.664879</td>
      <td>219523</td>
      <td>310911</td>
      <td>312231</td>
      <td>3751425</td>
      <td>Unmatched</td>
      <td>Intersection</td>
      <td>25174.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>NaN</td>
      <td>0</td>
      <td>650595</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>2</td>
      <td>-122.340474</td>
      <td>47.614496</td>
      <td>219524</td>
      <td>312179</td>
      <td>313599</td>
      <td>E885580</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29545.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>Vehicle turning left hits pedestrian</td>
      <td>0</td>
      <td>523322</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>1</td>
      <td>-122.337137</td>
      <td>47.610709</td>
      <td>219525</td>
      <td>307834</td>
      <td>309114</td>
      <td>3811279</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11</td>
      <td>From same direction - both going straight - bo...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>1</td>
      <td>-122.302554</td>
      <td>47.584099</td>
      <td>219528</td>
      <td>311697</td>
      <td>313057</td>
      <td>E886941</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>1</td>
      <td>-122.298955</td>
      <td>47.717456</td>
      <td>219529</td>
      <td>308019</td>
      <td>309299</td>
      <td>3767486</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>219531</td>
      <td>308990</td>
      <td>310270</td>
      <td>3811749</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52</td>
      <td>Vehicle overturned</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>2</td>
      <td>-122.306092</td>
      <td>47.617881</td>
      <td>219532</td>
      <td>308532</td>
      <td>309812</td>
      <td>3814599</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>1</td>
      <td>-122.337880</td>
      <td>47.625793</td>
      <td>219535</td>
      <td>307802</td>
      <td>309082</td>
      <td>E869008</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>2</td>
      <td>-122.299160</td>
      <td>47.579673</td>
      <td>219536</td>
      <td>309335</td>
      <td>310615</td>
      <td>E880807</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>1</td>
      <td>-122.325887</td>
      <td>47.643191</td>
      <td>219537</td>
      <td>309222</td>
      <td>310502</td>
      <td>E879537</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>28300.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>1</td>
      <td>-122.304217</td>
      <td>47.669537</td>
      <td>219538</td>
      <td>308480</td>
      <td>309760</td>
      <td>3642620</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>26005.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>2</td>
      <td>-122.344569</td>
      <td>47.694547</td>
      <td>219539</td>
      <td>309170</td>
      <td>310450</td>
      <td>E879712</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>1</td>
      <td>-122.361672</td>
      <td>47.556722</td>
      <td>219541</td>
      <td>307804</td>
      <td>309084</td>
      <td>3745813</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>50</td>
      <td>Fixed object</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>2</td>
      <td>-122.290826</td>
      <td>47.565408</td>
      <td>219543</td>
      <td>309534</td>
      <td>310814</td>
      <td>E871089</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24</td>
      <td>From opposite direction - both moving - head-on</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>1</td>
      <td>-122.344526</td>
      <td>47.690924</td>
      <td>219544</td>
      <td>309085</td>
      <td>310365</td>
      <td>E876731</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13</td>
      <td>From same direction - both going straight - bo...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>2</td>
      <td>-122.306689</td>
      <td>47.683047</td>
      <td>219545</td>
      <td>311280</td>
      <td>312640</td>
      <td>3809984</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>24760.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>2</td>
      <td>-122.355317</td>
      <td>47.678734</td>
      <td>219546</td>
      <td>309514</td>
      <td>310794</td>
      <td>3810083</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>24349.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>Vehicle Strikes Pedalcyclist</td>
      <td>4308</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>1</td>
      <td>-122.289360</td>
      <td>47.611017</td>
      <td>219547</td>
      <td>308220</td>
      <td>309500</td>
      <td>E868008</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 38 columns</p>
</div>




```python
import numpy as np
# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
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
      <th>SEVERITYCODE</th>
      <th>X</th>
      <th>Y</th>
      <th>OBJECTID</th>
      <th>INCKEY</th>
      <th>COLDETKEY</th>
      <th>REPORTNO</th>
      <th>STATUS</th>
      <th>ADDRTYPE</th>
      <th>INTKEY</th>
      <th>...</th>
      <th>ROADCOND</th>
      <th>LIGHTCOND</th>
      <th>PEDROWNOTGRNT</th>
      <th>SDOTCOLNUM</th>
      <th>SPEEDING</th>
      <th>ST_COLCODE</th>
      <th>ST_COLDESC</th>
      <th>SEGLANEKEY</th>
      <th>CROSSWALKKEY</th>
      <th>HITPARKEDCAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>-122.323148</td>
      <td>47.703140</td>
      <td>1</td>
      <td>1307</td>
      <td>1307</td>
      <td>3502005</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>37475.0</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-122.347294</td>
      <td>47.647172</td>
      <td>2</td>
      <td>52200</td>
      <td>52200</td>
      <td>2607959</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>6354039.0</td>
      <td>NaN</td>
      <td>11</td>
      <td>From same direction - both going straight - bo...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-122.334540</td>
      <td>47.607871</td>
      <td>3</td>
      <td>26700</td>
      <td>26700</td>
      <td>1482393</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>4323031.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-122.334803</td>
      <td>47.604803</td>
      <td>4</td>
      <td>1144</td>
      <td>1144</td>
      <td>3503937</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>From same direction - all others</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>-122.306426</td>
      <td>47.545739</td>
      <td>5</td>
      <td>17700</td>
      <td>17700</td>
      <td>1807429</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>34387.0</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>4028032.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-122.387598</td>
      <td>47.690575</td>
      <td>6</td>
      <td>320840</td>
      <td>322340</td>
      <td>E919477</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>36974.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>-122.338485</td>
      <td>47.618534</td>
      <td>7</td>
      <td>83300</td>
      <td>83300</td>
      <td>3282542</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29510.0</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>8344002.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>9</td>
      <td>330897</td>
      <td>332397</td>
      <td>EA30304</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29745.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>Vehicle Strikes Pedalcyclist</td>
      <td>6855</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>-122.335930</td>
      <td>47.611904</td>
      <td>10</td>
      <td>63400</td>
      <td>63400</td>
      <td>2071243</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>6166014.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>-122.384700</td>
      <td>47.528475</td>
      <td>12</td>
      <td>58600</td>
      <td>58600</td>
      <td>2072105</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>34679.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>6079001.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>48900</td>
      <td>48900</td>
      <td>2024040</td>
      <td>Matched</td>
      <td>Alley</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>6145029.0</td>
      <td>NaN</td>
      <td>22</td>
      <td>One car leaving driveway access</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>-122.333831</td>
      <td>47.547371</td>
      <td>15</td>
      <td>38800</td>
      <td>38800</td>
      <td>C654800</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>33194.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>5223041.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>-122.356273</td>
      <td>47.571375</td>
      <td>16</td>
      <td>2771</td>
      <td>2771</td>
      <td>1211870</td>
      <td>Unmatched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>3137016.0</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>-122.323966</td>
      <td>47.606374</td>
      <td>17</td>
      <td>32800</td>
      <td>32800</td>
      <td>2128498</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - No Street Lights</td>
      <td>NaN</td>
      <td>5356027.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2</td>
      <td>-122.317414</td>
      <td>47.664028</td>
      <td>19</td>
      <td>1212</td>
      <td>1212</td>
      <td>3507861</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30</td>
      <td>From opposite direction - all others</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>-122.337663</td>
      <td>47.617510</td>
      <td>20</td>
      <td>330878</td>
      <td>332378</td>
      <td>3838086</td>
      <td>Unmatched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>-122.344539</td>
      <td>47.692012</td>
      <td>21</td>
      <td>46300</td>
      <td>46300</td>
      <td>2023080</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>37365.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>5182022.0</td>
      <td>NaN</td>
      <td>28</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>23000</td>
      <td>23000</td>
      <td>537838</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>4016025.0</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>-122.328270</td>
      <td>47.571420</td>
      <td>24</td>
      <td>330833</td>
      <td>332333</td>
      <td>EA29752</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>-122.383802</td>
      <td>47.583715</td>
      <td>25</td>
      <td>97100</td>
      <td>97100</td>
      <td>2894590</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>9152035.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>-122.292403</td>
      <td>47.732847</td>
      <td>26</td>
      <td>1347</td>
      <td>1347</td>
      <td>3608880</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>-122.313786</td>
      <td>47.708535</td>
      <td>28</td>
      <td>1323</td>
      <td>1323</td>
      <td>3502831</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>36505.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>-122.318169</td>
      <td>47.615837</td>
      <td>29</td>
      <td>80000</td>
      <td>80000</td>
      <td>2882620</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>8200013.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>-122.337486</td>
      <td>47.589746</td>
      <td>31</td>
      <td>28700</td>
      <td>28700</td>
      <td>1213894</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>4256026.0</td>
      <td>NaN</td>
      <td>51</td>
      <td>Other object</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>-122.279658</td>
      <td>47.553405</td>
      <td>33</td>
      <td>1268</td>
      <td>1268</td>
      <td>3672152</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>33499.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Y</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>-122.312857</td>
      <td>47.599218</td>
      <td>34</td>
      <td>320932</td>
      <td>322432</td>
      <td>E926429</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>Vehicle Strikes Pedalcyclist</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>-122.330730</td>
      <td>47.615450</td>
      <td>35</td>
      <td>113300</td>
      <td>113300</td>
      <td>3346338</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29865.0</td>
      <td>...</td>
      <td>Wet</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>10317016.0</td>
      <td>NaN</td>
      <td>28</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>-122.385859</td>
      <td>47.581191</td>
      <td>36</td>
      <td>64700</td>
      <td>64700</td>
      <td>2798260</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>7204015.0</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>-122.304990</td>
      <td>47.611474</td>
      <td>37</td>
      <td>1083</td>
      <td>1083</td>
      <td>3605909</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>From same direction - all others</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38</td>
      <td>61500</td>
      <td>61500</td>
      <td>2607270</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>6320040.0</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>1</td>
      <td>-122.345863</td>
      <td>47.612991</td>
      <td>219510</td>
      <td>307577</td>
      <td>308857</td>
      <td>3811871</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20</td>
      <td>One car leaving parked position</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>1</td>
      <td>-122.330298</td>
      <td>47.603233</td>
      <td>219511</td>
      <td>312117</td>
      <td>313537</td>
      <td>E885893</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>30479.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>2</td>
      <td>-122.328079</td>
      <td>47.604161</td>
      <td>219512</td>
      <td>307692</td>
      <td>308972</td>
      <td>E851047</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29973.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16</td>
      <td>From same direction - one right turn - one str...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>1</td>
      <td>-122.320756</td>
      <td>47.608656</td>
      <td>219513</td>
      <td>312289</td>
      <td>313709</td>
      <td>E872044</td>
      <td>Unmatched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>2</td>
      <td>-122.355449</td>
      <td>47.704720</td>
      <td>219514</td>
      <td>308575</td>
      <td>309855</td>
      <td>E872078</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Vehicle going straight hits pedestrian</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>1</td>
      <td>-122.357624</td>
      <td>47.635956</td>
      <td>219515</td>
      <td>307792</td>
      <td>309072</td>
      <td>3578593</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>2</td>
      <td>-122.302382</td>
      <td>47.626759</td>
      <td>219516</td>
      <td>307985</td>
      <td>309265</td>
      <td>E869075</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29328.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>2</td>
      <td>-122.355298</td>
      <td>47.684382</td>
      <td>219517</td>
      <td>310417</td>
      <td>311717</td>
      <td>3751221</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>1</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>219519</td>
      <td>309594</td>
      <td>310874</td>
      <td>E881227</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29745.0</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Vehicle going straight hits pedestrian</td>
      <td>0</td>
      <td>522257</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>1</td>
      <td>-122.335527</td>
      <td>47.617434</td>
      <td>219520</td>
      <td>308698</td>
      <td>309978</td>
      <td>3812450</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>2</td>
      <td>-122.322097</td>
      <td>47.649615</td>
      <td>219521</td>
      <td>308342</td>
      <td>309622</td>
      <td>3693206</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>1</td>
      <td>-122.312679</td>
      <td>47.719414</td>
      <td>219522</td>
      <td>311117</td>
      <td>312437</td>
      <td>E881587</td>
      <td>Unmatched</td>
      <td>Intersection</td>
      <td>36427.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>2</td>
      <td>-122.380016</td>
      <td>47.664879</td>
      <td>219523</td>
      <td>310911</td>
      <td>312231</td>
      <td>3751425</td>
      <td>Unmatched</td>
      <td>Intersection</td>
      <td>25174.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td></td>
      <td>NaN</td>
      <td>0</td>
      <td>650595</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>2</td>
      <td>-122.340474</td>
      <td>47.614496</td>
      <td>219524</td>
      <td>312179</td>
      <td>313599</td>
      <td>E885580</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29545.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>Vehicle turning left hits pedestrian</td>
      <td>0</td>
      <td>523322</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>1</td>
      <td>-122.337137</td>
      <td>47.610709</td>
      <td>219525</td>
      <td>307834</td>
      <td>309114</td>
      <td>3811279</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11</td>
      <td>From same direction - both going straight - bo...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>1</td>
      <td>-122.302554</td>
      <td>47.584099</td>
      <td>219528</td>
      <td>311697</td>
      <td>313057</td>
      <td>E886941</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>1</td>
      <td>-122.298955</td>
      <td>47.717456</td>
      <td>219529</td>
      <td>308019</td>
      <td>309299</td>
      <td>3767486</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>219531</td>
      <td>308990</td>
      <td>310270</td>
      <td>3811749</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52</td>
      <td>Vehicle overturned</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>2</td>
      <td>-122.306092</td>
      <td>47.617881</td>
      <td>219532</td>
      <td>308532</td>
      <td>309812</td>
      <td>3814599</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>1</td>
      <td>-122.337880</td>
      <td>47.625793</td>
      <td>219535</td>
      <td>307802</td>
      <td>309082</td>
      <td>E869008</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>2</td>
      <td>-122.299160</td>
      <td>47.579673</td>
      <td>219536</td>
      <td>309335</td>
      <td>310615</td>
      <td>E880807</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>1</td>
      <td>-122.325887</td>
      <td>47.643191</td>
      <td>219537</td>
      <td>309222</td>
      <td>310502</td>
      <td>E879537</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>28300.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>1</td>
      <td>-122.304217</td>
      <td>47.669537</td>
      <td>219538</td>
      <td>308480</td>
      <td>309760</td>
      <td>3642620</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>26005.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>2</td>
      <td>-122.344569</td>
      <td>47.694547</td>
      <td>219539</td>
      <td>309170</td>
      <td>310450</td>
      <td>E879712</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>1</td>
      <td>-122.361672</td>
      <td>47.556722</td>
      <td>219541</td>
      <td>307804</td>
      <td>309084</td>
      <td>3745813</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>50</td>
      <td>Fixed object</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>2</td>
      <td>-122.290826</td>
      <td>47.565408</td>
      <td>219543</td>
      <td>309534</td>
      <td>310814</td>
      <td>E871089</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24</td>
      <td>From opposite direction - both moving - head-on</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>1</td>
      <td>-122.344526</td>
      <td>47.690924</td>
      <td>219544</td>
      <td>309085</td>
      <td>310365</td>
      <td>E876731</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13</td>
      <td>From same direction - both going straight - bo...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>2</td>
      <td>-122.306689</td>
      <td>47.683047</td>
      <td>219545</td>
      <td>311280</td>
      <td>312640</td>
      <td>3809984</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>24760.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>2</td>
      <td>-122.355317</td>
      <td>47.678734</td>
      <td>219546</td>
      <td>309514</td>
      <td>310794</td>
      <td>3810083</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>24349.0</td>
      <td>...</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>Vehicle Strikes Pedalcyclist</td>
      <td>4308</td>
      <td>0</td>
      <td>N</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>1</td>
      <td>-122.289360</td>
      <td>47.611017</td>
      <td>219547</td>
      <td>308220</td>
      <td>309500</td>
      <td>E868008</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>...</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 38 columns</p>
</div>




```python
#to detect missing data
missing_data = df.isnull()
missing_data
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
      <th>SEVERITYCODE</th>
      <th>X</th>
      <th>Y</th>
      <th>OBJECTID</th>
      <th>INCKEY</th>
      <th>COLDETKEY</th>
      <th>REPORTNO</th>
      <th>STATUS</th>
      <th>ADDRTYPE</th>
      <th>INTKEY</th>
      <th>...</th>
      <th>ROADCOND</th>
      <th>LIGHTCOND</th>
      <th>PEDROWNOTGRNT</th>
      <th>SDOTCOLNUM</th>
      <th>SPEEDING</th>
      <th>ST_COLCODE</th>
      <th>ST_COLDESC</th>
      <th>SEGLANEKEY</th>
      <th>CROSSWALKKEY</th>
      <th>HITPARKEDCAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 38 columns</p>
</div>




```python
df.columns
```




    Index(['SEVERITYCODE', 'X', 'Y', 'OBJECTID', 'INCKEY', 'COLDETKEY', 'REPORTNO',
           'STATUS', 'ADDRTYPE', 'INTKEY', 'LOCATION', 'EXCEPTRSNCODE',
           'EXCEPTRSNDESC', 'SEVERITYCODE.1', 'SEVERITYDESC', 'COLLISIONTYPE',
           'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'INCDATE',
           'INCDTTM', 'JUNCTIONTYPE', 'SDOT_COLCODE', 'SDOT_COLDESC',
           'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND',
           'PEDROWNOTGRNT', 'SDOTCOLNUM', 'SPEEDING', 'ST_COLCODE', 'ST_COLDESC',
           'SEGLANEKEY', 'CROSSWALKKEY', 'HITPARKEDCAR'],
          dtype='object')




```python
df.drop(['SEVERITYCODE.1','REPORTNO','EXCEPTRSNCODE','EXCEPTRSNDESC','INCDATE','INCDTTM','INATTENTIONIND','UNDERINFL','PEDROWNOTGRNT','SPEEDING','ST_COLCODE','HITPARKEDCAR'], inplace=True, axis=1, errors='ignore')
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
      <th>SEVERITYCODE</th>
      <th>X</th>
      <th>Y</th>
      <th>OBJECTID</th>
      <th>INCKEY</th>
      <th>COLDETKEY</th>
      <th>STATUS</th>
      <th>ADDRTYPE</th>
      <th>INTKEY</th>
      <th>LOCATION</th>
      <th>...</th>
      <th>JUNCTIONTYPE</th>
      <th>SDOT_COLCODE</th>
      <th>SDOT_COLDESC</th>
      <th>WEATHER</th>
      <th>ROADCOND</th>
      <th>LIGHTCOND</th>
      <th>SDOTCOLNUM</th>
      <th>ST_COLDESC</th>
      <th>SEGLANEKEY</th>
      <th>CROSSWALKKEY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>-122.323148</td>
      <td>47.703140</td>
      <td>1</td>
      <td>1307</td>
      <td>1307</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>37475.0</td>
      <td>5TH AVE NE AND NE 103RD ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-122.347294</td>
      <td>47.647172</td>
      <td>2</td>
      <td>52200</td>
      <td>52200</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>AURORA BR BETWEEN RAYE ST AND BRIDGE WAY N</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>16</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, LEFT SIDE ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>6354039.0</td>
      <td>From same direction - both going straight - bo...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-122.334540</td>
      <td>47.607871</td>
      <td>3</td>
      <td>26700</td>
      <td>26700</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>4TH AVE BETWEEN SENECA ST AND UNIVERSITY ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>4323031.0</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-122.334803</td>
      <td>47.604803</td>
      <td>4</td>
      <td>1144</td>
      <td>1144</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>2ND AVE BETWEEN MARION ST AND MADISON ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>From same direction - all others</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>-122.306426</td>
      <td>47.545739</td>
      <td>5</td>
      <td>17700</td>
      <td>17700</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>34387.0</td>
      <td>SWIFT AVE S AND SWIFT AV OFF RP</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>4028032.0</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-122.387598</td>
      <td>47.690575</td>
      <td>6</td>
      <td>320840</td>
      <td>322340</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>36974.0</td>
      <td>24TH AVE NW AND NW 85TH ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>-122.338485</td>
      <td>47.618534</td>
      <td>7</td>
      <td>83300</td>
      <td>83300</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29510.0</td>
      <td>DENNY WAY AND WESTLAKE AVE</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>8344002.0</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>9</td>
      <td>330897</td>
      <td>332397</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29745.0</td>
      <td>BROADWAY AND E PIKE ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>51</td>
      <td>PEDALCYCLIST STRUCK MOTOR VEHICLE FRONT END AT...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>Vehicle Strikes Pedalcyclist</td>
      <td>6855</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>-122.335930</td>
      <td>47.611904</td>
      <td>10</td>
      <td>63400</td>
      <td>63400</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>PINE ST BETWEEN 5TH AVE AND 6TH AVE</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>6166014.0</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>-122.384700</td>
      <td>47.528475</td>
      <td>12</td>
      <td>58600</td>
      <td>58600</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>34679.0</td>
      <td>41ST AVE SW AND SW THISTLE ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>6079001.0</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>48900</td>
      <td>48900</td>
      <td>Matched</td>
      <td>Alley</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>Driveway Junction</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>6145029.0</td>
      <td>One car leaving driveway access</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>-122.333831</td>
      <td>47.547371</td>
      <td>15</td>
      <td>38800</td>
      <td>38800</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>33194.0</td>
      <td>1ST AV S BR NB AND EAST MARGINAL WAY S</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>5223041.0</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>-122.356273</td>
      <td>47.571375</td>
      <td>16</td>
      <td>2771</td>
      <td>2771</td>
      <td>Unmatched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>SW SPOKANE ST BETWEEN SW SPOKANE W BR AND TERM...</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>3137016.0</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>-122.323966</td>
      <td>47.606374</td>
      <td>17</td>
      <td>32800</td>
      <td>32800</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>TERRY AVE BETWEEN JAMES ST AND CHERRY ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>13</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, LEFT SIDE ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - No Street Lights</td>
      <td>5356027.0</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2</td>
      <td>-122.317414</td>
      <td>47.664028</td>
      <td>19</td>
      <td>1212</td>
      <td>1212</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>ROOSEVELT WAY NE BETWEEN NE 47TH ST AND NE 50T...</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>From opposite direction - all others</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>-122.337663</td>
      <td>47.617510</td>
      <td>20</td>
      <td>330878</td>
      <td>332378</td>
      <td>Unmatched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>9TH AVE BETWEEN LENORA ST AND BLANCHARD ST</td>
      <td>...</td>
      <td>Driveway Junction</td>
      <td>26</td>
      <td>MOTOR VEHICLE STRUCK OBJECT IN ROAD</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>-122.344539</td>
      <td>47.692012</td>
      <td>21</td>
      <td>46300</td>
      <td>46300</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>37365.0</td>
      <td>AURORA AVE N AND N 87TH ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>5182022.0</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>23000</td>
      <td>23000</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>BATTERY ST TUN ON RP BETWEEN BELL ST AND ALASK...</td>
      <td>...</td>
      <td>Mid-Block (but intersection related)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>4016025.0</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>-122.328270</td>
      <td>47.571420</td>
      <td>24</td>
      <td>330833</td>
      <td>332333</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>S SPOKANE SR ST BETWEEN 4TH AVE S AND 5TH AVE S</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>-122.383802</td>
      <td>47.583715</td>
      <td>25</td>
      <td>97100</td>
      <td>97100</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>41ST AVE SW BETWEEN SW WALKER ST AND SW COLLEG...</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Unknown</td>
      <td>Dry</td>
      <td>Unknown</td>
      <td>9152035.0</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>-122.292403</td>
      <td>47.732847</td>
      <td>26</td>
      <td>1347</td>
      <td>1347</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>LAKE CITY WAY NE BETWEEN NE 143RD ST AND NE 14...</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>-122.313786</td>
      <td>47.708535</td>
      <td>28</td>
      <td>1323</td>
      <td>1323</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>36505.0</td>
      <td>14TH AVE NE AND NE NORTHGATE WAY</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>-122.318169</td>
      <td>47.615837</td>
      <td>29</td>
      <td>80000</td>
      <td>80000</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>11TH AVE BETWEEN E PINE ST AND E OLIVE ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>8200013.0</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>-122.337486</td>
      <td>47.589746</td>
      <td>31</td>
      <td>28700</td>
      <td>28700</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>ALASKAN WY VI SB BETWEEN ALASKAN WY VI SB EFR ...</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>28</td>
      <td>MOTOR VEHICLE RAN OFF ROAD - HIT FIXED OBJECT</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>4256026.0</td>
      <td>Other object</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>-122.279658</td>
      <td>47.553405</td>
      <td>33</td>
      <td>1268</td>
      <td>1268</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>33499.0</td>
      <td>RAINIER AVE S AND S BRANDON ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>-122.312857</td>
      <td>47.599218</td>
      <td>34</td>
      <td>320932</td>
      <td>322432</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>S JACKSON ST BETWEEN 14TH AVE S AND 16TH AVE S</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>18</td>
      <td>MOTOR VEHICLE STRUCK PEDALCYCLIST, FRONT END A...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>Vehicle Strikes Pedalcyclist</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>-122.330730</td>
      <td>47.615450</td>
      <td>35</td>
      <td>113300</td>
      <td>113300</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29865.0</td>
      <td>BOREN AVE AND OLIVE WAY</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Unknown</td>
      <td>10317016.0</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>-122.385859</td>
      <td>47.581191</td>
      <td>36</td>
      <td>64700</td>
      <td>64700</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>SW ADMIRAL WAY BETWEEN 42ND AVE SW AND CALIFOR...</td>
      <td>...</td>
      <td>Driveway Junction</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>7204015.0</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>-122.304990</td>
      <td>47.611474</td>
      <td>37</td>
      <td>1083</td>
      <td>1083</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>21ST AVE BETWEEN E MARION ST AND E UNION ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>From same direction - all others</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38</td>
      <td>61500</td>
      <td>61500</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>1ST AVE S BETWEEN 1ST AVS ON N RP AND S ROYAL ...</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>16</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, LEFT SIDE ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>6320040.0</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>1</td>
      <td>-122.345863</td>
      <td>47.612991</td>
      <td>219510</td>
      <td>307577</td>
      <td>308857</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>1ST AVE BETWEEN BLANCHARD ST AND BELL ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>One car leaving parked position</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>1</td>
      <td>-122.330298</td>
      <td>47.603233</td>
      <td>219511</td>
      <td>312117</td>
      <td>313537</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>30479.0</td>
      <td>4TH AVE AND JAMES ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>2</td>
      <td>-122.328079</td>
      <td>47.604161</td>
      <td>219512</td>
      <td>307692</td>
      <td>308972</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29973.0</td>
      <td>6TH AVE AND JAMES ST</td>
      <td>...</td>
      <td>At Intersection (but not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>From same direction - one right turn - one str...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>1</td>
      <td>-122.320756</td>
      <td>47.608656</td>
      <td>219513</td>
      <td>312289</td>
      <td>313709</td>
      <td>Unmatched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>BROADWAY BETWEEN CHERRY ST AND E COLUMBIA ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>2</td>
      <td>-122.355449</td>
      <td>47.704720</td>
      <td>219514</td>
      <td>308575</td>
      <td>309855</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>GREENWOOD AVE N BETWEEN N 104TH ST AND HOLMAN ...</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>Vehicle going straight hits pedestrian</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>1</td>
      <td>-122.357624</td>
      <td>47.635956</td>
      <td>219515</td>
      <td>307792</td>
      <td>309072</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>W HOWE ST BETWEEN QUEEN ANNE AVE N AND 1ST AVE W</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>2</td>
      <td>-122.302382</td>
      <td>47.626759</td>
      <td>219516</td>
      <td>307985</td>
      <td>309265</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29328.0</td>
      <td>23RD AVE E AND E ALOHA ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>2</td>
      <td>-122.355298</td>
      <td>47.684382</td>
      <td>219517</td>
      <td>310417</td>
      <td>311717</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>GREENWOOD AVE N BETWEEN N 76TH ST AND N 77TH ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>1</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>219519</td>
      <td>309594</td>
      <td>310874</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29745.0</td>
      <td>BROADWAY AND E PIKE ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>24</td>
      <td>MOTOR VEHCILE STRUCK PEDESTRIAN</td>
      <td>Clear</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>Vehicle going straight hits pedestrian</td>
      <td>0</td>
      <td>522257</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>1</td>
      <td>-122.335527</td>
      <td>47.617434</td>
      <td>219520</td>
      <td>308698</td>
      <td>309978</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>TERRY AVE BETWEEN VIRGINIA ST AND LENORA ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>2</td>
      <td>-122.322097</td>
      <td>47.649615</td>
      <td>219521</td>
      <td>308342</td>
      <td>309622</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>HARVARD AVE E BETWEEN E ALLISON ST AND EASTLAK...</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>1</td>
      <td>-122.312679</td>
      <td>47.719414</td>
      <td>219522</td>
      <td>311117</td>
      <td>312437</td>
      <td>Unmatched</td>
      <td>Intersection</td>
      <td>36427.0</td>
      <td>15TH AVE NE AND NE 125TH ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>2</td>
      <td>-122.380016</td>
      <td>47.664879</td>
      <td>219523</td>
      <td>310911</td>
      <td>312231</td>
      <td>Unmatched</td>
      <td>Intersection</td>
      <td>25174.0</td>
      <td>LEARY AVE NW AND NW DOCK PL</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>24</td>
      <td>MOTOR VEHCILE STRUCK PEDESTRIAN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>650595</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>2</td>
      <td>-122.340474</td>
      <td>47.614496</td>
      <td>219524</td>
      <td>312179</td>
      <td>313599</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>29545.0</td>
      <td>5TH AVE AND LENORA ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>24</td>
      <td>MOTOR VEHCILE STRUCK PEDESTRIAN</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>Vehicle turning left hits pedestrian</td>
      <td>0</td>
      <td>523322</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>1</td>
      <td>-122.337137</td>
      <td>47.610709</td>
      <td>219525</td>
      <td>307834</td>
      <td>309114</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>4TH AVE BETWEEN PIKE ST AND PINE ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>0</td>
      <td>NOT ENOUGH INFORMATION / NOT APPLICABLE</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>From same direction - both going straight - bo...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>1</td>
      <td>-122.302554</td>
      <td>47.584099</td>
      <td>219528</td>
      <td>311697</td>
      <td>313057</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>RAINIER AVE S BETWEEN 23RD AVE S AND S WALKER ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>1</td>
      <td>-122.298955</td>
      <td>47.717456</td>
      <td>219529</td>
      <td>308019</td>
      <td>309299</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>NE 123RD ST BETWEEN HIRAM PL NE AND 28TH AVE NE</td>
      <td>...</td>
      <td>NaN</td>
      <td>0</td>
      <td>NOT ENOUGH INFORMATION / NOT APPLICABLE</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>219531</td>
      <td>308990</td>
      <td>310270</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>BATTERY ST TUNNEL NB BETWEEN ALASKAN WY VI NB ...</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>29</td>
      <td>MOTOR VEHICLE OVERTURNED IN ROAD</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Other</td>
      <td>NaN</td>
      <td>Vehicle overturned</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>2</td>
      <td>-122.306092</td>
      <td>47.617881</td>
      <td>219532</td>
      <td>308532</td>
      <td>309812</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>20TH AVE BETWEEN E MADISON ST AND E DENNY WAY</td>
      <td>...</td>
      <td>Mid-Block (but intersection related)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>1</td>
      <td>-122.337880</td>
      <td>47.625793</td>
      <td>219535</td>
      <td>307802</td>
      <td>309082</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>VALLEY ST BETWEEN WESTLAKE AVE N AND TERRY AVE N</td>
      <td>...</td>
      <td>Mid-Block (but intersection related)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>One parked--one moving</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>2</td>
      <td>-122.299160</td>
      <td>47.579673</td>
      <td>219536</td>
      <td>309335</td>
      <td>310615</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>RAINIER AVE S BETWEEN S BAYVIEW ST AND S MCCLE...</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>1</td>
      <td>-122.325887</td>
      <td>47.643191</td>
      <td>219537</td>
      <td>309222</td>
      <td>310502</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>28300.0</td>
      <td>EASTLAKE AVE E AND E ROANOKE ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>1</td>
      <td>-122.304217</td>
      <td>47.669537</td>
      <td>219538</td>
      <td>308480</td>
      <td>309760</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>26005.0</td>
      <td>NE PARK RD AND NE RAVENNA WB BV</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>2</td>
      <td>-122.344569</td>
      <td>47.694547</td>
      <td>219539</td>
      <td>309170</td>
      <td>310450</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>AURORA AVE N BETWEEN N 90TH ST AND N 91ST ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>Entering at angle</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>1</td>
      <td>-122.361672</td>
      <td>47.556722</td>
      <td>219541</td>
      <td>307804</td>
      <td>309084</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>PUGET BLVD SW BETWEEN SW HUDSON ST AND DEAD END 1</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>28</td>
      <td>MOTOR VEHICLE RAN OFF ROAD - HIT FIXED OBJECT</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>NaN</td>
      <td>Fixed object</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>2</td>
      <td>-122.290826</td>
      <td>47.565408</td>
      <td>219543</td>
      <td>309534</td>
      <td>310814</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>34TH AVE S BETWEEN S DAKOTA ST AND S GENESEE ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>From opposite direction - both moving - head-on</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>1</td>
      <td>-122.344526</td>
      <td>47.690924</td>
      <td>219544</td>
      <td>309085</td>
      <td>310365</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>AURORA AVE N BETWEEN N 85TH ST AND N 86TH ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>From same direction - both going straight - bo...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>2</td>
      <td>-122.306689</td>
      <td>47.683047</td>
      <td>219545</td>
      <td>311280</td>
      <td>312640</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>24760.0</td>
      <td>20TH AVE NE AND NE 75TH ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>11</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>From opposite direction - one left turn - one ...</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>2</td>
      <td>-122.355317</td>
      <td>47.678734</td>
      <td>219546</td>
      <td>309514</td>
      <td>310794</td>
      <td>Matched</td>
      <td>Intersection</td>
      <td>24349.0</td>
      <td>GREENWOOD AVE N AND N 68TH ST</td>
      <td>...</td>
      <td>At Intersection (intersection related)</td>
      <td>51</td>
      <td>PEDALCYCLIST STRUCK MOTOR VEHICLE FRONT END AT...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>NaN</td>
      <td>Vehicle Strikes Pedalcyclist</td>
      <td>4308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>1</td>
      <td>-122.289360</td>
      <td>47.611017</td>
      <td>219547</td>
      <td>308220</td>
      <td>309500</td>
      <td>Matched</td>
      <td>Block</td>
      <td>NaN</td>
      <td>34TH AVE BETWEEN E MARION ST AND E SPRING ST</td>
      <td>...</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>14</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>NaN</td>
      <td>From same direction - both going straight - on...</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 26 columns</p>
</div>




```python
df.shape
```




    (194673, 26)




```python
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
```

    SEVERITYCODE
    False    194673
    Name: SEVERITYCODE, dtype: int64
    
    X
    False    189339
    True       5334
    Name: X, dtype: int64
    
    Y
    False    189339
    True       5334
    Name: Y, dtype: int64
    
    OBJECTID
    False    194673
    Name: OBJECTID, dtype: int64
    
    INCKEY
    False    194673
    Name: INCKEY, dtype: int64
    
    COLDETKEY
    False    194673
    Name: COLDETKEY, dtype: int64
    
    REPORTNO
    False    194673
    Name: REPORTNO, dtype: int64
    
    STATUS
    False    194673
    Name: STATUS, dtype: int64
    
    ADDRTYPE
    False    192747
    True       1926
    Name: ADDRTYPE, dtype: int64
    
    INTKEY
    True     129603
    False     65070
    Name: INTKEY, dtype: int64
    
    LOCATION
    False    191996
    True       2677
    Name: LOCATION, dtype: int64
    
    EXCEPTRSNCODE
    True     109862
    False     84811
    Name: EXCEPTRSNCODE, dtype: int64
    
    EXCEPTRSNDESC
    True     189035
    False      5638
    Name: EXCEPTRSNDESC, dtype: int64
    
    SEVERITYCODE.1
    False    194673
    Name: SEVERITYCODE.1, dtype: int64
    
    SEVERITYDESC
    False    194673
    Name: SEVERITYDESC, dtype: int64
    
    COLLISIONTYPE
    False    189769
    True       4904
    Name: COLLISIONTYPE, dtype: int64
    
    PERSONCOUNT
    False    194673
    Name: PERSONCOUNT, dtype: int64
    
    PEDCOUNT
    False    194673
    Name: PEDCOUNT, dtype: int64
    
    PEDCYLCOUNT
    False    194673
    Name: PEDCYLCOUNT, dtype: int64
    
    VEHCOUNT
    False    194673
    Name: VEHCOUNT, dtype: int64
    
    INCDATE
    False    194673
    Name: INCDATE, dtype: int64
    
    INCDTTM
    False    194673
    Name: INCDTTM, dtype: int64
    
    JUNCTIONTYPE
    False    188344
    True       6329
    Name: JUNCTIONTYPE, dtype: int64
    
    SDOT_COLCODE
    False    194673
    Name: SDOT_COLCODE, dtype: int64
    
    SDOT_COLDESC
    False    194673
    Name: SDOT_COLDESC, dtype: int64
    
    INATTENTIONIND
    True     164868
    False     29805
    Name: INATTENTIONIND, dtype: int64
    
    UNDERINFL
    False    189789
    True       4884
    Name: UNDERINFL, dtype: int64
    
    WEATHER
    False    189592
    True       5081
    Name: WEATHER, dtype: int64
    
    ROADCOND
    False    189661
    True       5012
    Name: ROADCOND, dtype: int64
    
    LIGHTCOND
    False    189503
    True       5170
    Name: LIGHTCOND, dtype: int64
    
    PEDROWNOTGRNT
    True     190006
    False      4667
    Name: PEDROWNOTGRNT, dtype: int64
    
    SDOTCOLNUM
    False    114936
    True      79737
    Name: SDOTCOLNUM, dtype: int64
    
    SPEEDING
    True     185340
    False      9333
    Name: SPEEDING, dtype: int64
    
    ST_COLCODE
    False    194655
    True         18
    Name: ST_COLCODE, dtype: int64
    
    ST_COLDESC
    False    189769
    True       4904
    Name: ST_COLDESC, dtype: int64
    
    SEGLANEKEY
    False    194673
    Name: SEGLANEKEY, dtype: int64
    
    CROSSWALKKEY
    False    194673
    Name: CROSSWALKKEY, dtype: int64
    
    HITPARKEDCAR
    False    194673
    Name: HITPARKEDCAR, dtype: int64
    



```python
num= df.select_dtypes(include=['float','int']).copy()
num
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
      <th>SEVERITYCODE</th>
      <th>X</th>
      <th>Y</th>
      <th>OBJECTID</th>
      <th>INCKEY</th>
      <th>COLDETKEY</th>
      <th>INTKEY</th>
      <th>PERSONCOUNT</th>
      <th>PEDCOUNT</th>
      <th>PEDCYLCOUNT</th>
      <th>VEHCOUNT</th>
      <th>SDOT_COLCODE</th>
      <th>SDOTCOLNUM</th>
      <th>SEGLANEKEY</th>
      <th>CROSSWALKKEY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>-122.323148</td>
      <td>47.703140</td>
      <td>1</td>
      <td>1307</td>
      <td>1307</td>
      <td>37475.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-122.347294</td>
      <td>47.647172</td>
      <td>2</td>
      <td>52200</td>
      <td>52200</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>16</td>
      <td>6354039.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-122.334540</td>
      <td>47.607871</td>
      <td>3</td>
      <td>26700</td>
      <td>26700</td>
      <td>NaN</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
      <td>4323031.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-122.334803</td>
      <td>47.604803</td>
      <td>4</td>
      <td>1144</td>
      <td>1144</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>-122.306426</td>
      <td>47.545739</td>
      <td>5</td>
      <td>17700</td>
      <td>17700</td>
      <td>34387.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>4028032.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-122.387598</td>
      <td>47.690575</td>
      <td>6</td>
      <td>320840</td>
      <td>322340</td>
      <td>36974.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>-122.338485</td>
      <td>47.618534</td>
      <td>7</td>
      <td>83300</td>
      <td>83300</td>
      <td>29510.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>8344002.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>9</td>
      <td>330897</td>
      <td>332397</td>
      <td>29745.0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>51</td>
      <td>NaN</td>
      <td>6855</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>-122.335930</td>
      <td>47.611904</td>
      <td>10</td>
      <td>63400</td>
      <td>63400</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>6166014.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>-122.384700</td>
      <td>47.528475</td>
      <td>12</td>
      <td>58600</td>
      <td>58600</td>
      <td>34679.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>6079001.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>48900</td>
      <td>48900</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>6145029.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>-122.333831</td>
      <td>47.547371</td>
      <td>15</td>
      <td>38800</td>
      <td>38800</td>
      <td>33194.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>5223041.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>-122.356273</td>
      <td>47.571375</td>
      <td>16</td>
      <td>2771</td>
      <td>2771</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>3137016.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>-122.323966</td>
      <td>47.606374</td>
      <td>17</td>
      <td>32800</td>
      <td>32800</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>13</td>
      <td>5356027.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2</td>
      <td>-122.317414</td>
      <td>47.664028</td>
      <td>19</td>
      <td>1212</td>
      <td>1212</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>-122.337663</td>
      <td>47.617510</td>
      <td>20</td>
      <td>330878</td>
      <td>332378</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>-122.344539</td>
      <td>47.692012</td>
      <td>21</td>
      <td>46300</td>
      <td>46300</td>
      <td>37365.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>5182022.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>23000</td>
      <td>23000</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>4016025.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>-122.328270</td>
      <td>47.571420</td>
      <td>24</td>
      <td>330833</td>
      <td>332333</td>
      <td>NaN</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>-122.383802</td>
      <td>47.583715</td>
      <td>25</td>
      <td>97100</td>
      <td>97100</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>9152035.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>-122.292403</td>
      <td>47.732847</td>
      <td>26</td>
      <td>1347</td>
      <td>1347</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>-122.313786</td>
      <td>47.708535</td>
      <td>28</td>
      <td>1323</td>
      <td>1323</td>
      <td>36505.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>-122.318169</td>
      <td>47.615837</td>
      <td>29</td>
      <td>80000</td>
      <td>80000</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>8200013.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>-122.337486</td>
      <td>47.589746</td>
      <td>31</td>
      <td>28700</td>
      <td>28700</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>28</td>
      <td>4256026.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>-122.279658</td>
      <td>47.553405</td>
      <td>33</td>
      <td>1268</td>
      <td>1268</td>
      <td>33499.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>-122.312857</td>
      <td>47.599218</td>
      <td>34</td>
      <td>320932</td>
      <td>322432</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>-122.330730</td>
      <td>47.615450</td>
      <td>35</td>
      <td>113300</td>
      <td>113300</td>
      <td>29865.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>10317016.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>-122.385859</td>
      <td>47.581191</td>
      <td>36</td>
      <td>64700</td>
      <td>64700</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7204015.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>-122.304990</td>
      <td>47.611474</td>
      <td>37</td>
      <td>1083</td>
      <td>1083</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38</td>
      <td>61500</td>
      <td>61500</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>16</td>
      <td>6320040.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>1</td>
      <td>-122.345863</td>
      <td>47.612991</td>
      <td>219510</td>
      <td>307577</td>
      <td>308857</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>1</td>
      <td>-122.330298</td>
      <td>47.603233</td>
      <td>219511</td>
      <td>312117</td>
      <td>313537</td>
      <td>30479.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>2</td>
      <td>-122.328079</td>
      <td>47.604161</td>
      <td>219512</td>
      <td>307692</td>
      <td>308972</td>
      <td>29973.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>1</td>
      <td>-122.320756</td>
      <td>47.608656</td>
      <td>219513</td>
      <td>312289</td>
      <td>313709</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>2</td>
      <td>-122.355449</td>
      <td>47.704720</td>
      <td>219514</td>
      <td>308575</td>
      <td>309855</td>
      <td>NaN</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>1</td>
      <td>-122.357624</td>
      <td>47.635956</td>
      <td>219515</td>
      <td>307792</td>
      <td>309072</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>2</td>
      <td>-122.302382</td>
      <td>47.626759</td>
      <td>219516</td>
      <td>307985</td>
      <td>309265</td>
      <td>29328.0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>2</td>
      <td>-122.355298</td>
      <td>47.684382</td>
      <td>219517</td>
      <td>310417</td>
      <td>311717</td>
      <td>NaN</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>1</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>219519</td>
      <td>309594</td>
      <td>310874</td>
      <td>29745.0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>NaN</td>
      <td>0</td>
      <td>522257</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>1</td>
      <td>-122.335527</td>
      <td>47.617434</td>
      <td>219520</td>
      <td>308698</td>
      <td>309978</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>2</td>
      <td>-122.322097</td>
      <td>47.649615</td>
      <td>219521</td>
      <td>308342</td>
      <td>309622</td>
      <td>NaN</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>1</td>
      <td>-122.312679</td>
      <td>47.719414</td>
      <td>219522</td>
      <td>311117</td>
      <td>312437</td>
      <td>36427.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>2</td>
      <td>-122.380016</td>
      <td>47.664879</td>
      <td>219523</td>
      <td>310911</td>
      <td>312231</td>
      <td>25174.0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>NaN</td>
      <td>0</td>
      <td>650595</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>2</td>
      <td>-122.340474</td>
      <td>47.614496</td>
      <td>219524</td>
      <td>312179</td>
      <td>313599</td>
      <td>29545.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>NaN</td>
      <td>0</td>
      <td>523322</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>1</td>
      <td>-122.337137</td>
      <td>47.610709</td>
      <td>219525</td>
      <td>307834</td>
      <td>309114</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>1</td>
      <td>-122.302554</td>
      <td>47.584099</td>
      <td>219528</td>
      <td>311697</td>
      <td>313057</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>1</td>
      <td>-122.298955</td>
      <td>47.717456</td>
      <td>219529</td>
      <td>308019</td>
      <td>309299</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>219531</td>
      <td>308990</td>
      <td>310270</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>29</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>2</td>
      <td>-122.306092</td>
      <td>47.617881</td>
      <td>219532</td>
      <td>308532</td>
      <td>309812</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>1</td>
      <td>-122.337880</td>
      <td>47.625793</td>
      <td>219535</td>
      <td>307802</td>
      <td>309082</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>2</td>
      <td>-122.299160</td>
      <td>47.579673</td>
      <td>219536</td>
      <td>309335</td>
      <td>310615</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>1</td>
      <td>-122.325887</td>
      <td>47.643191</td>
      <td>219537</td>
      <td>309222</td>
      <td>310502</td>
      <td>28300.0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>1</td>
      <td>-122.304217</td>
      <td>47.669537</td>
      <td>219538</td>
      <td>308480</td>
      <td>309760</td>
      <td>26005.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>2</td>
      <td>-122.344569</td>
      <td>47.694547</td>
      <td>219539</td>
      <td>309170</td>
      <td>310450</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>1</td>
      <td>-122.361672</td>
      <td>47.556722</td>
      <td>219541</td>
      <td>307804</td>
      <td>309084</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>28</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>2</td>
      <td>-122.290826</td>
      <td>47.565408</td>
      <td>219543</td>
      <td>309534</td>
      <td>310814</td>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>1</td>
      <td>-122.344526</td>
      <td>47.690924</td>
      <td>219544</td>
      <td>309085</td>
      <td>310365</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>2</td>
      <td>-122.306689</td>
      <td>47.683047</td>
      <td>219545</td>
      <td>311280</td>
      <td>312640</td>
      <td>24760.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>2</td>
      <td>-122.355317</td>
      <td>47.678734</td>
      <td>219546</td>
      <td>309514</td>
      <td>310794</td>
      <td>24349.0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>51</td>
      <td>NaN</td>
      <td>4308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>1</td>
      <td>-122.289360</td>
      <td>47.611017</td>
      <td>219547</td>
      <td>308220</td>
      <td>309500</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 15 columns</p>
</div>




```python
avg_Y = num["Y"].astype("float").mean(axis=0)
print("Average of Y:", avg_Y)
num["Y"].replace(np.nan, avg_Y, inplace=True)

avg_X = num["X"].astype("float").mean(axis=0)
print("Average of X:", avg_Y)
num["X"].replace(np.nan, avg_X, inplace=True)

avg_INT = num["INTKEY"].astype("float").mean(axis=0)
print("Average of INTKEY:", avg_INT)
num["INTKEY"].replace(np.nan, avg_INT, inplace=True)

avg_SDO = num["SDOTCOLNUM"].astype("float").mean(axis=0)
print("Average of SDOTCOLNUM:", avg_SDO)
num["SDOTCOLNUM"].replace(np.nan, avg_SDO, inplace=True)
num
```

    Average of Y: 47.619542517688615
    Average of X: 47.619542517688615
    Average of INTKEY: 37558.45057630244
    Average of SDOTCOLNUM: 7972521.3371441495





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
      <th>SEVERITYCODE</th>
      <th>X</th>
      <th>Y</th>
      <th>OBJECTID</th>
      <th>INCKEY</th>
      <th>COLDETKEY</th>
      <th>INTKEY</th>
      <th>PERSONCOUNT</th>
      <th>PEDCOUNT</th>
      <th>PEDCYLCOUNT</th>
      <th>VEHCOUNT</th>
      <th>SDOT_COLCODE</th>
      <th>SDOTCOLNUM</th>
      <th>SEGLANEKEY</th>
      <th>CROSSWALKKEY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>-122.323148</td>
      <td>47.703140</td>
      <td>1</td>
      <td>1307</td>
      <td>1307</td>
      <td>37475.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-122.347294</td>
      <td>47.647172</td>
      <td>2</td>
      <td>52200</td>
      <td>52200</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>16</td>
      <td>6.354039e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-122.334540</td>
      <td>47.607871</td>
      <td>3</td>
      <td>26700</td>
      <td>26700</td>
      <td>37558.450576</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
      <td>4.323031e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-122.334803</td>
      <td>47.604803</td>
      <td>4</td>
      <td>1144</td>
      <td>1144</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>-122.306426</td>
      <td>47.545739</td>
      <td>5</td>
      <td>17700</td>
      <td>17700</td>
      <td>34387.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>4.028032e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-122.387598</td>
      <td>47.690575</td>
      <td>6</td>
      <td>320840</td>
      <td>322340</td>
      <td>36974.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>-122.338485</td>
      <td>47.618534</td>
      <td>7</td>
      <td>83300</td>
      <td>83300</td>
      <td>29510.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>8.344002e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>9</td>
      <td>330897</td>
      <td>332397</td>
      <td>29745.000000</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>51</td>
      <td>7.972521e+06</td>
      <td>6855</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>-122.335930</td>
      <td>47.611904</td>
      <td>10</td>
      <td>63400</td>
      <td>63400</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>6.166014e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>-122.384700</td>
      <td>47.528475</td>
      <td>12</td>
      <td>58600</td>
      <td>58600</td>
      <td>34679.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>6.079001e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>14</td>
      <td>48900</td>
      <td>48900</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>6.145029e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>-122.333831</td>
      <td>47.547371</td>
      <td>15</td>
      <td>38800</td>
      <td>38800</td>
      <td>33194.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>5.223041e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>-122.356273</td>
      <td>47.571375</td>
      <td>16</td>
      <td>2771</td>
      <td>2771</td>
      <td>37558.450576</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>3.137016e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>-122.323966</td>
      <td>47.606374</td>
      <td>17</td>
      <td>32800</td>
      <td>32800</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>13</td>
      <td>5.356027e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2</td>
      <td>-122.317414</td>
      <td>47.664028</td>
      <td>19</td>
      <td>1212</td>
      <td>1212</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>-122.337663</td>
      <td>47.617510</td>
      <td>20</td>
      <td>330878</td>
      <td>332378</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>-122.344539</td>
      <td>47.692012</td>
      <td>21</td>
      <td>46300</td>
      <td>46300</td>
      <td>37365.000000</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>5.182022e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>23</td>
      <td>23000</td>
      <td>23000</td>
      <td>37558.450576</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>4.016025e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>-122.328270</td>
      <td>47.571420</td>
      <td>24</td>
      <td>330833</td>
      <td>332333</td>
      <td>37558.450576</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>-122.383802</td>
      <td>47.583715</td>
      <td>25</td>
      <td>97100</td>
      <td>97100</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>9.152035e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>-122.292403</td>
      <td>47.732847</td>
      <td>26</td>
      <td>1347</td>
      <td>1347</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>-122.313786</td>
      <td>47.708535</td>
      <td>28</td>
      <td>1323</td>
      <td>1323</td>
      <td>36505.000000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>-122.318169</td>
      <td>47.615837</td>
      <td>29</td>
      <td>80000</td>
      <td>80000</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>8.200013e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>-122.337486</td>
      <td>47.589746</td>
      <td>31</td>
      <td>28700</td>
      <td>28700</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>28</td>
      <td>4.256026e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>-122.279658</td>
      <td>47.553405</td>
      <td>33</td>
      <td>1268</td>
      <td>1268</td>
      <td>33499.000000</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>-122.312857</td>
      <td>47.599218</td>
      <td>34</td>
      <td>320932</td>
      <td>322432</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>-122.330730</td>
      <td>47.615450</td>
      <td>35</td>
      <td>113300</td>
      <td>113300</td>
      <td>29865.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>1.031702e+07</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>-122.385859</td>
      <td>47.581191</td>
      <td>36</td>
      <td>64700</td>
      <td>64700</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.204015e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>-122.304990</td>
      <td>47.611474</td>
      <td>37</td>
      <td>1083</td>
      <td>1083</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>38</td>
      <td>61500</td>
      <td>61500</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>16</td>
      <td>6.320040e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>1</td>
      <td>-122.345863</td>
      <td>47.612991</td>
      <td>219510</td>
      <td>307577</td>
      <td>308857</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>1</td>
      <td>-122.330298</td>
      <td>47.603233</td>
      <td>219511</td>
      <td>312117</td>
      <td>313537</td>
      <td>30479.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>2</td>
      <td>-122.328079</td>
      <td>47.604161</td>
      <td>219512</td>
      <td>307692</td>
      <td>308972</td>
      <td>29973.000000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>1</td>
      <td>-122.320756</td>
      <td>47.608656</td>
      <td>219513</td>
      <td>312289</td>
      <td>313709</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>2</td>
      <td>-122.355449</td>
      <td>47.704720</td>
      <td>219514</td>
      <td>308575</td>
      <td>309855</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>1</td>
      <td>-122.357624</td>
      <td>47.635956</td>
      <td>219515</td>
      <td>307792</td>
      <td>309072</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>2</td>
      <td>-122.302382</td>
      <td>47.626759</td>
      <td>219516</td>
      <td>307985</td>
      <td>309265</td>
      <td>29328.000000</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>2</td>
      <td>-122.355298</td>
      <td>47.684382</td>
      <td>219517</td>
      <td>310417</td>
      <td>311717</td>
      <td>37558.450576</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>1</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>219519</td>
      <td>309594</td>
      <td>310874</td>
      <td>29745.000000</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>522257</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>1</td>
      <td>-122.335527</td>
      <td>47.617434</td>
      <td>219520</td>
      <td>308698</td>
      <td>309978</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>2</td>
      <td>-122.322097</td>
      <td>47.649615</td>
      <td>219521</td>
      <td>308342</td>
      <td>309622</td>
      <td>37558.450576</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>1</td>
      <td>-122.312679</td>
      <td>47.719414</td>
      <td>219522</td>
      <td>311117</td>
      <td>312437</td>
      <td>36427.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>2</td>
      <td>-122.380016</td>
      <td>47.664879</td>
      <td>219523</td>
      <td>310911</td>
      <td>312231</td>
      <td>25174.000000</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>650595</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>2</td>
      <td>-122.340474</td>
      <td>47.614496</td>
      <td>219524</td>
      <td>312179</td>
      <td>313599</td>
      <td>29545.000000</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>523322</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>1</td>
      <td>-122.337137</td>
      <td>47.610709</td>
      <td>219525</td>
      <td>307834</td>
      <td>309114</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>1</td>
      <td>-122.302554</td>
      <td>47.584099</td>
      <td>219528</td>
      <td>311697</td>
      <td>313057</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>1</td>
      <td>-122.298955</td>
      <td>47.717456</td>
      <td>219529</td>
      <td>308019</td>
      <td>309299</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>2</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>219531</td>
      <td>308990</td>
      <td>310270</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>29</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>2</td>
      <td>-122.306092</td>
      <td>47.617881</td>
      <td>219532</td>
      <td>308532</td>
      <td>309812</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>1</td>
      <td>-122.337880</td>
      <td>47.625793</td>
      <td>219535</td>
      <td>307802</td>
      <td>309082</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>2</td>
      <td>-122.299160</td>
      <td>47.579673</td>
      <td>219536</td>
      <td>309335</td>
      <td>310615</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>1</td>
      <td>-122.325887</td>
      <td>47.643191</td>
      <td>219537</td>
      <td>309222</td>
      <td>310502</td>
      <td>28300.000000</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>1</td>
      <td>-122.304217</td>
      <td>47.669537</td>
      <td>219538</td>
      <td>308480</td>
      <td>309760</td>
      <td>26005.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>2</td>
      <td>-122.344569</td>
      <td>47.694547</td>
      <td>219539</td>
      <td>309170</td>
      <td>310450</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>1</td>
      <td>-122.361672</td>
      <td>47.556722</td>
      <td>219541</td>
      <td>307804</td>
      <td>309084</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>28</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>2</td>
      <td>-122.290826</td>
      <td>47.565408</td>
      <td>219543</td>
      <td>309534</td>
      <td>310814</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>1</td>
      <td>-122.344526</td>
      <td>47.690924</td>
      <td>219544</td>
      <td>309085</td>
      <td>310365</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>2</td>
      <td>-122.306689</td>
      <td>47.683047</td>
      <td>219545</td>
      <td>311280</td>
      <td>312640</td>
      <td>24760.000000</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>2</td>
      <td>-122.355317</td>
      <td>47.678734</td>
      <td>219546</td>
      <td>309514</td>
      <td>310794</td>
      <td>24349.000000</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>51</td>
      <td>7.972521e+06</td>
      <td>4308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>1</td>
      <td>-122.289360</td>
      <td>47.611017</td>
      <td>219547</td>
      <td>308220</td>
      <td>309500</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>7.972521e+06</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 15 columns</p>
</div>




```python
num.shape
```




    (194673, 15)




```python
num.dtypes
```




    SEVERITYCODE      int64
    X               float64
    Y               float64
    OBJECTID          int64
    INCKEY            int64
    COLDETKEY         int64
    INTKEY          float64
    PERSONCOUNT       int64
    PEDCOUNT          int64
    PEDCYLCOUNT       int64
    VEHCOUNT          int64
    SDOT_COLCODE      int64
    SDOTCOLNUM      float64
    SEGLANEKEY        int64
    CROSSWALKKEY      int64
    dtype: object




```python
obj= df.select_dtypes(include=['object']).copy()
obj
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
      <th>STATUS</th>
      <th>ADDRTYPE</th>
      <th>LOCATION</th>
      <th>SEVERITYDESC</th>
      <th>COLLISIONTYPE</th>
      <th>JUNCTIONTYPE</th>
      <th>SDOT_COLDESC</th>
      <th>WEATHER</th>
      <th>ROADCOND</th>
      <th>LIGHTCOND</th>
      <th>ST_COLDESC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>5TH AVE NE AND NE 103RD ST</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Matched</td>
      <td>Block</td>
      <td>AURORA BR BETWEEN RAYE ST AND BRIDGE WAY N</td>
      <td>Property Damage Only Collision</td>
      <td>Sideswipe</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, LEFT SIDE ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - both going straight - bo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Matched</td>
      <td>Block</td>
      <td>4TH AVE BETWEEN SENECA ST AND UNIVERSITY ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Matched</td>
      <td>Block</td>
      <td>2ND AVE BETWEEN MARION ST AND MADISON ST</td>
      <td>Property Damage Only Collision</td>
      <td>Other</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From same direction - all others</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>SWIFT AVE S AND SWIFT AV OFF RP</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>24TH AVE NW AND NW 85TH ST</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>DENNY WAY AND WESTLAKE AVE</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>BROADWAY AND E PIKE ST</td>
      <td>Injury Collision</td>
      <td>Cycles</td>
      <td>At Intersection (intersection related)</td>
      <td>PEDALCYCLIST STRUCK MOTOR VEHICLE FRONT END AT...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Vehicle Strikes Pedalcyclist</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Matched</td>
      <td>Block</td>
      <td>PINE ST BETWEEN 5TH AVE AND 6TH AVE</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>41ST AVE SW AND SW THISTLE ST</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Matched</td>
      <td>Alley</td>
      <td>NaN</td>
      <td>Property Damage Only Collision</td>
      <td>Other</td>
      <td>Driveway Junction</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One car leaving driveway access</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>1ST AV S BR NB AND EAST MARGINAL WAY S</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Unmatched</td>
      <td>Block</td>
      <td>SW SPOKANE ST BETWEEN SW SPOKANE W BR AND TERM...</td>
      <td>Property Damage Only Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Matched</td>
      <td>Block</td>
      <td>TERRY AVE BETWEEN JAMES ST AND CHERRY ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, LEFT SIDE ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - No Street Lights</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Matched</td>
      <td>Block</td>
      <td>ROOSEVELT WAY NE BETWEEN NE 47TH ST AND NE 50T...</td>
      <td>Injury Collision</td>
      <td>Head On</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>From opposite direction - all others</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Unmatched</td>
      <td>Block</td>
      <td>9TH AVE BETWEEN LENORA ST AND BLANCHARD ST</td>
      <td>Property Damage Only Collision</td>
      <td>NaN</td>
      <td>Driveway Junction</td>
      <td>MOTOR VEHICLE STRUCK OBJECT IN ROAD</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>AURORA AVE N AND N 87TH ST</td>
      <td>Injury Collision</td>
      <td>Left Turn</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From opposite direction - one left turn - one ...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Matched</td>
      <td>Block</td>
      <td>BATTERY ST TUN ON RP BETWEEN BELL ST AND ALASK...</td>
      <td>Property Damage Only Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (but intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Matched</td>
      <td>Block</td>
      <td>S SPOKANE SR ST BETWEEN 4TH AVE S AND 5TH AVE S</td>
      <td>Injury Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Matched</td>
      <td>Block</td>
      <td>41ST AVE SW BETWEEN SW WALKER ST AND SW COLLEG...</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Unknown</td>
      <td>Dry</td>
      <td>Unknown</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Matched</td>
      <td>Block</td>
      <td>LAKE CITY WAY NE BETWEEN NE 143RD ST AND NE 14...</td>
      <td>Injury Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>14TH AVE NE AND NE NORTHGATE WAY</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Matched</td>
      <td>Block</td>
      <td>11TH AVE BETWEEN E PINE ST AND E OLIVE ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Matched</td>
      <td>Block</td>
      <td>ALASKAN WY VI SB BETWEEN ALASKAN WY VI SB EFR ...</td>
      <td>Property Damage Only Collision</td>
      <td>Other</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE RAN OFF ROAD - HIT FIXED OBJECT</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Other object</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>RAINIER AVE S AND S BRANDON ST</td>
      <td>Injury Collision</td>
      <td>Rear Ended</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Matched</td>
      <td>Block</td>
      <td>S JACKSON ST BETWEEN 14TH AVE S AND 16TH AVE S</td>
      <td>Injury Collision</td>
      <td>Cycles</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK PEDALCYCLIST, FRONT END A...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Vehicle Strikes Pedalcyclist</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>BOREN AVE AND OLIVE WAY</td>
      <td>Property Damage Only Collision</td>
      <td>Left Turn</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Unknown</td>
      <td>From opposite direction - one left turn - one ...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Matched</td>
      <td>Block</td>
      <td>SW ADMIRAL WAY BETWEEN 42ND AVE SW AND CALIFOR...</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>Driveway Junction</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Matched</td>
      <td>Block</td>
      <td>21ST AVE BETWEEN E MARION ST AND E UNION ST</td>
      <td>Property Damage Only Collision</td>
      <td>Other</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From same direction - all others</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Matched</td>
      <td>Block</td>
      <td>1ST AVE S BETWEEN 1ST AVS ON N RP AND S ROYAL ...</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, LEFT SIDE ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>Matched</td>
      <td>Block</td>
      <td>1ST AVE BETWEEN BLANCHARD ST AND BELL ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>One car leaving parked position</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>4TH AVE AND JAMES ST</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>6TH AVE AND JAMES ST</td>
      <td>Injury Collision</td>
      <td>Right Turn</td>
      <td>At Intersection (but not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - one right turn - one str...</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>Unmatched</td>
      <td>Block</td>
      <td>BROADWAY BETWEEN CHERRY ST AND E COLUMBIA ST</td>
      <td>Property Damage Only Collision</td>
      <td>NaN</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>Matched</td>
      <td>Block</td>
      <td>GREENWOOD AVE N BETWEEN N 104TH ST AND HOLMAN ...</td>
      <td>Injury Collision</td>
      <td>Pedestrian</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dusk</td>
      <td>Vehicle going straight hits pedestrian</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>Matched</td>
      <td>Block</td>
      <td>W HOWE ST BETWEEN QUEEN ANNE AVE N AND 1ST AVE W</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>23RD AVE E AND E ALOHA ST</td>
      <td>Injury Collision</td>
      <td>Left Turn</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>From opposite direction - one left turn - one ...</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>Matched</td>
      <td>Block</td>
      <td>GREENWOOD AVE N BETWEEN N 76TH ST AND N 77TH ST</td>
      <td>Injury Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>BROADWAY AND E PIKE ST</td>
      <td>Property Damage Only Collision</td>
      <td>Pedestrian</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHCILE STRUCK PEDESTRIAN</td>
      <td>Clear</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>Vehicle going straight hits pedestrian</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>Matched</td>
      <td>Block</td>
      <td>TERRY AVE BETWEEN VIRGINIA ST AND LENORA ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Dusk</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>Matched</td>
      <td>Block</td>
      <td>HARVARD AVE E BETWEEN E ALLISON ST AND EASTLAK...</td>
      <td>Injury Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>Unmatched</td>
      <td>Intersection</td>
      <td>15TH AVE NE AND NE 125TH ST</td>
      <td>Property Damage Only Collision</td>
      <td>NaN</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>Unmatched</td>
      <td>Intersection</td>
      <td>LEARY AVE NW AND NW DOCK PL</td>
      <td>Injury Collision</td>
      <td>NaN</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHCILE STRUCK PEDESTRIAN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>5TH AVE AND LENORA ST</td>
      <td>Injury Collision</td>
      <td>Pedestrian</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHCILE STRUCK PEDESTRIAN</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Vehicle turning left hits pedestrian</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>Matched</td>
      <td>Block</td>
      <td>4TH AVE BETWEEN PIKE ST AND PINE ST</td>
      <td>Property Damage Only Collision</td>
      <td>Sideswipe</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>NOT ENOUGH INFORMATION / NOT APPLICABLE</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - both going straight - bo...</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>Matched</td>
      <td>Block</td>
      <td>RAINIER AVE S BETWEEN 23RD AVE S AND S WALKER ST</td>
      <td>Property Damage Only Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>Matched</td>
      <td>Block</td>
      <td>NE 123RD ST BETWEEN HIRAM PL NE AND 28TH AVE NE</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>NaN</td>
      <td>NOT ENOUGH INFORMATION / NOT APPLICABLE</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>Matched</td>
      <td>Block</td>
      <td>BATTERY ST TUNNEL NB BETWEEN ALASKAN WY VI NB ...</td>
      <td>Injury Collision</td>
      <td>Other</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE OVERTURNED IN ROAD</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Other</td>
      <td>Vehicle overturned</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>Matched</td>
      <td>Block</td>
      <td>20TH AVE BETWEEN E MADISON ST AND E DENNY WAY</td>
      <td>Injury Collision</td>
      <td>Left Turn</td>
      <td>Mid-Block (but intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From opposite direction - one left turn - one ...</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>Matched</td>
      <td>Block</td>
      <td>VALLEY ST BETWEEN WESTLAKE AVE N AND TERRY AVE N</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (but intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>Matched</td>
      <td>Block</td>
      <td>RAINIER AVE S BETWEEN S BAYVIEW ST AND S MCCLE...</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>EASTLAKE AVE E AND E ROANOKE ST</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>NE PARK RD AND NE RAVENNA WB BV</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>Matched</td>
      <td>Block</td>
      <td>AURORA AVE N BETWEEN N 90TH ST AND N 91ST ST</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>Matched</td>
      <td>Block</td>
      <td>PUGET BLVD SW BETWEEN SW HUDSON ST AND DEAD END 1</td>
      <td>Property Damage Only Collision</td>
      <td>Other</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE RAN OFF ROAD - HIT FIXED OBJECT</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>Fixed object</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>Matched</td>
      <td>Block</td>
      <td>34TH AVE S BETWEEN S DAKOTA ST AND S GENESEE ST</td>
      <td>Injury Collision</td>
      <td>Head On</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From opposite direction - both moving - head-on</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>Matched</td>
      <td>Block</td>
      <td>AURORA AVE N BETWEEN N 85TH ST AND N 86TH ST</td>
      <td>Property Damage Only Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>From same direction - both going straight - bo...</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>20TH AVE NE AND NE 75TH ST</td>
      <td>Injury Collision</td>
      <td>Left Turn</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From opposite direction - one left turn - one ...</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>GREENWOOD AVE N AND N 68TH ST</td>
      <td>Injury Collision</td>
      <td>Cycles</td>
      <td>At Intersection (intersection related)</td>
      <td>PEDALCYCLIST STRUCK MOTOR VEHICLE FRONT END AT...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>Vehicle Strikes Pedalcyclist</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>Matched</td>
      <td>Block</td>
      <td>34TH AVE BETWEEN E MARION ST AND E SPRING ST</td>
      <td>Property Damage Only Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 11 columns</p>
</div>




```python
obj.shape
```




    (194673, 11)




```python
obj.apply(lambda x: pd.value_counts(x).idxmax(x)[0])
```




    STATUS           M
    ADDRTYPE         B
    LOCATION         B
    SEVERITYDESC     P
    COLLISIONTYPE    P
    JUNCTIONTYPE     M
    SDOT_COLDESC     M
    WEATHER          C
    ROADCOND         D
    LIGHTCOND        D
    ST_COLDESC       O
    dtype: object




```python
obj["ADDRTYPE"].replace(np.nan, "Block", inplace=True) 
obj["COLLISIONTYPE"].replace(np.nan, "Parked Car", inplace=True) 
obj["JUNCTIONTYPE"].replace(np.nan, "Mid-Block (not related to intersection)", inplace=True) 
obj["WEATHER"].replace(np.nan, "Clear", inplace=True)  
obj["ROADCOND"].replace(np.nan, "Dry", inplace=True)
obj["LIGHTCOND"].replace(np.nan, "Daylight", inplace=True)  
obj["ST_COLDESC"].replace(np.nan, "One parked--one moving", inplace=True)

obj

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
      <th>STATUS</th>
      <th>ADDRTYPE</th>
      <th>LOCATION</th>
      <th>SEVERITYDESC</th>
      <th>COLLISIONTYPE</th>
      <th>JUNCTIONTYPE</th>
      <th>SDOT_COLDESC</th>
      <th>WEATHER</th>
      <th>ROADCOND</th>
      <th>LIGHTCOND</th>
      <th>ST_COLDESC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>5TH AVE NE AND NE 103RD ST</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Matched</td>
      <td>Block</td>
      <td>AURORA BR BETWEEN RAYE ST AND BRIDGE WAY N</td>
      <td>Property Damage Only Collision</td>
      <td>Sideswipe</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, LEFT SIDE ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - both going straight - bo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Matched</td>
      <td>Block</td>
      <td>4TH AVE BETWEEN SENECA ST AND UNIVERSITY ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Matched</td>
      <td>Block</td>
      <td>2ND AVE BETWEEN MARION ST AND MADISON ST</td>
      <td>Property Damage Only Collision</td>
      <td>Other</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From same direction - all others</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>SWIFT AVE S AND SWIFT AV OFF RP</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>24TH AVE NW AND NW 85TH ST</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>DENNY WAY AND WESTLAKE AVE</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>BROADWAY AND E PIKE ST</td>
      <td>Injury Collision</td>
      <td>Cycles</td>
      <td>At Intersection (intersection related)</td>
      <td>PEDALCYCLIST STRUCK MOTOR VEHICLE FRONT END AT...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Vehicle Strikes Pedalcyclist</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Matched</td>
      <td>Block</td>
      <td>PINE ST BETWEEN 5TH AVE AND 6TH AVE</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>41ST AVE SW AND SW THISTLE ST</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Matched</td>
      <td>Alley</td>
      <td>NaN</td>
      <td>Property Damage Only Collision</td>
      <td>Other</td>
      <td>Driveway Junction</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One car leaving driveway access</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>1ST AV S BR NB AND EAST MARGINAL WAY S</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Unmatched</td>
      <td>Block</td>
      <td>SW SPOKANE ST BETWEEN SW SPOKANE W BR AND TERM...</td>
      <td>Property Damage Only Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Matched</td>
      <td>Block</td>
      <td>TERRY AVE BETWEEN JAMES ST AND CHERRY ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, LEFT SIDE ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - No Street Lights</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Matched</td>
      <td>Block</td>
      <td>ROOSEVELT WAY NE BETWEEN NE 47TH ST AND NE 50T...</td>
      <td>Injury Collision</td>
      <td>Head On</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>From opposite direction - all others</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Unmatched</td>
      <td>Block</td>
      <td>9TH AVE BETWEEN LENORA ST AND BLANCHARD ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Driveway Junction</td>
      <td>MOTOR VEHICLE STRUCK OBJECT IN ROAD</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>AURORA AVE N AND N 87TH ST</td>
      <td>Injury Collision</td>
      <td>Left Turn</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From opposite direction - one left turn - one ...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Matched</td>
      <td>Block</td>
      <td>BATTERY ST TUN ON RP BETWEEN BELL ST AND ALASK...</td>
      <td>Property Damage Only Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (but intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Overcast</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Matched</td>
      <td>Block</td>
      <td>S SPOKANE SR ST BETWEEN 4TH AVE S AND 5TH AVE S</td>
      <td>Injury Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Matched</td>
      <td>Block</td>
      <td>41ST AVE SW BETWEEN SW WALKER ST AND SW COLLEG...</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Unknown</td>
      <td>Dry</td>
      <td>Unknown</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Matched</td>
      <td>Block</td>
      <td>LAKE CITY WAY NE BETWEEN NE 143RD ST AND NE 14...</td>
      <td>Injury Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>14TH AVE NE AND NE NORTHGATE WAY</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Matched</td>
      <td>Block</td>
      <td>11TH AVE BETWEEN E PINE ST AND E OLIVE ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Matched</td>
      <td>Block</td>
      <td>ALASKAN WY VI SB BETWEEN ALASKAN WY VI SB EFR ...</td>
      <td>Property Damage Only Collision</td>
      <td>Other</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE RAN OFF ROAD - HIT FIXED OBJECT</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Other object</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>RAINIER AVE S AND S BRANDON ST</td>
      <td>Injury Collision</td>
      <td>Rear Ended</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Matched</td>
      <td>Block</td>
      <td>S JACKSON ST BETWEEN 14TH AVE S AND 16TH AVE S</td>
      <td>Injury Collision</td>
      <td>Cycles</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK PEDALCYCLIST, FRONT END A...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Vehicle Strikes Pedalcyclist</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>BOREN AVE AND OLIVE WAY</td>
      <td>Property Damage Only Collision</td>
      <td>Left Turn</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Unknown</td>
      <td>From opposite direction - one left turn - one ...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Matched</td>
      <td>Block</td>
      <td>SW ADMIRAL WAY BETWEEN 42ND AVE SW AND CALIFOR...</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>Driveway Junction</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Matched</td>
      <td>Block</td>
      <td>21ST AVE BETWEEN E MARION ST AND E UNION ST</td>
      <td>Property Damage Only Collision</td>
      <td>Other</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From same direction - all others</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Matched</td>
      <td>Block</td>
      <td>1ST AVE S BETWEEN 1ST AVS ON N RP AND S ROYAL ...</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, LEFT SIDE ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>Matched</td>
      <td>Block</td>
      <td>1ST AVE BETWEEN BLANCHARD ST AND BELL ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>One car leaving parked position</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>4TH AVE AND JAMES ST</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>6TH AVE AND JAMES ST</td>
      <td>Injury Collision</td>
      <td>Right Turn</td>
      <td>At Intersection (but not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - one right turn - one str...</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>Unmatched</td>
      <td>Block</td>
      <td>BROADWAY BETWEEN CHERRY ST AND E COLUMBIA ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>Matched</td>
      <td>Block</td>
      <td>GREENWOOD AVE N BETWEEN N 104TH ST AND HOLMAN ...</td>
      <td>Injury Collision</td>
      <td>Pedestrian</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dusk</td>
      <td>Vehicle going straight hits pedestrian</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>Matched</td>
      <td>Block</td>
      <td>W HOWE ST BETWEEN QUEEN ANNE AVE N AND 1ST AVE W</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>23RD AVE E AND E ALOHA ST</td>
      <td>Injury Collision</td>
      <td>Left Turn</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dark - Street Lights On</td>
      <td>From opposite direction - one left turn - one ...</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>Matched</td>
      <td>Block</td>
      <td>GREENWOOD AVE N BETWEEN N 76TH ST AND N 77TH ST</td>
      <td>Injury Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>BROADWAY AND E PIKE ST</td>
      <td>Property Damage Only Collision</td>
      <td>Pedestrian</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHCILE STRUCK PEDESTRIAN</td>
      <td>Clear</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>Vehicle going straight hits pedestrian</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>Matched</td>
      <td>Block</td>
      <td>TERRY AVE BETWEEN VIRGINIA ST AND LENORA ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Dusk</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>Matched</td>
      <td>Block</td>
      <td>HARVARD AVE E BETWEEN E ALLISON ST AND EASTLAK...</td>
      <td>Injury Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Overcast</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>Unmatched</td>
      <td>Intersection</td>
      <td>15TH AVE NE AND NE 125TH ST</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>Unmatched</td>
      <td>Intersection</td>
      <td>LEARY AVE NW AND NW DOCK PL</td>
      <td>Injury Collision</td>
      <td>Parked Car</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHCILE STRUCK PEDESTRIAN</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>5TH AVE AND LENORA ST</td>
      <td>Injury Collision</td>
      <td>Pedestrian</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHCILE STRUCK PEDESTRIAN</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Vehicle turning left hits pedestrian</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>Matched</td>
      <td>Block</td>
      <td>4TH AVE BETWEEN PIKE ST AND PINE ST</td>
      <td>Property Damage Only Collision</td>
      <td>Sideswipe</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>NOT ENOUGH INFORMATION / NOT APPLICABLE</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - both going straight - bo...</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>Matched</td>
      <td>Block</td>
      <td>RAINIER AVE S BETWEEN 23RD AVE S AND S WALKER ST</td>
      <td>Property Damage Only Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>Matched</td>
      <td>Block</td>
      <td>NE 123RD ST BETWEEN HIRAM PL NE AND 28TH AVE NE</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>NOT ENOUGH INFORMATION / NOT APPLICABLE</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>Matched</td>
      <td>Block</td>
      <td>BATTERY ST TUNNEL NB BETWEEN ALASKAN WY VI NB ...</td>
      <td>Injury Collision</td>
      <td>Other</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE OVERTURNED IN ROAD</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Other</td>
      <td>Vehicle overturned</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>Matched</td>
      <td>Block</td>
      <td>20TH AVE BETWEEN E MADISON ST AND E DENNY WAY</td>
      <td>Injury Collision</td>
      <td>Left Turn</td>
      <td>Mid-Block (but intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From opposite direction - one left turn - one ...</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>Matched</td>
      <td>Block</td>
      <td>VALLEY ST BETWEEN WESTLAKE AVE N AND TERRY AVE N</td>
      <td>Property Damage Only Collision</td>
      <td>Parked Car</td>
      <td>Mid-Block (but intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>One parked--one moving</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>Matched</td>
      <td>Block</td>
      <td>RAINIER AVE S BETWEEN S BAYVIEW ST AND S MCCLE...</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>EASTLAKE AVE E AND E ROANOKE ST</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>NE PARK RD AND NE RAVENNA WB BV</td>
      <td>Property Damage Only Collision</td>
      <td>Angles</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>Matched</td>
      <td>Block</td>
      <td>AURORA AVE N BETWEEN N 90TH ST AND N 91ST ST</td>
      <td>Injury Collision</td>
      <td>Angles</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>Entering at angle</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>Matched</td>
      <td>Block</td>
      <td>PUGET BLVD SW BETWEEN SW HUDSON ST AND DEAD END 1</td>
      <td>Property Damage Only Collision</td>
      <td>Other</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE RAN OFF ROAD - HIT FIXED OBJECT</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Dark - Street Lights On</td>
      <td>Fixed object</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>Matched</td>
      <td>Block</td>
      <td>34TH AVE S BETWEEN S DAKOTA ST AND S GENESEE ST</td>
      <td>Injury Collision</td>
      <td>Head On</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From opposite direction - both moving - head-on</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>Matched</td>
      <td>Block</td>
      <td>AURORA AVE N BETWEEN N 85TH ST AND N 86TH ST</td>
      <td>Property Damage Only Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Raining</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>From same direction - both going straight - bo...</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>20TH AVE NE AND NE 75TH ST</td>
      <td>Injury Collision</td>
      <td>Left Turn</td>
      <td>At Intersection (intersection related)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, FRONT END ...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Daylight</td>
      <td>From opposite direction - one left turn - one ...</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>Matched</td>
      <td>Intersection</td>
      <td>GREENWOOD AVE N AND N 68TH ST</td>
      <td>Injury Collision</td>
      <td>Cycles</td>
      <td>At Intersection (intersection related)</td>
      <td>PEDALCYCLIST STRUCK MOTOR VEHICLE FRONT END AT...</td>
      <td>Clear</td>
      <td>Dry</td>
      <td>Dusk</td>
      <td>Vehicle Strikes Pedalcyclist</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>Matched</td>
      <td>Block</td>
      <td>34TH AVE BETWEEN E MARION ST AND E SPRING ST</td>
      <td>Property Damage Only Collision</td>
      <td>Rear Ended</td>
      <td>Mid-Block (not related to intersection)</td>
      <td>MOTOR VEHICLE STRUCK MOTOR VEHICLE, REAR END</td>
      <td>Clear</td>
      <td>Wet</td>
      <td>Daylight</td>
      <td>From same direction - both going straight - on...</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 11 columns</p>
</div>




```python
obj1 = obj.apply(lambda x: pd.factorize(x)[0])
obj1

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
      <th>STATUS</th>
      <th>ADDRTYPE</th>
      <th>LOCATION</th>
      <th>SEVERITYDESC</th>
      <th>COLLISIONTYPE</th>
      <th>JUNCTIONTYPE</th>
      <th>SDOT_COLDESC</th>
      <th>WEATHER</th>
      <th>ROADCOND</th>
      <th>LIGHTCOND</th>
      <th>ST_COLDESC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>2</td>
      <td>-1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>1</td>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>1</td>
      <td>19</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>1</td>
      <td>21</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>1</td>
      <td>22</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>1</td>
      <td>26</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>1</td>
      <td>27</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>1</td>
      <td>28</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>0</td>
      <td>1</td>
      <td>75</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>0</td>
      <td>0</td>
      <td>1249</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>0</td>
      <td>0</td>
      <td>590</td>
      <td>0</td>
      <td>9</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>1</td>
      <td>1</td>
      <td>9606</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>0</td>
      <td>1</td>
      <td>5371</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>14</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>0</td>
      <td>1</td>
      <td>2506</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>0</td>
      <td>0</td>
      <td>1229</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>0</td>
      <td>1</td>
      <td>2639</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>0</td>
      <td>1</td>
      <td>5281</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>0</td>
      <td>1</td>
      <td>7728</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>1</td>
      <td>0</td>
      <td>2270</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>1</td>
      <td>0</td>
      <td>15785</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>0</td>
      <td>0</td>
      <td>1375</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>0</td>
      <td>1</td>
      <td>1712</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>0</td>
      <td>1</td>
      <td>711</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>0</td>
      <td>1</td>
      <td>4519</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>0</td>
      <td>1</td>
      <td>175</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>15</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>18</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>0</td>
      <td>1</td>
      <td>484</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>0</td>
      <td>1</td>
      <td>726</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>0</td>
      <td>1</td>
      <td>1095</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>0</td>
      <td>0</td>
      <td>2633</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>0</td>
      <td>0</td>
      <td>24100</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>0</td>
      <td>1</td>
      <td>493</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>0</td>
      <td>1</td>
      <td>24101</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>0</td>
      <td>1</td>
      <td>11651</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>29</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>0</td>
      <td>1</td>
      <td>5376</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>0</td>
      <td>0</td>
      <td>1326</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>0</td>
      <td>0</td>
      <td>19912</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>0</td>
      <td>1</td>
      <td>14260</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 11 columns</p>
</div>




```python
df_1= pd.concat([num, obj1],axis=1)
df_1
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
      <th>SEVERITYCODE</th>
      <th>X</th>
      <th>Y</th>
      <th>OBJECTID</th>
      <th>INCKEY</th>
      <th>COLDETKEY</th>
      <th>INTKEY</th>
      <th>PERSONCOUNT</th>
      <th>PEDCOUNT</th>
      <th>PEDCYLCOUNT</th>
      <th>...</th>
      <th>ADDRTYPE</th>
      <th>LOCATION</th>
      <th>SEVERITYDESC</th>
      <th>COLLISIONTYPE</th>
      <th>JUNCTIONTYPE</th>
      <th>SDOT_COLDESC</th>
      <th>WEATHER</th>
      <th>ROADCOND</th>
      <th>LIGHTCOND</th>
      <th>ST_COLDESC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>-122.323148</td>
      <td>47.703140</td>
      <td>1</td>
      <td>1307</td>
      <td>1307</td>
      <td>37475.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-122.347294</td>
      <td>47.647172</td>
      <td>2</td>
      <td>52200</td>
      <td>52200</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>-122.334540</td>
      <td>47.607871</td>
      <td>3</td>
      <td>26700</td>
      <td>26700</td>
      <td>37558.450576</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-122.334803</td>
      <td>47.604803</td>
      <td>4</td>
      <td>1144</td>
      <td>1144</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>-122.306426</td>
      <td>47.545739</td>
      <td>5</td>
      <td>17700</td>
      <td>17700</td>
      <td>34387.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>-122.387598</td>
      <td>47.690575</td>
      <td>6</td>
      <td>320840</td>
      <td>322340</td>
      <td>36974.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>-122.338485</td>
      <td>47.618534</td>
      <td>7</td>
      <td>83300</td>
      <td>83300</td>
      <td>29510.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>9</td>
      <td>330897</td>
      <td>332397</td>
      <td>29745.000000</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>-122.335930</td>
      <td>47.611904</td>
      <td>10</td>
      <td>63400</td>
      <td>63400</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>-122.384700</td>
      <td>47.528475</td>
      <td>12</td>
      <td>58600</td>
      <td>58600</td>
      <td>34679.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>14</td>
      <td>48900</td>
      <td>48900</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>-1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>-122.333831</td>
      <td>47.547371</td>
      <td>15</td>
      <td>38800</td>
      <td>38800</td>
      <td>33194.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>-122.356273</td>
      <td>47.571375</td>
      <td>16</td>
      <td>2771</td>
      <td>2771</td>
      <td>37558.450576</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>-122.323966</td>
      <td>47.606374</td>
      <td>17</td>
      <td>32800</td>
      <td>32800</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2</td>
      <td>-122.317414</td>
      <td>47.664028</td>
      <td>19</td>
      <td>1212</td>
      <td>1212</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>-122.337663</td>
      <td>47.617510</td>
      <td>20</td>
      <td>330878</td>
      <td>332378</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>14</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>-122.344539</td>
      <td>47.692012</td>
      <td>21</td>
      <td>46300</td>
      <td>46300</td>
      <td>37365.000000</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>23</td>
      <td>23000</td>
      <td>23000</td>
      <td>37558.450576</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>-122.328270</td>
      <td>47.571420</td>
      <td>24</td>
      <td>330833</td>
      <td>332333</td>
      <td>37558.450576</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>-122.383802</td>
      <td>47.583715</td>
      <td>25</td>
      <td>97100</td>
      <td>97100</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>-122.292403</td>
      <td>47.732847</td>
      <td>26</td>
      <td>1347</td>
      <td>1347</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>19</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>-122.313786</td>
      <td>47.708535</td>
      <td>28</td>
      <td>1323</td>
      <td>1323</td>
      <td>36505.000000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>-122.318169</td>
      <td>47.615837</td>
      <td>29</td>
      <td>80000</td>
      <td>80000</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>21</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>-122.337486</td>
      <td>47.589746</td>
      <td>31</td>
      <td>28700</td>
      <td>28700</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>22</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2</td>
      <td>-122.279658</td>
      <td>47.553405</td>
      <td>33</td>
      <td>1268</td>
      <td>1268</td>
      <td>33499.000000</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2</td>
      <td>-122.312857</td>
      <td>47.599218</td>
      <td>34</td>
      <td>320932</td>
      <td>322432</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>24</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>-122.330730</td>
      <td>47.615450</td>
      <td>35</td>
      <td>113300</td>
      <td>113300</td>
      <td>29865.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>25</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>-122.385859</td>
      <td>47.581191</td>
      <td>36</td>
      <td>64700</td>
      <td>64700</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>26</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>-122.304990</td>
      <td>47.611474</td>
      <td>37</td>
      <td>1083</td>
      <td>1083</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>27</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>38</td>
      <td>61500</td>
      <td>61500</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>28</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>1</td>
      <td>-122.345863</td>
      <td>47.612991</td>
      <td>219510</td>
      <td>307577</td>
      <td>308857</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>75</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>1</td>
      <td>-122.330298</td>
      <td>47.603233</td>
      <td>219511</td>
      <td>312117</td>
      <td>313537</td>
      <td>30479.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1249</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>2</td>
      <td>-122.328079</td>
      <td>47.604161</td>
      <td>219512</td>
      <td>307692</td>
      <td>308972</td>
      <td>29973.000000</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>590</td>
      <td>0</td>
      <td>9</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>1</td>
      <td>-122.320756</td>
      <td>47.608656</td>
      <td>219513</td>
      <td>312289</td>
      <td>313709</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>9606</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>2</td>
      <td>-122.355449</td>
      <td>47.704720</td>
      <td>219514</td>
      <td>308575</td>
      <td>309855</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>5371</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>14</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>1</td>
      <td>-122.357624</td>
      <td>47.635956</td>
      <td>219515</td>
      <td>307792</td>
      <td>309072</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2506</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>2</td>
      <td>-122.302382</td>
      <td>47.626759</td>
      <td>219516</td>
      <td>307985</td>
      <td>309265</td>
      <td>29328.000000</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1229</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>2</td>
      <td>-122.355298</td>
      <td>47.684382</td>
      <td>219517</td>
      <td>310417</td>
      <td>311717</td>
      <td>37558.450576</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2639</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>1</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>219519</td>
      <td>309594</td>
      <td>310874</td>
      <td>29745.000000</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>1</td>
      <td>-122.335527</td>
      <td>47.617434</td>
      <td>219520</td>
      <td>308698</td>
      <td>309978</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>5281</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>2</td>
      <td>-122.322097</td>
      <td>47.649615</td>
      <td>219521</td>
      <td>308342</td>
      <td>309622</td>
      <td>37558.450576</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>7728</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>1</td>
      <td>-122.312679</td>
      <td>47.719414</td>
      <td>219522</td>
      <td>311117</td>
      <td>312437</td>
      <td>36427.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>2270</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>2</td>
      <td>-122.380016</td>
      <td>47.664879</td>
      <td>219523</td>
      <td>310911</td>
      <td>312231</td>
      <td>25174.000000</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>15785</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>2</td>
      <td>-122.340474</td>
      <td>47.614496</td>
      <td>219524</td>
      <td>312179</td>
      <td>313599</td>
      <td>29545.000000</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1375</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>1</td>
      <td>-122.337137</td>
      <td>47.610709</td>
      <td>219525</td>
      <td>307834</td>
      <td>309114</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1712</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>1</td>
      <td>-122.302554</td>
      <td>47.584099</td>
      <td>219528</td>
      <td>311697</td>
      <td>313057</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>711</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>1</td>
      <td>-122.298955</td>
      <td>47.717456</td>
      <td>219529</td>
      <td>308019</td>
      <td>309299</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>4519</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>2</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>219531</td>
      <td>308990</td>
      <td>310270</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>175</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>15</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>18</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>2</td>
      <td>-122.306092</td>
      <td>47.617881</td>
      <td>219532</td>
      <td>308532</td>
      <td>309812</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>484</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>1</td>
      <td>-122.337880</td>
      <td>47.625793</td>
      <td>219535</td>
      <td>307802</td>
      <td>309082</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>726</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>2</td>
      <td>-122.299160</td>
      <td>47.579673</td>
      <td>219536</td>
      <td>309335</td>
      <td>310615</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1095</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>1</td>
      <td>-122.325887</td>
      <td>47.643191</td>
      <td>219537</td>
      <td>309222</td>
      <td>310502</td>
      <td>28300.000000</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>2633</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>1</td>
      <td>-122.304217</td>
      <td>47.669537</td>
      <td>219538</td>
      <td>308480</td>
      <td>309760</td>
      <td>26005.000000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>24100</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>2</td>
      <td>-122.344569</td>
      <td>47.694547</td>
      <td>219539</td>
      <td>309170</td>
      <td>310450</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>493</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>1</td>
      <td>-122.361672</td>
      <td>47.556722</td>
      <td>219541</td>
      <td>307804</td>
      <td>309084</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>24101</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>2</td>
      <td>-122.290826</td>
      <td>47.565408</td>
      <td>219543</td>
      <td>309534</td>
      <td>310814</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>11651</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>29</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>1</td>
      <td>-122.344526</td>
      <td>47.690924</td>
      <td>219544</td>
      <td>309085</td>
      <td>310365</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>5376</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>2</td>
      <td>-122.306689</td>
      <td>47.683047</td>
      <td>219545</td>
      <td>311280</td>
      <td>312640</td>
      <td>24760.000000</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1326</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>2</td>
      <td>-122.355317</td>
      <td>47.678734</td>
      <td>219546</td>
      <td>309514</td>
      <td>310794</td>
      <td>24349.000000</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>19912</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>1</td>
      <td>-122.289360</td>
      <td>47.611017</td>
      <td>219547</td>
      <td>308220</td>
      <td>309500</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>14260</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 26 columns</p>
</div>




```python
df_1.shape
```




    (194673, 26)




```python
df_1.to_csv('clean_df.csv')
```


```python
df=pd.read_csv('clean_df.csv',low_memory=False)
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
      <th>Unnamed: 0</th>
      <th>SEVERITYCODE</th>
      <th>X</th>
      <th>Y</th>
      <th>OBJECTID</th>
      <th>INCKEY</th>
      <th>COLDETKEY</th>
      <th>INTKEY</th>
      <th>PERSONCOUNT</th>
      <th>PEDCOUNT</th>
      <th>...</th>
      <th>ADDRTYPE</th>
      <th>LOCATION</th>
      <th>SEVERITYDESC</th>
      <th>COLLISIONTYPE</th>
      <th>JUNCTIONTYPE</th>
      <th>SDOT_COLDESC</th>
      <th>WEATHER</th>
      <th>ROADCOND</th>
      <th>LIGHTCOND</th>
      <th>ST_COLDESC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>-122.323148</td>
      <td>47.703140</td>
      <td>1</td>
      <td>1307</td>
      <td>1307</td>
      <td>37475.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>-122.347294</td>
      <td>47.647172</td>
      <td>2</td>
      <td>52200</td>
      <td>52200</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>-122.334540</td>
      <td>47.607871</td>
      <td>3</td>
      <td>26700</td>
      <td>26700</td>
      <td>37558.450576</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>-122.334803</td>
      <td>47.604803</td>
      <td>4</td>
      <td>1144</td>
      <td>1144</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2</td>
      <td>-122.306426</td>
      <td>47.545739</td>
      <td>5</td>
      <td>17700</td>
      <td>17700</td>
      <td>34387.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>1</td>
      <td>-122.387598</td>
      <td>47.690575</td>
      <td>6</td>
      <td>320840</td>
      <td>322340</td>
      <td>36974.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>1</td>
      <td>-122.338485</td>
      <td>47.618534</td>
      <td>7</td>
      <td>83300</td>
      <td>83300</td>
      <td>29510.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>2</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>9</td>
      <td>330897</td>
      <td>332397</td>
      <td>29745.000000</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>1</td>
      <td>-122.335930</td>
      <td>47.611904</td>
      <td>10</td>
      <td>63400</td>
      <td>63400</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>2</td>
      <td>-122.384700</td>
      <td>47.528475</td>
      <td>12</td>
      <td>58600</td>
      <td>58600</td>
      <td>34679.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>1</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>14</td>
      <td>48900</td>
      <td>48900</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>-1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>1</td>
      <td>-122.333831</td>
      <td>47.547371</td>
      <td>15</td>
      <td>38800</td>
      <td>38800</td>
      <td>33194.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>1</td>
      <td>-122.356273</td>
      <td>47.571375</td>
      <td>16</td>
      <td>2771</td>
      <td>2771</td>
      <td>37558.450576</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>1</td>
      <td>-122.323966</td>
      <td>47.606374</td>
      <td>17</td>
      <td>32800</td>
      <td>32800</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>2</td>
      <td>-122.317414</td>
      <td>47.664028</td>
      <td>19</td>
      <td>1212</td>
      <td>1212</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>1</td>
      <td>-122.337663</td>
      <td>47.617510</td>
      <td>20</td>
      <td>330878</td>
      <td>332378</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>14</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>2</td>
      <td>-122.344539</td>
      <td>47.692012</td>
      <td>21</td>
      <td>46300</td>
      <td>46300</td>
      <td>37365.000000</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>1</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>23</td>
      <td>23000</td>
      <td>23000</td>
      <td>37558.450576</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>2</td>
      <td>-122.328270</td>
      <td>47.571420</td>
      <td>24</td>
      <td>330833</td>
      <td>332333</td>
      <td>37558.450576</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>1</td>
      <td>-122.383802</td>
      <td>47.583715</td>
      <td>25</td>
      <td>97100</td>
      <td>97100</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>2</td>
      <td>-122.292403</td>
      <td>47.732847</td>
      <td>26</td>
      <td>1347</td>
      <td>1347</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>19</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>2</td>
      <td>-122.313786</td>
      <td>47.708535</td>
      <td>28</td>
      <td>1323</td>
      <td>1323</td>
      <td>36505.000000</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>1</td>
      <td>-122.318169</td>
      <td>47.615837</td>
      <td>29</td>
      <td>80000</td>
      <td>80000</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>21</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>1</td>
      <td>-122.337486</td>
      <td>47.589746</td>
      <td>31</td>
      <td>28700</td>
      <td>28700</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>22</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>2</td>
      <td>-122.279658</td>
      <td>47.553405</td>
      <td>33</td>
      <td>1268</td>
      <td>1268</td>
      <td>33499.000000</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>2</td>
      <td>-122.312857</td>
      <td>47.599218</td>
      <td>34</td>
      <td>320932</td>
      <td>322432</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>24</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>1</td>
      <td>-122.330730</td>
      <td>47.615450</td>
      <td>35</td>
      <td>113300</td>
      <td>113300</td>
      <td>29865.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>25</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>1</td>
      <td>-122.385859</td>
      <td>47.581191</td>
      <td>36</td>
      <td>64700</td>
      <td>64700</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>26</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>1</td>
      <td>-122.304990</td>
      <td>47.611474</td>
      <td>37</td>
      <td>1083</td>
      <td>1083</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>27</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>1</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>38</td>
      <td>61500</td>
      <td>61500</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>28</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>194643</th>
      <td>194643</td>
      <td>1</td>
      <td>-122.345863</td>
      <td>47.612991</td>
      <td>219510</td>
      <td>307577</td>
      <td>308857</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>75</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>194644</th>
      <td>194644</td>
      <td>1</td>
      <td>-122.330298</td>
      <td>47.603233</td>
      <td>219511</td>
      <td>312117</td>
      <td>313537</td>
      <td>30479.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1249</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194645</th>
      <td>194645</td>
      <td>2</td>
      <td>-122.328079</td>
      <td>47.604161</td>
      <td>219512</td>
      <td>307692</td>
      <td>308972</td>
      <td>29973.000000</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>590</td>
      <td>0</td>
      <td>9</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>19</td>
    </tr>
    <tr>
      <th>194646</th>
      <td>194646</td>
      <td>1</td>
      <td>-122.320756</td>
      <td>47.608656</td>
      <td>219513</td>
      <td>312289</td>
      <td>313709</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>9606</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194647</th>
      <td>194647</td>
      <td>2</td>
      <td>-122.355449</td>
      <td>47.704720</td>
      <td>219514</td>
      <td>308575</td>
      <td>309855</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>5371</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>14</td>
    </tr>
    <tr>
      <th>194648</th>
      <td>194648</td>
      <td>1</td>
      <td>-122.357624</td>
      <td>47.635956</td>
      <td>219515</td>
      <td>307792</td>
      <td>309072</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2506</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194649</th>
      <td>194649</td>
      <td>2</td>
      <td>-122.302382</td>
      <td>47.626759</td>
      <td>219516</td>
      <td>307985</td>
      <td>309265</td>
      <td>29328.000000</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1229</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>194650</th>
      <td>194650</td>
      <td>2</td>
      <td>-122.355298</td>
      <td>47.684382</td>
      <td>219517</td>
      <td>310417</td>
      <td>311717</td>
      <td>37558.450576</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>2639</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194651</th>
      <td>194651</td>
      <td>1</td>
      <td>-122.320780</td>
      <td>47.614076</td>
      <td>219519</td>
      <td>309594</td>
      <td>310874</td>
      <td>29745.000000</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>14</td>
    </tr>
    <tr>
      <th>194652</th>
      <td>194652</td>
      <td>1</td>
      <td>-122.335527</td>
      <td>47.617434</td>
      <td>219520</td>
      <td>308698</td>
      <td>309978</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>5281</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194653</th>
      <td>194653</td>
      <td>2</td>
      <td>-122.322097</td>
      <td>47.649615</td>
      <td>219521</td>
      <td>308342</td>
      <td>309622</td>
      <td>37558.450576</td>
      <td>6</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>7728</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194654</th>
      <td>194654</td>
      <td>1</td>
      <td>-122.312679</td>
      <td>47.719414</td>
      <td>219522</td>
      <td>311117</td>
      <td>312437</td>
      <td>36427.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>2270</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194655</th>
      <td>194655</td>
      <td>2</td>
      <td>-122.380016</td>
      <td>47.664879</td>
      <td>219523</td>
      <td>310911</td>
      <td>312231</td>
      <td>25174.000000</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>15785</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194656</th>
      <td>194656</td>
      <td>2</td>
      <td>-122.340474</td>
      <td>47.614496</td>
      <td>219524</td>
      <td>312179</td>
      <td>313599</td>
      <td>29545.000000</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1375</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>194657</th>
      <td>194657</td>
      <td>1</td>
      <td>-122.337137</td>
      <td>47.610709</td>
      <td>219525</td>
      <td>307834</td>
      <td>309114</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1712</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>194658</th>
      <td>194658</td>
      <td>1</td>
      <td>-122.302554</td>
      <td>47.584099</td>
      <td>219528</td>
      <td>311697</td>
      <td>313057</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>711</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>194659</th>
      <td>194659</td>
      <td>1</td>
      <td>-122.298956</td>
      <td>47.717456</td>
      <td>219529</td>
      <td>308019</td>
      <td>309299</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>4519</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194660</th>
      <td>194660</td>
      <td>2</td>
      <td>-122.330518</td>
      <td>47.619543</td>
      <td>219531</td>
      <td>308990</td>
      <td>310270</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>175</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>15</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>18</td>
    </tr>
    <tr>
      <th>194661</th>
      <td>194661</td>
      <td>2</td>
      <td>-122.306092</td>
      <td>47.617881</td>
      <td>219532</td>
      <td>308532</td>
      <td>309812</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>484</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>38</td>
    </tr>
    <tr>
      <th>194662</th>
      <td>194662</td>
      <td>1</td>
      <td>-122.337880</td>
      <td>47.625793</td>
      <td>219535</td>
      <td>307802</td>
      <td>309082</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>726</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>194663</th>
      <td>194663</td>
      <td>2</td>
      <td>-122.299160</td>
      <td>47.579673</td>
      <td>219536</td>
      <td>309335</td>
      <td>310615</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1095</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194664</th>
      <td>194664</td>
      <td>1</td>
      <td>-122.325887</td>
      <td>47.643191</td>
      <td>219537</td>
      <td>309222</td>
      <td>310502</td>
      <td>28300.000000</td>
      <td>8</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>2633</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194665</th>
      <td>194665</td>
      <td>1</td>
      <td>-122.304217</td>
      <td>47.669537</td>
      <td>219538</td>
      <td>308480</td>
      <td>309760</td>
      <td>26005.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>24100</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194666</th>
      <td>194666</td>
      <td>2</td>
      <td>-122.344569</td>
      <td>47.694547</td>
      <td>219539</td>
      <td>309170</td>
      <td>310450</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>493</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>194667</th>
      <td>194667</td>
      <td>1</td>
      <td>-122.361672</td>
      <td>47.556722</td>
      <td>219541</td>
      <td>307804</td>
      <td>309084</td>
      <td>37558.450576</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>24101</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
    </tr>
    <tr>
      <th>194668</th>
      <td>194668</td>
      <td>2</td>
      <td>-122.290826</td>
      <td>47.565408</td>
      <td>219543</td>
      <td>309534</td>
      <td>310814</td>
      <td>37558.450576</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>11651</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>29</td>
    </tr>
    <tr>
      <th>194669</th>
      <td>194669</td>
      <td>1</td>
      <td>-122.344526</td>
      <td>47.690924</td>
      <td>219544</td>
      <td>309085</td>
      <td>310365</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>5376</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>194670</th>
      <td>194670</td>
      <td>2</td>
      <td>-122.306689</td>
      <td>47.683047</td>
      <td>219545</td>
      <td>311280</td>
      <td>312640</td>
      <td>24760.000000</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1326</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>194671</th>
      <td>194671</td>
      <td>2</td>
      <td>-122.355317</td>
      <td>47.678734</td>
      <td>219546</td>
      <td>309514</td>
      <td>310794</td>
      <td>24349.000000</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>19912</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>194672</th>
      <td>194672</td>
      <td>1</td>
      <td>-122.289360</td>
      <td>47.611017</td>
      <td>219547</td>
      <td>308220</td>
      <td>309500</td>
      <td>37558.450576</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>14260</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>194673 rows × 27 columns</p>
</div>




```python

```
