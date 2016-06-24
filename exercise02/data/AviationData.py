# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:51:39 2016

@author: Corinne
"""
###################
# READING IN DATA #
###################

# reading in the text file of AviationData

# note: while converting xml to pandas is probably more elegant, 
#   it's much faster to use a txt file if it's also available
#   also faster to simply replace the delimiter in the txt with a simple '|' delim

import pandas as pd
import numpy as np

data = pd.read_table('AviationData.txt', sep='|', header=0, index_col = False)

# reading the table added an extra field at the end, removing now
data.drop(data.columns[31], axis=1, inplace=True)

# more pandas-friendly column names
data_cols = ['EventId', 'InvType', 'AccNumber', 'Date', 'Location', 'Country', 'Lat', 'Long', 'AirportCode', 'AirportName', 'InjurySev',  'AircraftDamage', 'AircraftCat', 'RegNumber', 'Make', 'Model', 'AmateurBuilt', 'EngineNum', 'EngineType', 'FAR', 'Schedule', 'Purpose', 'AirCarrier', 'NumFatal', 'NumSerious', 'NumMinor', 'NumUninjured', 'Weather', 'Phase', 'Status', 'PublicationDate']

data.columns = data_cols

# reading in the json Narrative files
# remember to navigate to the appropriate directory that contains all the json files
# (should be the same one with the AviationData.txt but check just in case)

import glob

# reads all files in directory with json extension, performs pd.concat
allfiles = glob.glob('*.json')
narratives = pd.concat(pd.read_json(f, orient='split') for f in allfiles)

#################
# DATA CLEANING #
#################

# based on some data exploration

# changing date formats to datetime
data.Date = pd.to_datetime(data.Date)
data.PublicationDate = pd.to_datetime(data.PublicationDate)

# adding a year column
data['Year'] = data.Date.dt.year

# changing the 'Fatal (*)' in InjurySev to just 'Fatal'
data['InjurySev'] = data['InjurySev'].str.replace(r'Fatal.+', 'Fatal')

# making Make all lowercase
data['Make'] = data['Make'].str.lower()

# assuming null values in number of various injuries = 0
data.NumFatal.fillna(value = 0, inplace=True)
data.NumSerious.fillna(value = 0, inplace=True)
data.NumMinor.fillna(value=0, inplace=True)
data.NumUninjured.fillna(value=0, inplace=True)

# adding a total casualty column, assuming mutually exclusive
data['NumCasualty'] = data.NumFatal + data.NumSerious + data.NumMinor

# adding total number of involved persons in event
data['TotalPerson'] = data.NumFatal + data.NumSerious + data.NumMinor + data.NumUninjured

# adding a Cas binary column (0 = did not result in casualty, 1 = resulted in at least one casualty)
data['Cas'] = np.where(data.NumCasualty > 0, 1, 0)

##############
# DATA MERGE #
##############

# merging the AviationData with the narrative data
alldata = data.merge(narratives, how='outer', on='EventId')

####################
# DATA EXPLORATION #
####################

# two data sets to explore:
#   data: contains general info about flight incidents
#   narratives: contains text comments about flight incidents

#
# looking at 'data' first
#

data.head()

# getting sense of categorical values

data.InvType.value_counts()
    # accident = 75305
    # incident = 3142

data.Location.value_counts()
    # nearest city to event
    # top 5 are: Anchorage (372), Miami (185), Chicago (169), Albuquerque (165), and Houston (155)

data.Country.value_counts()
    # country in which event took place
    # top 5 are: US (74041), Canada (239), Mexico (209), Brazil (208), UK (205)

data.AirportCode.value_counts()
    # closest airport within 3 miles of event, or the involved aircraft was taking off from or on approach to an airport
    # top 5 are: no airport (1461), private (349), ORD- Chicago (146), APA- Centennial in Denver (141), MRI- Anchorage (141)

data.AirportName.value_counts()
    # several iterations of private (PRIVATE, Private, Private Airstrip, PRIVATE STRIP, PRIVATE AIRSTRIP)
    # Airport Code is more accurate to use since there are various name iterations of the same airport (e.g. Merrill Field vs Merrill)

data.InjurySev.value_counts()
    # indicates highest level of injury among all injuries sustained
    # non-fatal = 59426
    # fatal = 15667
    # incident = 3142
    # unavailable = 212

data.AircraftDamage.value_counts()
    # substantial = 56385
    # destroyed = 17190
    # minor = 2498

data.AircraftCat.value_counts()
    # airplane = 18495
    # helicopter = 2275
    # glider = 370
    # balloon = 170
    # gyrocraft = 101
    # weight-shift = 58
    # powered parachute = 46
    # unk = 37
    # ultralight = 28
    # powered-lift = 5
    # blimp = 3
    # rocket = 1 (who is riding a rocket?)

data.Make.value_counts()
    # cessna = 24626
    # piper = 13416
    # beech = 4843
    # bell = 2447
    # boeing = 2130

data.AmateurBuilt.value_counts()
    # no = 70378
    # yes = 7495

data.FAR.value_counts()
    # Part 91: General Aviation            17305
    # Part 137: Agricultural                1062
    # Non-U.S., Non-Commercial               744
    # Part 135: Air Taxi & Commuter          739
    # Part 121: Air Carrier                  519
    # Non-U.S., Commercial                   503
    # Part 129: Foreign                      195
    # Unknown                                175
    # Public Use                             174
    # Part 133: Rotorcraft Ext. Load          95
    # Part 91 Subpart K: Fractional           14
    # Part 125: 20+ Pax,6000+ lbs              7
    # Armed Forces                             7
    # Part 103: Ultralight                     7
    # Part 91F: Special Flt Ops.               1
    # Part 437: Commercial Space Flight        1 (thus the rocket?)

data.Purpose.value_counts()
    # Personal                     44098
    # Instructional                 9386
    # Unknown                       6788
    # Aerial Application            4327
    # Business                      3843
    # Positioning                   1501
    # Other Work Use                1125
    # Ferry                          772
    # Public Use                     707
    # Aerial Observation             664
    # Executive/Corporate            513
    # Flight Test                    305
    # Air Race/Show                  137
    # Skydiving                      114
    # Public Aircraft - Federal       84
    # External Load                   81
    # Banner Tow                      79
    # Public Aircraft - State         54
    # Public Aircraft - Local         51
    # Glider Tow                      42
    # Fire Fighting                   21
    # Air Drop                        10

data.AirCarrier.value_counts()
    # values are messy and nonstandard
    # would be a fun (ish) data cleaning exercise

data.Weather.value_counts()
    # vmc = 69782 (visual meteorological conditions, conditions in which pilots have sufficient visibility to fly the aircraft mintaining visual separation)
    # imc = 5637 (instrument meteorological conditions, conditions that require pilots to fly primarily by reference to instruments)
    # unk = 950

data.Phase.value_counts()
    # LANDING        18890
    # TAKEOFF        15063
    # CRUISE         10679
    # MANEUVERING     9684
    # APPROACH        7634
    # TAXI            2306 
    # CLIMB           2260
    # DESCENT         2181
    # GO-AROUND       1599
    # STANDING        1197
    # UNKNOWN          645
    # OTHER            152

data.Status.value_counts()
    # furthest level to which a report has been completed
    # Probable Cause    73261
    # Foreign            3822
    # Preliminary        1097
    # Factual             267

# getting a sense of the number of passengers involved/injured

data.NumFatal.describe()
    # count    78447.000000
    # mean         0.572641
    # std          5.212096
    # min          0.000000
    # 25%          0.000000
    # 50%          0.000000
    # 75%          0.000000
    # max        349.000000

# looking at only events where there were fatalities
data[data.NumFatal <> 0].NumFatal.describe()
    # count    15686.000000
    # mean         2.863828
    # std         11.371197
    # min          1.000000
    # 25%          1.000000
    # 50%          2.000000
    # 75%          2.000000g
    # max        349.000000

# getting sense of date values

data.Date.min() # first recorded incident/accident is 10/24/1948
data.Date.max() # last recorded incident/accident is 6/19/2016

data.PublicationDate.min() # database first published on 4/16/1980

data.groupby(data['Date'].map(lambda x: x.year)).EventId.count()
# looking since 1990, there has been a decrease in the number of reported aviation events
    # 1990    2518
    # 1991    2462
    # 1992    2354
    # 1993    2313
    # 1994    2257
    # 1995    2309
    # 1996    2187
    # 1997    2148
    # 1998    2226
    # 1999    2209
    # 2000    2220
    # 2001    2063
    # 2002    2020
    # 2003    2085
    # 2004    1952
    # 2005    2031
    # 2006    1851
    # 2007    2017
    # 2008    1931
    # 2009    1806
    # 2010    1821
    # 2011    1886
    # 2012    1863
    # 2013    1556
    # 2014    1537
    # 2015    1576
    # 2016     559

#
# Since 2005:
#

data[data.Year > 2005].groupby(data['Year'])[['NumFatal', 'NumSerious', 'NumMinor', 'NumUninjured', 'TotalPerson']].sum()

'''
      NumFatal  NumSerious  NumMinor  NumUninjured  TotalPerson
Year                                                           
2006      1489         420       473         10607        12989
2007      1335         402       543         12081        14361
2008      1292         517       786         12753        15348
2009      1207         378       632         10982        13199
2010      1382         343       609         13286        15620
2011       959         430       489         15447        17325
2012      1007         335       467         11885        13694
2013       826         359       659          9059        10903
2014      1435         320       469          9779        12003
2015       877         357       452          7730         9416
2016       300         105       150          1534         2089
'''

data[data.Year > 2005].groupby(data['Year'])[['NumFatal', 'NumSerious', 'NumMinor', 'NumUninjured', 'TotalPerson']].mean()

'''
      NumFatal  NumSerious  NumMinor  NumUninjured  TotalPerson
Year                                                           
2006  0.804430    0.226904  0.255538      5.730416     7.017288
2007  0.661874    0.199306  0.269212      5.989588     7.119980
2008  0.669083    0.267737  0.407043      6.604350     7.948213
2009  0.668328    0.209302  0.349945      6.080842     7.308416
2010  0.758924    0.188358  0.334432      7.295991     8.577705
2011  0.508484    0.227996  0.259279      8.190350     9.186108
2012  0.540526    0.179817  0.250671      6.379495     7.350510
2013  0.530848    0.230720  0.423522      5.821979     7.007069
2014  0.933637    0.208198  0.305140      6.362394     7.809369
2015  0.556472    0.226523  0.286802      4.904822     5.974619
2016  0.536673    0.187835  0.268336      2.744186     3.737030
'''

data[data.Year > 2005].groupby(data['Year']).Cas.mean()

'''
Year
2006    0.454349
2007    0.443233
2008    0.441222
2009    0.457918
2010    0.433828
2011    0.458643
2012    0.458937
2013    0.469794
2014    0.495771
2015    0.487944
2016    0.452594
'''

#
# looking at 'narratives'
#

import numpy as np
import scipy as sp
import sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# create series of just the narratives
narrative = narratives.narrative

#
# remove english stopwords, looking at 1-2 word combinations that appear at least 1000 times
#
 
vect = CountVectorizer(stop_words = 'english', ngram_range=(1, 2), min_df=1000)


narr_dtm = vect.fit_transform(narrative)    # create dtm
narr_array = narr_dtm.toarray()             # turn into array
dtm_cols = vect.get_feature_names()         # get tokens (words)
dtm_pd = pd.DataFrame(narr_array, columns = dtm_cols)
    # transforming into pandas dataframe
narr_token = pd.DataFrame({'token':dtm_cols, 'count':np.sum(narr_array, axis=0)})
    # transforming into pandas dataframe based on token counts
narr_token.sort('count', ascending=False)

#
# remove english stopwords, looking at single words that appear at least 1000 times
#

vect = CountVectorizer(stop_words = 'english', min_df=1000)


narr_dtm = vect.fit_transform(narrative)    # create dtm
narr_array = narr_dtm.toarray()             # turn into array
dtm_cols = vect.get_feature_names()         # get tokens (words)
dtm_pd = pd.DataFrame(narr_array, columns = dtm_cols)
    # transforming into pandas dataframe
narr_token = pd.DataFrame({'token':dtm_cols, 'count':np.sum(narr_array, axis=0)})
    # transforming into pandas dataframe based on token counts
narr_token.sort('count', ascending=False)

#
# remove english stopwords, looking at single words, only top 500 features ordered by term frequency
# looking at the term frequency-inverse document frequency
#

vect = TfidfVectorizer(stop_words = 'english', max_features=500)

narr_dtm = vect.fit_transform(narrative)    # create dtm
dtm_cols = vect.get_feature_names()         # get tokens (words)

# returning top tfidf features
# https://buhrmann.github.io/tfidf-analysis.html

def top_tfidf(row, features, top_n=10):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n = 10):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf(row, features, top_n)

# will return the top n tokens based on average of tf-idf values across all documents in corpus
def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n = 10):
    if grp_ids:
        D = np.squeeze(Xtr[grp_ids].toarray())
    else:
        D = np.squeeze(Xtr.toarray())
    
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf(tfidf_means, features, top_n)
    
#################
# DATA ANALYSIS #
#################

#
# Creating a model to predict probability of a casualty (logistic regression)
#

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

# create a new data set based on some parameters gleaned from data exploration
    # only look at events post 1995
    # only look at non-home built planes
    # consider only flights that have flight pattern over US
    # consider only flights that have following authorities:
        # Part 91: General Aviation            17305
        # Part 135: Air Taxi & Commuter          739
        # Part 121: Air Carrier                  519
        # assuming client is not going to be using the flight for agricultural, military, or other specialized purposes, and will only be flying domestically

FAR_features = ['Part 91: General Aviation', 'Part 135: Air Taxi & Commuter', 'Part 121: Air Carrier']

av_data = data[(data.Year >= 1995) & (data.AmateurBuilt == 'No') & (data.FAR.isin(FAR_features)) & (data.Country == 'United States')]

# what features do we know for sure we're not going to look at?
    # AccNumber
    # Date
    # AirportName
    # RegNumber
    # Model
    # AmateurBuilt
    # Schedule
    # AirCarrier
    # Status
    # PublicationDate

drop_cols = ['AccNumber', 'Date', 'AirportName', 'RegNumber', 'Model', 'AmateurBuilt', 'Schedule', 'AirCarrier', 'Status', 'PublicationDate']

av_data.drop(drop_cols, axis=1, inplace=True)

# add a state column so we don't have to look at Location

av_data['State'] = data['Location'].str[-2:]

av_data['Weather2'] = av_data.Weather.map({'VMC':0, 'IMC':1, 'UNK':np.nan})
    
# only looking at features that can be measured before an event (ie no AircraftDamage since that would be another example of something being predicted)
# possible features to try:
    # State (will have 16 nulls to remove)
    # Lat / Long --> can use, don't overfit because state is in there
    # Airport Code --> too many null values
    # AircraftCat --> remove nulls (60)
    # Engine info --> too many null values
    # Purpose --> too many null values
    # Weather --> remove null values? 142
    # Make --> too many variations
    # total features left to use: State, Lat, Long, AircraftCat, Weather, Make, FAR, Year

# reducing the options for Make - keeping top 6, all else called 'other'
av_data['Make2'] = np.where((av_data.Make == 'cessna') | (av_data.Make == 'piper') | (av_data.Make == 'beech') | (av_data.Make == 'bell') | (av_data.Make == 'mooney') | (av_data.Make == 'boeing'), av_data.Make, 'other')



# removing null values
av_data = av_data[av_data.State.notnull()]
av_data = av_data[av_data.AircraftCat.notnull()]
av_data = av_data[av_data.Weather2.notnull()]
av_data = av_data[av_data.Lat.notnull()]
av_data = av_data[av_data.Long.notnull()]

# creating a lot of dummy variables

# dummy variables for State
state_dummies = pd.get_dummies(av_data.State, prefix='State').iloc[:, 1:]

# dummy variables for AircraftCat
aircat_dummies = pd.get_dummies(av_data.AircraftCat, prefix='AirCat').iloc[:, 1:]

# dummy variables for Make2
make_dummies = pd.get_dummies(av_data.Make2, prefix='Make').iloc[:, 1:]

# av_data = pd.concat(dummy, axis=1)

logreg = LogisticRegression(C=1e9)

# features combo 1
# Lat, Long, AircraftCat, Make2, Weather
combo1 = pd.concat([av_data, aircat_dummies, make_dummies], axis=1)

feature_cols1 = ['Lat', 'Long', 'Weather2', 'AirCat_Balloon',
       u'AirCat_Blimp', u'AirCat_Glider', u'AirCat_Gyrocraft',
       u'AirCat_Helicopter', u'AirCat_Powered Parachute',
       u'AirCat_Powered-Lift', u'AirCat_Ultralight', u'AirCat_Unknown',
       u'AirCat_Weight-Shift', u'Make_bell',
       u'Make_boeing', u'Make_cessna', u'Make_mooney', u'Make_other',
       u'Make_piper']

X = combo1[feature_cols1]
y = combo1.Cas

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

logreg.fit(X_train, y_train)

zip(feature_cols1, logreg.coef_[0])

'''
[('Lat', -0.017146201482451315),
 ('Long', 0.0011116124297999965),
 ('Weather2', 1.6597774406654475),
 ('AirCat_Balloon', 2.1693039656893869),
 (u'AirCat_Blimp', -0.26282933100790906),
 (u'AirCat_Glider', 0.60922331329376378),
 (u'AirCat_Gyrocraft', -0.24190009843062429),
 (u'AirCat_Helicopter', 0.10128391055672985),
 (u'AirCat_Powered Parachute', 2.5897013346865334),
 (u'AirCat_Powered-Lift', 0.61542716335742398),
 (u'AirCat_Ultralight', 2.589441155313259),
 (u'AirCat_Unknown', -0.13025918144810691),
 (u'AirCat_Weight-Shift', 1.6113404204941721),
 (u'Make_bell', -0.22374480281223985),
 (u'Make_boeing', -0.14200135047260076),
 (u'Make_cessna', -0.40771909357042602),
 (u'Make_mooney', 0.27282390518609156),
 (u'Make_other', -0.073705153016272867),
 (u'Make_piper', -0.23954083104301263)]
 '''

zip(feature_cols1, np.exp(logreg.coef_[0]))

'''
[('Lat', 0.98299995807752727),
 ('Long', 1.0011122304998941),
 ('Weather2', 5.2581404659693129),
 ('AirCat_Balloon', 8.7521900955898637),
 (u'AirCat_Blimp', 0.76887310891250982),
 (u'AirCat_Glider', 1.8390025151497551),
 (u'AirCat_Gyrocraft', 0.78513460981370287),
 (u'AirCat_Helicopter', 1.1065907699707309),
 (u'AirCat_Powered Parachute', 13.325791057236559),
 (u'AirCat_Powered-Lift', 1.8504468738575706),
 (u'AirCat_Ultralight', 13.322324412265216),
 (u'AirCat_Unknown', 0.8778678743656918),
 (u'AirCat_Weight-Shift', 5.0095215948838847),
 (u'Make_bell', 0.79951914337394525),
 (u'Make_boeing', 0.86762008478797048),
 (u'Make_cessna', 0.66516570190366786),
 (u'Make_mooney', 1.313668894176055),
 (u'Make_other', 0.92894555025997461),
 (u'Make_piper', 0.78698913909853752)]
'''

# Looking at Weather2 --> 0 if VMC, 1 if IMC. So all else being equal, have IMC weather conditions is associated with an increase in the log-odds of a casualty resulting by 1.65, or an increase in the odds of a casualty by 5.3 as compared to VMC weather conditions.

y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

from sklearn.dummy import DummyClassifier
dumb = DummyClassifier(strategy='most_frequent')
dumb.fit(X_train, y_train)
y_dumb_class = dumb.predict(X_test)
print metrics.accuracy_score(y_test, y_dumb_class)

# testing accuracy = 0.6603
# null accuracy = 0.61

# features combo 2
# AircraftCat, Make2, Weather
combo2 = pd.concat([av_data, aircat_dummies, make_dummies], axis=1)

feature_cols2 = ['Weather2', 'AirCat_Balloon',
       u'AirCat_Blimp', u'AirCat_Glider', u'AirCat_Gyrocraft',
       u'AirCat_Helicopter', u'AirCat_Powered Parachute',
       u'AirCat_Powered-Lift', u'AirCat_Ultralight', u'AirCat_Unknown',
       u'AirCat_Weight-Shift', u'Make_bell',
       u'Make_boeing', u'Make_cessna', u'Make_mooney', u'Make_other',
       u'Make_piper']

X = combo2[feature_cols2]
y = combo2.Cas

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

logreg.fit(X_train, y_train)

zip(feature_cols2, logreg.coef_[0])

'''
[('Weather2', 1.6614599579862583),
 ('AirCat_Balloon', 2.1991462003024025),
 (u'AirCat_Blimp', -0.1520043119118592),
 (u'AirCat_Glider', 0.62929738183400585),
 (u'AirCat_Gyrocraft', -0.15558237598175562),
 (u'AirCat_Helicopter', 0.13486991162820108),
 (u'AirCat_Powered Parachute', 2.5752783244434347),
 (u'AirCat_Powered-Lift', 0.5441384616146836),
 (u'AirCat_Ultralight', 4.8202921930003129),
 (u'AirCat_Unknown', -0.15497466646344593),
 (u'AirCat_Weight-Shift', 1.6467914114203526),
 (u'Make_bell', -0.26856351511999266),
 (u'Make_boeing', -0.14958747020502855),
 (u'Make_cessna', -0.45077346629182069),
 (u'Make_mooney', 0.26830875302902679),
 (u'Make_other', -0.10940676107485683),
 (u'Make_piper', -0.30368320754953598)]
 '''

zip(feature_cols2, np.exp(logreg.coef_[0]))

'''
[('Weather2', 5.2669948250960692),
 ('AirCat_Balloon', 9.0173112342012498),
 (u'AirCat_Blimp', 0.85898457686735419),
 (u'AirCat_Glider', 1.8762917993163173),
 (u'AirCat_Gyrocraft', 0.8559165670572968),
 (u'AirCat_Helicopter', 1.1443879031086435),
 (u'AirCat_Powered Parachute', 13.134972433682337),
 (u'AirCat_Powered-Lift', 1.7231232059155088),
 (u'AirCat_Ultralight', 124.00131780401662),
 (u'AirCat_Unknown', 0.85643687378363864),
 (u'AirCat_Weight-Shift', 5.1902995455628016),
 (u'Make_bell', 0.76447686542796467),
 (u'Make_boeing', 0.86106311735817442),
 (u'Make_cessna', 0.63713515842116708),
 (u'Make_mooney', 1.3077508497244958),
 (u'Make_other', 0.89636573664377339),
 (u'Make_piper', 0.73809465223140558)]
'''

y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

from sklearn.dummy import DummyClassifier
dumb = DummyClassifier(strategy='most_frequent')
dumb.fit(X_train, y_train)
y_dumb_class = dumb.predict(X_test)
print metrics.accuracy_score(y_test, y_dumb_class)

# testing accuracy = 0.66
# null accuracy = 0.61
# not much change in testing accuracy

# features combo 3
# Lat, Long, Weather2, Make
combo3 = pd.concat([av_data, make_dummies], axis=1)

feature_cols3 = ['Lat', 'Long', 'Weather2', u'Make_bell',
       u'Make_boeing', u'Make_cessna', u'Make_mooney', u'Make_other',
       u'Make_piper']

X = combo3[feature_cols3]
y = combo3.Cas

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

logreg.fit(X_train, y_train)

zip(feature_cols3, logreg.coef_[0])

'''
[('Lat', -0.017448698522807043),
 ('Long', 0.001122113165161842),
 ('Weather2', 1.6199424570993146),
 (u'Make_bell', -0.13964427949384944),
 (u'Make_boeing', -0.1540461823850505),
 (u'Make_cessna', -0.41035533098319021),
 (u'Make_mooney', 0.26577217343235809),
 (u'Make_other', 0.076421343844122108),
 (u'Make_piper', -0.24068047141883944)]
 '''

zip(feature_cols3, np.exp(logreg.coef_[0]))

'''
[('Lat', 0.98270264846944433),
 ('Long', 1.0011227429696883),
 ('Weather2', 5.0527995554553256),
 (u'Make_bell', 0.86966753895987614),
 (u'Make_boeing', 0.85723243105907632),
 (u'Make_cessna', 0.66341447653175811),
 (u'Make_mooney', 1.3044378392353546),
 (u'Make_other', 1.0794172841838514),
 (u'Make_piper', 0.78609276536914663)]
'''

y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

from sklearn.dummy import DummyClassifier
dumb = DummyClassifier(strategy='most_frequent')
dumb.fit(X_train, y_train)
y_dumb_class = dumb.predict(X_test)
print metrics.accuracy_score(y_test, y_dumb_class)

# testing accuracy = 0.6467, so less (marginally) from other feature combinations

# features combo 4
# Weather2, Lat, Long
combo4 = av_data

feature_cols4 = ['Lat', 'Long', 'Weather2']

X = combo4[feature_cols4]
y = combo4.Cas

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

logreg.fit(X_train, y_train)

zip(feature_cols4, logreg.coef_[0])

'''
[('Lat', -0.019081517268606758),
 ('Long', 0.0011839118356444698),
 ('Weather2', 1.6274031185346383)]
 '''

zip(feature_cols4, np.exp(logreg.coef_[0]))

'''
[('Lat', 0.9810993824410944),
 ('Long', 1.0011846129359148),
 ('Weather2', 5.0906377557292597)]
'''

y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

# features combo 5
# Weather2, Aircraft Cat
combo5 = pd.concat([av_data, aircat_dummies], axis=1)


feature_cols5 = ['Weather2', 'AirCat_Balloon',
       u'AirCat_Blimp', u'AirCat_Glider', u'AirCat_Gyrocraft',
       u'AirCat_Helicopter', u'AirCat_Powered Parachute',
       u'AirCat_Powered-Lift', u'AirCat_Ultralight', u'AirCat_Unknown',
       u'AirCat_Weight-Shift']

X = combo5[feature_cols5]
y = combo5.Cas

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

logreg.fit(X_train, y_train)

zip(feature_cols5, logreg.coef_[0])

'''
[('Weather2', 1.6950885056349101),
 ('AirCat_Balloon', 2.3530281182055179),
 (u'AirCat_Blimp', -0.0023308686668015938),
 (u'AirCat_Glider', 0.78326146533375984),
 (u'AirCat_Gyrocraft', -0.0011633440998347459),
 (u'AirCat_Helicopter', 0.25930624146647913),
 (u'AirCat_Powered Parachute', 2.7293267022127625),
 (u'AirCat_Powered-Lift', 0.6925921486127028),
 (u'AirCat_Ultralight', 5.5697292362022175),
 (u'AirCat_Unknown', -0.0013567248784613882),
 (u'AirCat_Weight-Shift', 1.8004081495980135)]
 '''

zip(feature_cols5, np.exp(logreg.coef_[0]))

'''
[('Weather2', 5.4471280454051714),
 ('AirCat_Balloon', 10.517369391582399),
 (u'AirCat_Blimp', 0.99767184569821699),
 (u'AirCat_Glider', 2.1885986766230396),
 (u'AirCat_Gyrocraft', 0.99883733232258309),
 (u'AirCat_Helicopter', 1.2960306423854151),
 (u'AirCat_Powered Parachute', 15.322566895653011),
 (u'AirCat_Powered-Lift', 1.9988902441089909),
 (u'AirCat_Ultralight', 262.3630512059529),
 (u'AirCat_Unknown', 0.99864419505665658),
 (u'AirCat_Weight-Shift', 6.0521171295558034)]
'''

y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

# testing accuracy = 0.6609 --> slightly better than first set of features (so excludes the long, lat, and make of aircraft)

# using combo 5 to predict probability of casualty occuring in a flight with IMC weather conditions on an Airplane:
logreg.predict_proba([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[:,1]
    # probability of casualty = 0.7316

# using combo 5 to predict probability of casualty occuring in a flight with VMC weather conditions on an Airplane:
logreg.predict_proba([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[:,1]
    # probability of casualty = 0.3335

# 

# Looking at tf-idf changes over 5 year periods, from 1980-2014
#


vect = TfidfVectorizer(stop_words = 'english', max_features=500)

textdata = alldata[['EventId', 'Year', 'narrative']]
textdata.dropna(inplace=True)

#1980-1984
text8084 = textdata[(textdata.Year >= 1980) & (textdata.Year < 1985)]
narr_dtm = vect.fit_transform(text8084.narrative)    # create dtm
dtm_cols = vect.get_feature_names()         # get tokens (words)
top_mean_feats(narr_dtm, dtm_cols)
'''
   feature     tfidf
0     acft  0.075131
1      plt  0.054576
2    pilot  0.051653
3     fuel  0.042694
4  landing  0.041076
5     gear  0.038765
6      rwy  0.038411
7   engine  0.036359
8     left  0.035024
9   runway  0.033609
'''

#1985-1989
text8589 = textdata[(textdata.Year >= 1985) & (textdata.Year < 1990)]
narr_dtm = vect.fit_transform(text8589.narrative)    # create dtm
dtm_cols = vect.get_feature_names()         # get tokens (words)
top_mean_feats(narr_dtm, dtm_cols)
'''
    feature     tfidf
0      acft  0.071933
1       plt  0.053501
2     pilot  0.050979
3      fuel  0.043412
4   landing  0.037571
5  aircraft  0.035959
6    engine  0.035580
7       rwy  0.035447
8      gear  0.034117
9      left  0.032393
'''

#1990-1994
text9094 = textdata[(textdata.Year >= 1990) & (textdata.Year < 1995)]
narr_dtm = vect.fit_transform(text9094.narrative)    # create dtm
dtm_cols = vect.get_feature_names()         # get tokens (words)
top_mean_feats(narr_dtm, dtm_cols)
'''
    feature     tfidf
0  airplane  0.073978
1     pilot  0.058863
2    engine  0.050708
3    runway  0.048621
4      fuel  0.046836
5   landing  0.039830
6  aircraft  0.033350
7      left  0.032952
8    flight  0.031611
9     right  0.030157
'''

#1995-1999
text9599 = textdata[(textdata.Year >= 1995) & (textdata.Year < 2000)]
narr_dtm = vect.fit_transform(text9599.narrative)    # create dtm
dtm_cols = vect.get_feature_names()         # get tokens (words)
top_mean_feats(narr_dtm, dtm_cols)
'''
    feature     tfidf
0  airplane  0.083955
1     pilot  0.064140
2      fuel  0.056458
3    runway  0.054466
4    engine  0.053990
5   landing  0.039200
6      left  0.036272
7    flight  0.036221
8     right  0.032978
9      gear  0.031479
'''

#2000-2004
text0004 = textdata[(textdata.Year >= 2000) & (textdata.Year < 2005)]
narr_dtm = vect.fit_transform(text0004.narrative)    # create dtm
dtm_cols = vect.get_feature_names()         # get tokens (words)
top_mean_feats(narr_dtm, dtm_cols)
'''
      feature     tfidf
0    airplane  0.098336
1       pilot  0.067908
2      runway  0.058116
3      engine  0.054263
4        fuel  0.051962
5      flight  0.041092
6     landing  0.041069
7  helicopter  0.037786
8        left  0.035890
9       right  0.033645
'''

#2005-2009
text0509 = textdata[(textdata.Year >= 2005) & (textdata.Year < 2010)]
narr_dtm = vect.fit_transform(text0509.narrative)    # create dtm
dtm_cols = vect.get_feature_names()         # get tokens (words)
top_mean_feats(narr_dtm, dtm_cols)
'''
      feature     tfidf
0    airplane  0.110222
1       pilot  0.073111
2      runway  0.056483
3      engine  0.055359
4        fuel  0.047895
5     landing  0.043900
6  helicopter  0.040960
7      flight  0.040768
8        left  0.034565
9        gear  0.033457
'''

#2010-2014
text1014 = textdata[(textdata.Year >= 2010) & (textdata.Year < 2015)]
narr_dtm = vect.fit_transform(text1014.narrative)    # create dtm
dtm_cols = vect.get_feature_names()         # get tokens (words)
top_mean_feats(narr_dtm, dtm_cols)
'''
      feature     tfidf
0    airplane  0.120522
1       pilot  0.077888
2      engine  0.055868
3        fuel  0.048517
4      runway  0.046728
5      flight  0.045079
6     landing  0.042742
7  helicopter  0.042603
8        left  0.033488
9        gear  0.032208
'''

# 'airplane' suddenly appearing and jumping to the top is kind of weird...
# looked at aircraft category from 1980-now

data[(data.Year >= 1980) & (data.Year < 1990)].groupby(data['Year']).AircraftCat.value_counts(dropna=False)

data[(data.Year >= 1990) & (data.Year < 2000)].groupby(data['Year']).AircraftCat.value_counts(dropna=False)

data[(data.Year >= 2000) & (data.Year < 2005)].groupby(data['Year']).AircraftCat.value_counts(dropna=False)

data[(data.Year >= 2005) & (data.Year < 2010)].groupby(data['Year']).AircraftCat.value_counts(dropna=False)

data[data.Year >= 2010].groupby(data['Year']).AircraftCat.value_counts(dropna=False)

data[(data.Year > 1980) & (data.AircraftCat == 'Airplane')].groupby(data['Year']).AircraftCat.count()

# seems like aircraft category was not reliably recorded until late 2000's, which is reflected in the narratives where investigators started to use an aircraft type in their commentary rather than the generic 'aircraft'.
