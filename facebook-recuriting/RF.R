
## Prediction
## outcome ??? Label of a bidder indicating whether or not it is a robot. Value 1.0 indicates a robot, where value 0.0 indicates human. 

### setting path of repo folder.
setwd("/Users/bikash/repos/kaggle/facebook-recuriting/")
#setwd("/home/ekstern/haisen/bikash/kaggle/RestaurantRevenuePrediction/")

library(party)
library(e1071)
library(lubridate)
library(Boruta)
library(gtools)
#load data
train = read.csv("data/train.csv", header = TRUE, stringsAsFactors = FALSE)
test = read.csv("data/test.csv", header = TRUE, stringsAsFactors = FALSE)

bids = read.csv("data/bids.csv", header = TRUE, stringsAsFactors = FALSE)

df_train = merge( train, bids, by='bidder_id', all=FALSE, sort= T)

df_test = merge( test, bids, by='bidder_id', all=FALSE, sort= T)

###

MERCHANDISE_TYPES = c('auto parts', 'books and music', 'clothing', 'computers',
                     'furniture', 'home goods', 'jewelry', 'mobile',
                     'office equipment', 'sporting goods')

COUNTRIES = c('ad', 'ae', 'af', 'ag', 'al', 'am', 'an', 'ao', 'ar', 'at', 'au',
             'aw', 'az', 'ba', 'bb', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj',
             'bm', 'bn', 'bo', 'br', 'bs', 'bt', 'bw', 'by', 'bz', 'ca', 'cd',
             'cf', 'cg', 'ch', 'ci', 'cl', 'cm', 'cn', 'co', 'cr', 'cv', 'cy',
             'cz', 'de', 'dj', 'dk', 'dm', 'do', 'dz', 'ec', 'ee', 'eg', 'er',
             'es', 'et', 'eu', 'fi', 'fj', 'fo', 'fr', 'ga', 'gb', 'ge', 'gh',
             'gi', 'gl', 'gm', 'gn', 'gp', 'gq', 'gr', 'gt', 'gu', 'gy', 'hk',
             'hn', 'hr', 'ht', 'hu', 'id', 'ie', 'il', 'in', 'iq', 'ir', 'is',
             'it', 'je', 'jm', 'jo', 'jp', 'ke', 'kg', 'kh', 'kr', 'kw', 'kz',
             'la', 'lb', 'li', 'lk', 'lr', 'ls', 'lt', 'lu', 'lv', 'ly', 'ma',
             'mc', 'md', 'me', 'mg', 'mh', 'mk', 'ml', 'mm', 'mn', 'mo', 'mp',
             'mr', 'mt', 'mu', 'mv', 'mw', 'mx', 'my', 'mz', 'na', 'nc', 'ne',
             'ng', 'ni', 'nl', 'no', 'np', 'nz', 'om', 'pa', 'pe', 'pf', 'pg',
             'ph', 'pk', 'pl', 'pr', 'ps', 'pt', 'py', 'qa', 're', 'ro', 'rs',
             'ru', 'rw', 'sa', 'sb', 'sc', 'sd', 'se', 'sg', 'si', 'sk', 'sl',
             'sn', 'so', 'sr', 'sv', 'sy', 'sz', 'tc', 'td', 'tg', 'th', 'tj',
             'tl', 'tm', 'tn', 'tr', 'tt', 'tw', 'tz', 'ua', 'ug', 'uk', 'us',
             'uy', 'uz', 'vc', 've', 'vi', 'vn', 'ws', 'ye', 'za', 'zm', 'zw',
             'zz')
list_country <- data.frame(id=c(1:199),country=COUNTRIES)
list_merchandise <- data.frame(id=c(1:length(MERCHANDISE_TYPES)),merchandise=MERCHANDISE_TYPES)
##
head(df_train)
library(plyr)
library(data.table)

a <- df_train[c(1:200),]
library(sqldf)
sqldf("Select bidder_id, count(device) as n_devices, count(country) as n_countries, count(auction) as n_auctions,
     count(ip) as n_ip, count(url) as n_url from a Group by bidder_id")

train1 <- sqldf("Select bidder_id, outcome, count(distinct(device)) as n_devices, 
      count(distinct(country)) as n_countries from a Group by bidder_id")

#labels_to_write = ['bidder_id', 'n_auctions', 'n_per_auc', 'n_devices', 'country', 'n_countries', 'n_ip', 'n_url']


library(reshape)
df <- cast(df_train, bidder_id~device,count)
ddply(df_train,.(unique(bidder_id)),summarise,
      count = count(1))

## Random Forest
set.seed(12)

rf <- randomForest(as.factor(outcome) ~ ., data=train2, ntree=1000, method = "class")


#Make a Prediction
rf.pred = predict(rf, mydata[-c(1:135), ], OOB=TRUE, type = "response")

#ddply(df_train,.(bidder_id),summarise,merchandise=unique(merchandise),country=unique(country))


##
summary(df_train$device) #3071224
as.factor(df_train$device)
as.factor(df_train$country)
as.factor(df_train$merchandise)

df_train$is.Robot <- ifelse(round(df_train$outcome,0)==1,1,0)

head(df_train[df_train$outcome == 1,] ) #display robot



#uto parts books and music clothing computers furniture home goods jewelry mobile office equipment sporting goods
## Random Forest