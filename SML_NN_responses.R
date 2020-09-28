library(dplyr)
library(ggplot2)
library(reshape2)
library(scales)

allData <- read.csv('data/NNresponses.csv', header=TRUE)
#summary(allData)



# Clean data

allData$chrono[allData$chrono == 'true'] <- '1'
allData$chrono[allData$chrono == 'null'] <- NA
allData$chrono <- as.numeric(allData$chrono)

allData$classified[allData$classified == 'true'] <- '1'
allData$classified[allData$classified == 'false'] <- '0'
allData$classified <- as.numeric(allData$classified)

allData$userPrefered[allData$userPrefered == 'true'] <- '1'
allData$userPrefered[allData$userPrefered == 'false'] <- '0'
allData$userPrefered[allData$userPrefered == 'NAN'] <- NA
allData$userPrefered <- as.numeric(allData$userPrefered)


allData$userResponse <- as.factor(allData$userResponse)

# Drop NAs

allData <- allData[!is.na(allData$classified),]

# Number of participants
nParts = length(unique(allData$uID))
# also get number of playlists per participant
nPlaylists = count(allData, uID)
median(nPlaylists$n)
min(nPlaylists$n)
max(nPlaylists$n)



# save cleaned data
#write.csv(allData, 'NNresponses_clean.csv', row.names=TRUE)



# Get subset of dataframe

fitData = allData[,c('userResponse', 'classified', 'userPrefered')]

fitData$classified <- fitData$classified+1
fitData$userPrefered <- fitData$userPrefered+1

# Get percentage responses
pcentResp <- fitData %>% count(userResponse)

pcentAgree = pcentResp$n[2]/(pcentResp$n[1]+pcentResp$n[2])
pcentDisagree = pcentResp$n[1]/(pcentResp$n[1]+pcentResp$n[2])

# make a circular distance error function

circDistError <- function(point1, point2) {
  if (is.na(point1)) {
    point1=1
  }
  
  if (is.na(point2)) {
    point2=point1
  }
  
  absDiff <- abs(diff(c(point1,point2)))
  if (absDiff >=3) {
    thisError = 5-absDiff
  } else {
    thisError = absDiff
  }
  return(thisError)
  
}

fitData$distance <- apply(fitData[,2:3], 1, function(x) circDistError(x['classified'], x['userPrefered']))

# Get error scores
error.mean = mean(fitData$distance)


error.1 = mean(subset(fitData$distance, fitData$classified==1))
error.2 = mean(subset(fitData$distance, fitData$classified==2))
error.3 = mean(subset(fitData$distance, fitData$classified==3))
error.4 = mean(subset(fitData$distance, fitData$classified==4))
error.5 = mean(subset(fitData$distance, fitData$classified==5))


# now calculate confusion matrix

fitData <- subset(fitData, !is.na(classified))


fitData$confusion <- fitData$userPrefered

fitData$confusion <- ifelse(is.na(fitData$confusion), fitData$classified, fitData$confusion)



confusionTable <- table(fitData[, c('classified', 'confusion')])

confusionTablePercentage <- proportions(confusionTable, 1)

confusionDF <- melt(confusionTablePercentage*100)

xLabels = c('Morning', 'Afternoon', 'Evening', 'Night', 'Late Night/Early Morning')
yLabels <- xLabels


# make plot
ggplot(confusionDF, aes(factor(classified), factor(confusion))) +
  geom_tile(aes(fill=value)) +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient2(low = "white", 
                       mid = muted("pink"),
                       high = muted("darkred"),
                       midpoint = 60) +
  xlab('Predicted') +
  ylab('Participants response') +
  labs(title='Predicted subdivision vs participants response', subtitle='Predicted class, and users responses, in percentage',
       fill='Percentage') +
  scale_x_discrete(labels=xLabels) + 
  scale_y_discrete(labels=yLabels)

