library(twang)
mydata<-out[,-1]

treat2<-mydata['treat2']
esmo2020_mol<-mydata['esmo2020_mol']
lvi<-mydata['lvi']
MyoInvasion<-mydata['MyoInvasion']
hist_surgery_dx<-mydata['hist_surgery_dx']
grade_surgery_b12v3<-mydata['grade_surgery_b12v3']
stage<-mydata['stage']
ace_27_any<-mydata['ace_27_any']
tr<-treat2~esmo2020_mol+lvi+MyoInvasion+hist_surgery_dx+grade_surgery_b12v3+stage+ace_27_any

mydata$treat2 <- factor(mydata$treat2)
mydata$esmo2020_mol <- factor(mydata$esmo2020_mol)
mydata$lvi <- factor(mydata$lvi)
mydata$MyoInvasion <- factor(mydata$MyoInvasion)
mydata$hist_surgery_dx <- factor(mydata$hist_surgery_dx)
mydata$grade_surgery_b12v3 <- factor(mydata$grade_surgery_b12v3)
mydata$stage <- factor(mydata$stage)
mydata$ace_27_any <- factor(mydata$ace_27_any)

alpha <- twang::mnps(
  formula = tr,
  data = mydata,
  n.trees = 2e4,
  estimand = "ATE",
  verbose = FALSE,
  version = "gbm",
  stop.method = "es.max"
)

tmp1 <- alpha[["psList"]][["Any Chemo"]][["ps"]]
tmp2 <- alpha[["psList"]][["NoneOrBrachyOnly"]][["ps"]]
tmp3 <- alpha[["psList"]][["RT +/- brachy"]][["ps"]]
psout<-data.frame(tmp2, tmp1, tmp3)
write.csv(psout,file='ps2.csv')
