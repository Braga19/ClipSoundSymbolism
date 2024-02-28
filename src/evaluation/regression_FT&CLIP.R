library(ggplot2)
library(tidyverse)
library(plyr)
library(dplyr)
library(ggeffects)
library(effectsize)
require(gridExtra)



##### READ DATA #####
load('data_cogSci.RData')
setwd("./plots/")

# robustness check excluding madeup names tokenized as a single token.
# in order to run analyses excluding these names, uncomment and run the next four lines 

# "%ni%" <- Negate("%in%")
# problematic_madeup_names = c('argus', 'asim', 'brum', 'chi', 'kaz', 'lak', 'tasha')
# df.gender.aggr.subset = df.gender.aggr[df.gender.aggr$name %ni% problematic_madeup_names,]
# df.age.aggr.subset = df.age.aggr[df.age.aggr$name %ni% problematic_madeup_names,]

# and then, in all following linear models: 
# replace df.gender.aggr with df.gender.aggr.subset
# replace df.age.aggr with df.age.aggr.subset

##### GENDER #####
# language baseline
lm.gender.weat = lm(
  true_rating ~ type + weat_score, 
  data = df.gender.aggr[df.gender.aggr$classifier == 'leilab' & 
                                 df.gender.aggr$img_generator == 'vqgan',]
  )
summary(lm.gender.weat)
AIC(lm.gender.weat) # 201.9991
# ----- VQGAN+CLIP -----
# _____ leilab _____
lm.gender.weat_clip.vc.leilab = lm(
  true_rating ~ type + weat_score + avg_female_prob,
  data = df.gender.aggr[df.gender.aggr$classifier == 'leilab' & 
                                 df.gender.aggr$img_generator == 'vqgan',]
  )
summary(lm.gender.weat_clip.vc.leilab)
AIC(lm.gender.weat_clip.vc.leilab) # 175.7536

lm.gender.int.vc.leilab = lm(
  true_rating ~ type*avg_female_prob + weat_score,
  data = df.gender.aggr[df.gender.aggr$classifier == 'leilab' & 
                                 df.gender.aggr$img_generator == 'vqgan',])
summary(lm.gender.int.vc.leilab)
AIC(lm.gender.int.vc.leilab) # 179.5684

eta_squared(lm.gender.weat_clip.vc.leilab, partial = T)
# Parameter       | Eta2 (partial) |       95% CI
# type            |           0.03 | [0.00, 1.00]
# weat_score      |           0.72 | [0.66, 1.00]
# avg_female_prob |           0.15 | [0.07, 1.00]


# _____ rizvan _____
lm.gender.weat_clip.vc.rizvan = lm(
  true_rating ~ type + weat_score + avg_female_prob,
  data = df.gender.aggr[df.gender.aggr$classifier == 'rizvan' & 
                                 df.gender.aggr$img_generator == 'vqgan',]
  )
summary(lm.gender.weat_clip.vc.rizvan)
AIC(lm.gender.weat_clip.vc.rizvan) # 177.2609

lm.gender.int.vc.rizvan = lm(
  true_rating ~ type*avg_female_prob + weat_score,
  data = df.gender.aggr[df.gender.aggr$classifier == 'rizvan' & 
                                 df.gender.aggr$img_generator == 'vqgan',])
summary(lm.gender.int.vc.rizvan)
AIC(lm.gender.int.vc.rizvan) # 180.9631

eta_squared(lm.gender.weat_clip.vc.rizvan, partial = T)
# Parameter       | Eta2 (partial) |       95% CI
# type            |           0.03 | [0.00, 1.00]
# weat_score      |           0.71 | [0.66, 1.00]
# avg_female_prob |           0.14 | [0.07, 1.00]


# _____ crangana _____
lm.gender.weat_clip.vc.crangana = lm(
  true_rating ~ type + weat_score + avg_female_prob,
  data = df.gender.aggr[df.gender.aggr$classifier == 'crangana' & 
                                 df.gender.aggr$img_generator == 'vqgan',]
  )
summary(lm.gender.weat_clip.vc.crangana)
AIC(lm.gender.weat_clip.vc.crangana) # 182.6165

lm.gender.int.vc.crangana = lm(
  true_rating ~ type*avg_female_prob + weat_score,
  data = df.gender.aggr[df.gender.aggr$classifier == 'crangana' & 
                                 df.gender.aggr$img_generator == 'vqgan',]
  )
summary(lm.gender.int.vc.crangana)
AIC(lm.gender.int.vc.crangana) # 186.5948

eta_squared(lm.gender.weat_clip.vc.crangana, partial = T)
# Parameter       | Eta2 (partial) |       95% CI
# type            |           0.03 | [0.00, 1.00]
# weat_score      |           0.71 | [0.65, 1.00]
# avg_female_prob |           0.11 | [0.05, 1.00]


# ----- STABLE DIFFUSION -----
# _____ leilab _____
lm.gender.weat_clip.sd.leilab = lm(
  true_rating ~ type + weat_score + avg_female_prob,
  data = df.gender.aggr[df.gender.aggr$classifier == 'leilab' & 
                                 df.gender.aggr$img_generator == 'sd',]
  )
summary(lm.gender.weat_clip.sd.leilab)
AIC(lm.gender.weat_clip.sd.leilab) # 141.1825

lm.gender.int.sd.leilab = lm(
  true_rating ~ type*avg_female_prob + weat_score,
  data = df.gender.aggr[df.gender.aggr$classifier == 'leilab' & 
                                 df.gender.aggr$img_generator == 'sd',]
  )
summary(lm.gender.int.sd.leilab)
AIC(lm.gender.int.sd.leilab) # 145.0035

eta_squared(lm.gender.weat_clip.sd.leilab, partial = T)
# Parameter       | Eta2 (partial) |       95% CI
# type            |           0.03 | [0.00, 1.00]
# weat_score      |           0.75 | [0.71, 1.00]
# avg_female_prob |           0.30 | [0.21, 1.00]

# _____ rizvan _____
lm.gender.weat_clip.sd.rizvan = lm(
  true_rating ~ type + weat_score + avg_female_prob,
  data = df.gender.aggr[df.gender.aggr$classifier == 'rizvan' & 
                                 df.gender.aggr$img_generator == 'sd',]
  )
summary(lm.gender.weat_clip.sd.rizvan)
AIC(lm.gender.weat_clip.sd.rizvan) #  131.271

lm.gender.int.sd.rizvan = lm(
  true_rating ~ type*avg_female_prob + weat_score,
  data = df.gender.aggr[df.gender.aggr$classifier == 'rizvan' & 
                                 df.gender.aggr$img_generator == 'sd',]
  )
summary(lm.gender.int.sd.rizvan)
AIC(lm.gender.int.sd.rizvan) # 135.0802

eta_squared(lm.gender.weat_clip.sd.rizvan, partial = T)
# Parameter       | Eta2 (partial) |       95% CI
# type            |           0.04 | [0.00, 1.00]
# weat_score      |           0.76 | [0.72, 1.00]
# avg_female_prob |           0.33 | [0.24, 1.00]

# _____ crangana _____
lm.gender.weat_clip.sd.crangana = lm(
  true_rating ~ type + weat_score + avg_female_prob,
  data = df.gender.aggr[df.gender.aggr$classifier == 'crangana' & 
                                 df.gender.aggr$img_generator == 'sd',]
  )
summary(lm.gender.weat_clip.sd.crangana)
AIC(lm.gender.weat_clip.sd.crangana) # 147.3883

lm.gender.int.sd.crangana = lm(
  true_rating ~ type*avg_female_prob + weat_score,
  data = df.gender.aggr[df.gender.aggr$classifier == 'crangana' & 
                                 df.gender.aggr$img_generator == 'sd',]
  )
summary(lm.gender.int.sd.crangana)
AIC(lm.gender.int.sd.crangana) # 151.2761

eta_squared(lm.gender.weat_clip.sd.crangana, partial = T)
# Parameter       | Eta2 (partial) |       95% CI
# type            |           0.03 | [0.00, 1.00]
# weat_score      |           0.75 | [0.70, 1.00]
# avg_female_prob |           0.27 | [0.18, 1.00]

hist(lm.gender.int.sd.crangana$residuals)

gender.preds.weat = data.frame(
  ggpredict(lm.gender.weat_clip.sd.rizvan, terms = c(
    "weat_score[all]",
    "type[all]")
  )
)
names(gender.preds.weat)[names(gender.preds.weat) == 'x'] <- 'weat_score'
names(gender.preds.weat)[names(gender.preds.weat) == 'group'] <- 'type'

pdf(file = "lm_gender_rizvan_sd_weatFemale.pdf",
    width = 4.5, 
    height = 3.5)
ggplot(data = gender.preds.weat, aes(x = weat_score, y = predicted)) +
  geom_line(aes(color = type, linetype = type)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = type, fill = type, linetype = type), alpha = 0.3) +
  geom_point(aes(x = weat_score, y = true_rating, color = type), size = 0.5, 
             data=df.gender.aggr[df.gender.aggr$classifier == 'rizvan' & df.gender.aggr$img_generator == 'sd',]) +
  scale_fill_manual(values = c(
    "made-up" = "turquoise4",
    "real" = "firebrick3",
    "talking" = "darkorange1"
    )) +
  scale_color_manual(values = c(
    "made-up" = "turquoise4",
    "real" = "firebrick3",
    "talking" = "darkorange1"
  )) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
  ) +
  labs(x = 'FT-based age delta (ngrams)', y = 'gender rating (normalized)')
dev.off()
#       title = 'Relation between text-based gender delta and normalised gender ratings',
#       subtitle = 'Text model: FastText, corpus: CoCA; n-gram size: 2 to 5')

gender.preds.sd = data.frame(
  ggpredict(lm.gender.weat_clip.sd.rizvan, terms = c(
    "avg_female_prob[all]",
    "type[all]")
  )
)
names(gender.preds.sd)[names(gender.preds.sd) == 'x'] <- 'avg_female_prob'
names(gender.preds.sd)[names(gender.preds.sd) == 'group'] <- 'type'

pdf(file = "lm_gender_rizvan_sd_pFemale.pdf",
    width = 4.5, 
    height = 3.5)
p_female = ggplot(data = gender.preds.sd, aes(x = avg_female_prob, y = predicted)) +
  geom_line(aes(color = type, linetype = type)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = type, fill = type, linetype = type), alpha = 0.3) +
  geom_point(aes(x = avg_female_prob, y = true_rating, color = type), size = 0.5, 
             data=df.gender.aggr[df.gender.aggr$classifier == 'rizvan' & df.gender.aggr$img_generator == 'sd',]) +
  scale_fill_manual(values = c(
    "made-up" = "turquoise4",
    "real" = "firebrick3",
    "talking" = "darkorange1"
  )) +
  scale_color_manual(values = c(
    "made-up" = "turquoise4",
    "real" = "firebrick3",
    "talking" = "darkorange1"
  )) +
  theme(
    legend.position = 'none', 
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
  ) +
  labs(x = 'p(female) - rizvan and SD', y = 'gender rating (normalized)') + 
  geom_text(x=0.03, y=0.87, label="A", size = 10)
dev.off()
#       title = 'Relation between average probability of images featuring female faces and gender ratings',
#       subtitle = 'image generator: Stable Diffusion; gender classifier: rizvan')


##### AGE #####
# language baseline
lm.age.weat = lm(
  true_rating ~ type + weat_score, 
  data = df.age.aggr[df.age.aggr$classifier == 'ibombSwin' & 
                              df.age.aggr$img_generator == 'vqgan',])
summary(lm.age.weat)
AIC(lm.age.weat) # 108.8981

lm.age.weat_int = lm(
  true_rating ~ type*weat_score, 
  data = df.age.aggr[df.age.aggr$classifier == 'ibombSwin' & 
                              df.age.aggr$img_generator == 'vqgan',]
  )
summary(lm.age.weat_int)
AIC(lm.age.weat_int) # 108.373
# ----- VQGAN+CLIP -----
# _____ ibombSwin _____
lm.age.weat_clip.vc.ibombSwin = lm(
  true_rating ~ type + weat_score + avg_old_prob,
  data = df.age.aggr[df.age.aggr$classifier == 'ibombSwin' & 
                              df.age.aggr$img_generator == 'vqgan',]
  )
summary(lm.age.weat_clip.vc.ibombSwin)
AIC(lm.age.weat_clip.vc.ibombSwin) # 100.6902

lm.age.int.vc.ibombSwin = lm(
  true_rating ~ type*avg_old_prob + weat_score,
  data = df.age.aggr[df.age.aggr$classifier == 'ibombSwin' & 
                              df.age.aggr$img_generator == 'vqgan',]
  )
summary(lm.age.int.vc.ibombSwin)
AIC(lm.age.int.vc.ibombSwin) # 104.4951

eta_squared(lm.age.weat_clip.vc.ibombSwin, partial = T)
# Parameter    | Eta2 (partial) |       95% CI
# type         |           0.07 | [0.01, 1.00]
# weat_score   |           0.14 | [0.06, 1.00]
# avg_old_prob |           0.08 | [0.02, 1.00]


# _____ cranage _____
lm.age.weat_clip.vc.cranage = lm(
  true_rating ~ type + weat_score + avg_old_prob,
  data = df.age.aggr[df.age.aggr$classifier == 'cranage' & 
                              df.age.aggr$img_generator == 'vqgan',]
  )
summary(lm.age.weat_clip.vc.cranage)
AIC(lm.age.weat_clip.vc.cranage) # 89.1622

lm.age.int.vc.cranage = lm(
  true_rating ~ type*avg_old_prob + weat_score,
  data = df.age.aggr[df.age.aggr$classifier == 'cranage' & 
                              df.age.aggr$img_generator == 'vqgan',]
  )
summary(lm.age.int.vc.cranage)
AIC(lm.age.int.vc.cranage) # 90.28632

eta_squared(lm.age.weat_clip.vc.cranage, partial = T)
# Parameter    | Eta2 (partial) |       95% CI
# type         |           0.08 | [0.01, 1.00]
# weat_score   |           0.15 | [0.06, 1.00]
# avg_old_prob |           0.17 | [0.08, 1.00]


# _____ nateraw _____
lm.age.weat_clip.vc.nateraw = lm(
  true_rating ~ type + weat_score + avg_old_prob,
  data = df.age.aggr[df.age.aggr$classifier == 'nateraw' & 
                              df.age.aggr$img_generator == 'vqgan',]
  )
summary(lm.age.weat_clip.vc.nateraw)
AIC(lm.age.weat_clip.vc.nateraw) # 74.23817

lm.age.int.vc.nateraw = lm(
  true_rating ~ type*avg_old_prob + weat_score,
  data = df.age.aggr[df.age.aggr$classifier == 'nateraw' & 
                              df.age.aggr$img_generator == 'vqgan',]
  )
summary(lm.age.int.vc.nateraw)
AIC(lm.age.int.vc.nateraw) # 77.89363

eta_squared(lm.age.weat_clip.vc.nateraw, partial = T)
# Parameter    | Eta2 (partial) |       95% CI
# type         |           0.09 | [0.02, 1.00]
# weat_score   |           0.17 | [0.08, 1.00]
# avg_old_prob |           0.27 | [0.16, 1.00]


# ----- STABLE DIFFUSION -----
# _____ ibombSwin _____
lm.age.weat_clip.sd.ibombSwin = lm(
  true_rating ~ type + weat_score + avg_old_prob,
  data = df.age.aggr[df.age.aggr$classifier == 'ibombSwin' & 
                              df.age.aggr$img_generator == 'sd',]
  )
summary(lm.age.weat_clip.sd.ibombSwin)
AIC(lm.age.weat_clip.sd.ibombSwin) # 69.22431

lm.age.int.sd.ibombSwin = lm(
  true_rating ~ type*avg_old_prob + weat_score,
  data = df.age.aggr[df.age.aggr$classifier == 'ibombSwin' & 
                              df.age.aggr$img_generator == 'sd',]
  )
summary(lm.age.int.sd.ibombSwin)
AIC(lm.age.int.sd.ibombSwin) # 70.6222

eta_squared(lm.age.weat_clip.sd.ibombSwin, partial = T)
# Parameter    | Eta2 (partial) |       95% CI
# type         |           0.09 | [0.02, 1.00]
# weat_score   |           0.18 | [0.08, 1.00]
# avg_old_prob |           0.30 | [0.19, 1.00]


# _____ cranage _____
lm.age.weat_clip.sd.cranage = lm(
  true_rating ~ type + weat_score + avg_old_prob,
  data = df.age.aggr[df.age.aggr$classifier == 'cranage' & 
                              df.age.aggr$img_generator == 'sd',]
  )
summary(lm.age.weat_clip.sd.cranage)
AIC(lm.age.weat_clip.sd.cranage) #  69.64367

lm.age.int.sd.cranage = lm(
  true_rating ~ type*avg_old_prob + weat_score,
  data = df.age.aggr[df.age.aggr$classifier == 'cranage' & 
                              df.age.aggr$img_generator == 'sd',]
  )
summary(lm.age.int.sd.cranage)
AIC(lm.age.int.sd.cranage) # 69.91328

eta_squared(lm.age.weat_clip.sd.cranage, partial = T)
# Parameter    | Eta2 (partial) |       95% CI
# type         |           0.09 | [0.02, 1.00]
# weat_score   |           0.18 | [0.08, 1.00]
# avg_old_prob |           0.29 | [0.18, 1.00]

# _____ nateraw _____
lm.age.weat_clip.sd.nateraw = lm(
  true_rating ~ type + weat_score + avg_old_prob,
  data = df.age.aggr[df.age.aggr$classifier == 'nateraw' & 
                              df.age.aggr$img_generator == 'sd',]
  )
summary(lm.age.weat_clip.sd.nateraw)
AIC(lm.age.weat_clip.sd.nateraw) # 73.37629

lm.age.int.sd.nateraw = lm(
  true_rating ~ type*avg_old_prob + weat_score,
  data = df.age.aggr[df.age.aggr$classifier == 'nateraw' & 
                              df.age.aggr$img_generator == 'sd',]
  )
summary(lm.age.int.sd.nateraw)
AIC(lm.age.int.sd.nateraw) # 74.75282

eta_squared(lm.age.weat_clip.sd.nateraw, partial = T)
# Parameter    | Eta2 (partial) |       95% CI
# type         |           0.09 | [0.02, 1.00]
# weat_score   |           0.17 | [0.08, 1.00]
# avg_old_prob |           0.27 | [0.16, 1.00]


age.preds.weat = data.frame(
  ggpredict(lm.age.weat_clip.sd.ibombSwin, terms = c(
    "weat_score[all]",
    "type[all]")
  )
)
names(age.preds.weat)[names(age.preds.weat) == 'x'] <- 'weat_score'
names(age.preds.weat)[names(age.preds.weat) == 'group'] <- 'type'

pdf(file = "lm_age_ibombSwin_sd_weatOld.pdf",
    width = 4.5, 
    height = 3.5)
ggplot(data = age.preds.weat, aes(x = weat_score, y = predicted)) +
  geom_line(aes(color = type, linetype = type)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = type, fill = type, linetype = type), alpha = 0.3) +
  geom_point(aes(x = weat_score, y = true_rating, color = type), size = 0.5, 
             data=df.age.aggr[df.age.aggr$classifier == 'ibombSwin' & df.age.aggr$img_generator == 'sd',]) +
  scale_fill_manual(values = c(
    "made-up" = "turquoise4",
    "real" = "firebrick3",
    "talking" = "darkorange1"
  )) +
  scale_color_manual(values = c(
    "made-up" = "turquoise4",
    "real" = "firebrick3",
    "talking" = "darkorange1"
  )) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
  ) +
  labs(x = 'FT-based age delta (ngrams)', y = 'age rating (normalized)')
dev.off()
#       title = 'Relation between text-based age delta and normalised age ratings',
#       subtitle = 'Text model: FastText, corpus: CoCA; n-gram size: 2 to 5')

age.preds.sd = data.frame(
  ggpredict(lm.age.weat_clip.sd.ibombSwin, terms = c(
    "avg_old_prob[all]",
    "type[all]")
  )
)
names(age.preds.sd)[names(age.preds.sd) == 'x'] <- 'avg_old_prob'
names(age.preds.sd)[names(age.preds.sd) == 'group'] <- 'type'

pdf(file = "lm_age_ibombSwin_sd_pOld.pdf",
    width = 4.5, 
    height = 3.5)
p_old = ggplot(data = age.preds.sd, aes(x = avg_old_prob, y = predicted)) +
  geom_line(aes(color = type, linetype = type)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = type, fill = type, linetype = type), alpha = 0.3) +
  geom_point(aes(x = avg_old_prob, y = true_rating, color = type), size = 0.5, 
             data=df.age.aggr[df.age.aggr$classifier == 'ibombSwin' & df.age.aggr$img_generator == 'sd',]) +
  scale_fill_manual(values = c(
    "made-up" = "turquoise4",
    "real" = "firebrick3",
    "talking" = "darkorange1"
  ), name = 'Name type') +
  scale_color_manual(values = c(
    "made-up" = "turquoise4",
    "real" = "firebrick3",
    "talking" = "darkorange1"
  ), name = 'Name type') +
  scale_linetype(name = 'Name type') +
  theme(
    legend.justification = 'right', 
    legend.position = c(0.98, 0.2), 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    legend.background = element_rect(fill='transparent'),
    legend.box.background = element_rect(color = NA, fill = 'transparent'),
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
  ) +
  labs(x = 'p(old) - ibombSwin and SD', y = 'age rating (normalized)') +
  geom_text(x=0.03, y=0.87, label="B", size = 10) +
  ylim(c(-1, 1)) +
  guides(fill = guide_legend(title.hjust = 1, label.position = "left", label.hjust = 1),
         colour = guide_legend(title.hjust = 1, label.position = "left", label.hjust = 1),
         linetype = guide_legend(title.hjust = 1, label.position = "left", label.hjust = 1))
dev.off()
#       title = 'Relation between average probability of images featuring old faces and age ratings',
#       subtitle = 'image generator: Stable Diffusion; age classifier: ibombSwin')

p_old

pdf(file = "lm_regressionEffects_crossModal_horiz.pdf",
    width = 9, 
    height = 3.5)
grid.arrange(p_female, p_old, ncol=2)
dev.off()
