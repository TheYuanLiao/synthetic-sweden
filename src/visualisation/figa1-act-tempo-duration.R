# Title     : Temporal distribution of activities and parking duration
# Objective : Number of agents per minute
# Created by: Yuan Liao
# Created on: 2022-11-18

library(dplyr)
library(ggplot2)
library(ggpubr)
options(scipen=10000)

scenario <- "scenario_vg_car"
# Temporal distribution of activity
act.tempo <- read.csv(paste0("results/mobility_patterns/", scenario, "_act_tempo.csv"))
# Car parking duration
df <- read.csv(paste0("results/mobility_patterns/", scenario, "_car_trips.csv"))

act.tempo$act_purpose <- plyr::mapvalues(act.tempo$act_purpose,
                                        from=c('home', 'other', 'school', 'work'),
                                        to=c('H', 'O', 'S', 'W'))
df$act_purpose <- plyr::mapvalues(df$act_purpose,
                                  from=c('home', 'other', 'school', 'work'),
                                  to=c('H', 'O', 'S', 'W'))

cols <- c('#44bd32', '#0097e6', '#8c7ae6', '#40739e')
g1 <- ggplot(data = act.tempo, aes(x=minute/60, y=freq/1000, color=act_purpose)) +
  scale_color_manual("Activity", values = cols) +
  labs(x='Hour of day (h)', y='# of agents (thousand)') +
  geom_line(size=0.7) +
  theme_minimal() +
  theme(legend.position = c(0.2, 0.5))

g2 <- ggplot(data=df) +
  geom_freqpoly(aes(x=act_time/60, color = act_purpose), size=0.7, bins=40) +
  scale_color_manual(values = cols) +
  labs(x='Parking duration (h)', y='Activity count') +
  theme_minimal()

G <- ggarrange(g1, g2, ncol = 2, nrow = 1, common.legend = T, legend="bottom", labels = c('(a)', '(b)'))
ggsave(filename = paste0("figures/", scenario, "_mobi_act.png"), plot=G,
       width = 10, height = 4, unit = "in", dpi = 300)