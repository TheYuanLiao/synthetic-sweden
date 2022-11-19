# Title     : Daily travel distance
# Objective : Histogram
# Created by: Yuan Liao
# Created on: 2022-11-18

library(dplyr)
library(ggplot2)
library(ggpubr)
options(scipen=10000)

scenario <- "scenario_vg_car"
# Car trips
df <- read.csv(paste0("results/mobility_patterns/", scenario, "_car_trips.csv"))
df.d <- df %>%
  group_by(PId) %>%
  summarise(d = sum(distance))

g <- ggplot(data=df.d, aes(x=d)) +
  geom_histogram(fill='white', color='gray40', size=0.3, bins=100) +
  labs(x='Daily travel distance in log scale (km)', y='# of agents') +
  scale_x_log10() +
  theme_minimal()

ggsave(filename = paste0("figures/", scenario, "_mobi_distance.png"), plot=g,
       width = 6, height = 4, unit = "in", dpi = 300)