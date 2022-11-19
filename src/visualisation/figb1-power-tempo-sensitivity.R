# Title     : Temporal distribution of chargers for sensitivity test
# Objective : Power x charging strategy
# Created by: Yuan Liao
# Created on: 2022-08-07

library(dplyr)
library(tidyr)
library(sp)
library(spdep)
library(ggplot2)
library(ggpubr)
library(ggmap)
library(ggspatial)
library(ggsn)
library(viridis)
library(animation)
library(units)
options(scipen=10000)

scenario <- "scenario_vg_car"
# Temporal distribution of power
power <- read.csv(paste0("results/sensitivity/", scenario, "_stats_power_5days.csv"))
power2plot <- power %>%
  group_by(minute, Charging_type, power_fast) %>%
  summarise(power=sum(power))


cols <- c('#242676', '#3639b1', '#037540', '#05c069', 'purple')
g <- ggplot(data=power2plot,
             aes(x=minute/60, y=power/10^3)) +
#  scale_color_manual(name='Charging strategy', values = cols) +
  geom_line(aes(color=Charging_type, linetype=as.factor(power_fast))) +
  scale_color_manual("Charging strategy (threshold)", values = cols) +
  scale_linetype_discrete("Power of fast chargers (kW)") +
  labs(x = 'Time (hour)', y = 'Power demand (MW)') +
  theme_minimal() +
  theme(legend.position = 'right')

ggsave(filename = paste0("figures/", scenario, "_power_tempo_sensitivity.png"), plot=g,
       width = 8, height = 4, unit = "in", dpi = 300)