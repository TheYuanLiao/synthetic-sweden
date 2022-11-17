# Title     : Temporal distribution of power
# Objective : Charging strategy x temporal power profile
# Created by: Yuan Liao
# Created on: 2022-11-16

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
power <- read.csv(paste0("results/", scenario, "_stats_power_5days.csv"))
power.50 <- power %>%
  filter(power_fast == 50) %>%
  group_by(minute, Charging_type) %>%
  summarise(power=sum(power)) %>%
  mutate(power_fast = 50)

power.50 <- power.50 %>%
  filter(Charging_type %in% c('1 (0.2)', '2 (0.2)', '3 (0.9)'))
power.50$Charging_type <- plyr::mapvalues(power.50$Charging_type,
                                     from=c('1 (0.2)', '2 (0.2)', '3 (0.9)'),
                                     to=c("1 Liquid-fuel","2 Plan-ahead","3 Event-actuated"))

cols <- c('#242676', '#037540', 'purple')
g <- ggplot(data=power.50,
             aes(x=minute/60, y=power/10^3)) +
  scale_color_manual(name='Charging strategy', values = cols) +
  geom_line(aes(color=Charging_type)) +
  labs(x = 'Time (hour)', y = 'Power demand (MW)') +
  theme_minimal() +
  theme(legend.position = c(0.8, 0.7))

ggsave(filename = paste0("figures/", scenario, "_inf_tempo.png"), plot=g,
       width = 6, height = 4, unit = "in", dpi = 300)