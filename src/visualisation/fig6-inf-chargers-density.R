# Title     : Density distribution of chargers
# Objective : Charging strategy x density (deso zone)
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
charging <- read.csv(paste0("results/", scenario, "_stats_charger_5days.csv"))
zones <- read_sf('dbs/DeSO/DeSO_2018_v2.shp')
zones <- zones[zones$deso %in% unique(charging$deso),]
zones$Area <- st_area(zones)

# Charging density by purpose
tst <- charging %>%
  filter(power_fast==50) %>%
  group_by(deso, Charging_type, Purpose) %>%
  summarise(total=sum(number)) %>%
  filter(Charging_type %in% c('1 (0.2)', '2 (0.2)', '3 (0.9)'))
tst$Charging_type <- plyr::mapvalues(tst$Charging_type,
                                        from=c('1 (0.2)', '2 (0.2)', '3 (0.9)'),
                                        to=c("1 Liquid-fuel","2 Plan-ahead","3 Event-actuated"))
tst <- right_join(x = zones[, c('deso', 'Area')]%>% st_set_geometry(NULL),
                                             y = tst,
                                   by = "deso", all.y = TRUE)
tst$Charge_point_density <- tst$total / (as.numeric(tst$Area) / 10^6)# #/km^2

# Charging density overall
tst.all <- charging %>%
  filter(power_fast==50) %>%
  group_by(deso, Charging_type) %>%
  summarise(total=sum(number)) %>%
  filter(Charging_type %in% c('1 (0.2)', '2 (0.2)', '3 (0.9)'))
tst.all$Charging_type <- plyr::mapvalues(tst.all$Charging_type,
                                        from=c('1 (0.2)', '2 (0.2)', '3 (0.9)'),
                                        to=c("1 Liquid-fuel","2 Plan-ahead","3 Event-actuated"))
tst.all <- right_join(x = zones[, c('deso', 'Area')]%>% st_set_geometry(NULL),
                                             y = tst.all,
                                   by = "deso", all.y = TRUE)
tst.all$Charge_point_density <- tst.all$total / (as.numeric(tst.all$Area) / 10^6)# #/km^2

tst.stats <- tst.all %>%
  group_by(Charging_type) %>%
  summarise(q.05 = quantile(Charge_point_density, 0.05, na.rm=T),
            q.95 = quantile(Charge_point_density, 0.95, na.rm=T),
            q.50 = quantile(Charge_point_density, 0.5, na.rm=T),
            q.m = mean(Charge_point_density, na.rm=T)
            )

g <- ggplot(data=tst) +
  geom_histogram(aes(y=Charge_point_density, fill = Purpose), color='white', size=0.3,
                 position = "stack", bins = 40) +
  scale_fill_manual(values = c("gray40", 'gray70')) +
  geom_hline(data=tst.stats, aes(yintercept = q.50, color='50%')) +
  geom_text(data=tst.stats, aes(230, q.50, label = round(q.50, digits=2),
                                vjust = 1, hjust = 1, color='50%'), size = 3) +
  geom_hline(data=tst.stats, aes(yintercept = q.05, color='5%')) +
  geom_text(data=tst.stats, aes(230, q.05, label = round(q.05, digits=3),
                                vjust = 1, hjust = 1, color='5%'), size = 3) +
  geom_hline(data=tst.stats, aes(yintercept = q.95, color='95%')) +
  geom_text(data=tst.stats, aes(230, q.95, label = round(q.95, digits=2),
                                vjust = 1, hjust = 1, color='95%'), size = 3) +
  geom_hline(data=tst.stats, aes(yintercept = q.m, color='Mean')) +
  geom_text(data=tst.stats, aes(230, q.m, label = round(q.m, digits=2),
                                vjust = 1, hjust = 1, color='95%'), size = 3) +
  scale_color_manual(name='Total statistics', breaks=c('5%', '50%', 'Mean', '95%'),
                     values=c('50%'='darkgreen', '5%'='steelblue', '95%'='darkblue', 'Mean'='purple')) +
  facet_grid(.~Charging_type) +
  scale_y_log10() +
  xlim(0, 240) +
  labs(x='# of statistical zones', y=bquote('Charge points density'~(per~km^2))) +
  theme_minimal()

ggsave(filename = paste0("figures/", scenario, "_inf_density.png"), plot=g,
       width = 10, height = 5, unit = "in", dpi = 300)