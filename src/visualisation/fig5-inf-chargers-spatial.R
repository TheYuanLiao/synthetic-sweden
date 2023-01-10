# Title     : Spatial distribution of chargers
# Objective : Charging strategy x occasion x # of charger
# Created by: Yuan Liao
# Created on: 2022-07-12

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

# Charger data
zones <- read_sf('dbs/DeSO/DeSO_2018_v2.shp')
charging <- read.csv(paste0("results/", scenario, "_stats_charger_5days.csv"))
zones <- zones[zones$deso %in% unique(charging$deso),]
zones$Area <- st_area(zones)
muni <- st_read(paste0("dbs/municipalities/sweden_komk.shp"))
muni.vg <- st_crop(muni, zones)

# Consider work and other of all charging types
charging.agg <- charging %>%
  filter(power_fast==50) %>%
  group_by(deso, Charging_type, Purpose) %>%
  summarise(total=sum(number)) %>%
  filter(Charging_type %in% c('1 (0.2)', '2 (0.2)', '3 (0.9)'))
charging.agg$Charging_type <- plyr::mapvalues(charging.agg$Charging_type,
                                        from=c('1 (0.2)', '2 (0.2)', '3 (0.9)'),
                                        to=c("1 Liquid-fuel","2 Plan-ahead","3 Event-triggered"))

charging.agg.work <- right_join(x = zones[,c('deso', 'geometry')],
                                 y = charging.agg[charging.agg$Purpose == 'Work', ],
                                 by = "deso", all.y = TRUE)

charging.agg.other <- right_join(x = zones[,c('deso', 'geometry')],
                                 y = charging.agg[charging.agg$Purpose == 'Other', ],
                                 by = "deso", all.y = TRUE)
# Spatial distribution
g1 <- ggplot() +
  theme_void() +
  geom_sf(data = muni.vg, fill='gray90', color='white', size = 0.05) +
  geom_sf(data = zones, fill='gray55', color = NA, alpha=1, show.legend = FALSE) +
  geom_sf(data = charging.agg.work, aes(fill=total), color = NA, alpha=0.8, show.legend = TRUE) +
  facet_grid(.~Charging_type)+
  scale_fill_viridis(name = "# of chargers", trans = 'log10') +
  labs(title = 'Work') +
  coord_sf(datum=st_crs(3006)) +
  theme(legend.position = 'top', plot.title = element_text(hjust = 0.5))


g2 <- ggplot() +
  theme_void() +
  geom_sf(data = muni.vg, fill='gray90', color='white', size = 0.05) +
  geom_sf(data = zones, fill='gray55', color = NA, alpha=1, show.legend = FALSE) +
  geom_sf(data = charging.agg.other, aes(fill=total), color = NA, alpha=0.8, show.legend = TRUE) +
  facet_grid(.~Charging_type)+
  scale_fill_viridis(name = "# of chargers", trans = 'log10') +
  labs(title = 'Other') +
  coord_sf(datum=st_crs(3006)) +
  theme(legend.position = 'top', plot.title = element_text(hjust = 0.5))


G1 <- ggarrange(g1, g2, ncol = 2, nrow = 1, common.legend = T, legend="bottom")
ggsave(filename = paste0("figures/", scenario, "_inf_spatial.png"), plot=G1,
       width = 10, height = 4, unit = "in", dpi = 300)