# Title     : Spatiotemporal distribution of chargers
# Objective : Charging strategy x occasion x # of charger, and power x charging strategy
# Created by: Yuan Liao
# Created on: 2022-07-12

library(dplyr)
library(sp)
library(spdep)
library(ggplot2)
library(ggpubr)
library(ggmap)
library(ggspatial)
library(ggsn)
library(ggnewscale)
library(scico)
library(viridis)
library(animation)
library(units)
library(scales)
options(scipen=10000)

scenario <- "scenario_vg_car"
fake_scico <- scico(3, palette = "vik")
# Charger data
zones <- read_sf('dbs/DeSO/DeSO_2018_v2.shp')
charging <- read.csv(paste0("results/", scenario, "_inf_comp.csv"))
charging$diff <- charging$gt - charging$sim
charging <- charging %>%
  arrange(Charging_type, diff) %>%
  mutate(mycolor = ifelse(diff>0, "type1", "type2"))
charging$Charging_type <- plyr::mapvalues(charging$Charging_type,
                                        from=c('1 (0.2)', '2 (0.2)', '3 (0.9)'),
                                        to=c("1 Liquid-fuel","2 Plan-ahead","3 Event-triggered"))
zones <- zones[zones$deso %in% unique(charging$deso),]
muni <- st_read(paste0("dbs/municipalities/sweden_komk.shp"))
muni.vg <- st_crop(muni, zones)
charging.agg <- right_join(x = zones[,c('deso', 'geometry')],
                           y = charging,
                           by = "deso", all.y = TRUE)

# a) Scatter plot of # of charging points: simulated vs. ground-truth data
g1 <- ggplot(charging, aes(x=deso, y=diff)) +
  theme_minimal() +
  geom_segment( aes(x=deso, xend=deso, y=0, yend=diff, color=diff), size=0.4) +
  scale_color_gradient2(name='Disparity in charging points',
                       low = fake_scico[3], mid=fake_scico[2], high = fake_scico[1]) +
  facet_grid(.~Charging_type) +
  theme(legend.position = "none",
        panel.grid.major.x = element_blank(),
#        panel.grid.major.y = element_blank(),
        panel.border = element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
#        axis.text.y=element_blank(),
#        axis.ticks.y=element_blank(),
        strip.text.x = element_blank(),
        plot.margin = margin(0, 0, 0, 0, "cm")
      ) +
  xlab("DeSO zone") +
  ylab("Disparity in charging points")

g2 <- ggplot() +
  theme_void() +
  geom_sf(data = muni.vg, fill='gray25', color='white', size = 0.05) +
  geom_sf(data = zones, fill='gray55', color = NA, alpha=1, show.legend = FALSE) +
  geom_sf(data = charging.agg,
          aes(fill=diff), color = NA, alpha=1, show.legend = TRUE) +
  scale_fill_gradient2(name='Disparity in charging points',
                       low = fake_scico[3], mid=fake_scico[2], high = fake_scico[1]) +
  coord_sf(datum=st_crs(3006)) +
  facet_grid(.~Charging_type)+
  theme(legend.position = 'top',
        plot.margin = margin(-1, 0, -1, 0, "cm"))
G <- ggarrange(g2, g1, ncol = 1, nrow = 2, heights = c(2, 0.7))
ggsave(filename = paste0("figures/", scenario, "_inf_comp.png"), plot=G,
       width = 10, height = 8, unit = "in", dpi = 300)