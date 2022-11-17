# Title     : Visualise the input of MATSim simulation
# Objective : a) Car users in DeSO zones, b) Road network for the entire Sweden, and c) VG
# Created by: Yuan Liao
# Created on: 2022-07-10

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
library(units)
options(scipen=10000)

scenario <- "scenario_vg_car"
zones <- st_read(paste0("dbs/DeSO/DeSO_2018_v2.shp"))
muni <- st_read(paste0("dbs/municipalities/sweden_komk.shp"))
agents <- read.csv(paste0("dbs/output_summary/", scenario, "/valid_agents.csv"))
agents.agg <- agents %>%
  group_by(Deso) %>%
  count()
zones.agents <- right_join(x = zones, y = agents.agg, by = c("deso"="Deso"), all.y = TRUE)
roads <- st_read(paste0("dbs/output_summary/", scenario, "/volumes_slope.shp"))
roads.vg <- st_crop(roads, zones.agents)
muni.vg <- st_crop(muni, zones.agents)

g1 <- ggplot(data = zones.agents) +
  geom_sf(aes(fill = n), color=NA, show.legend = T) +
  scale_fill_viridis() +
  north(zones.agents, location = "topright") +
  annotation_scale() +
  labs(fill='# of car users') +
  theme_void() +
  theme(legend.position = 'top', plot.margin = margin(0,1,0,0, "cm"))

g2 <- ggplot() +
  geom_sf(data = muni, fill='gray90', color='white', size = 0.05) +
  geom_sf(data = roads, color='steelblue', size = 0.05) +
  north(roads, location = "topleft") +
  annotation_scale() +
  theme_void()

g3 <- ggplot() +
  geom_sf(data = muni.vg, fill='gray90', color='white', size = 0.05) +
  geom_sf(data = roads.vg, color='steelblue', size = 0.05) +
  north(roads.vg, location = "topright") +
  annotation_scale() +
  theme_void()
G11 <- ggarrange(g1, g3, ncol = 1, nrow = 2, labels=c('(a)', '(b)'))
G1 <- ggarrange(G11, g2, ncol = 2, nrow = 1, labels=c('', '(c)'))

# a) Charging dynamics
df <- read.csv('dbs/ev/charging_dynamics.csv')
df <- df %>%
  filter(power != 150)
labels <- c(`40` = 'B-40 kWh',
            `60` = 'C-60 kWh',
            `100` = 'D-100 kWh')

g4 <- ggplot(data=df) +
  geom_line(aes(x=soc_end, y=time/60, linetype=as.factor(power))) +
  theme_minimal() +
  scale_linetype_discrete(name='Power (kW)') +
  facet_grid(.~as.factor(battery_size),
             labeller = as_labeller(labels)) +
  scale_x_continuous(breaks = c(0, 0.5, 1)) +
  labs(x='SOC', y='Time (min)') + #title = 'Charging time for different battery sizes and powers'
  theme(legend.position = 'top')

# b) Fleet composition
df.fleet <- read.csv(paste0('dbs/output_summary/', scenario, '/valid_agents_car_fleet.csv'))
df.fleet <- stack(df.fleet, c('B', 'C', 'D'))
names(df.fleet) <- c('dbs/agents', 'car_seg')
df.fleet$income <- rep(c("None", "<180K", "180K - 300K", "300K - 420K", ">420K"), 3)
df.fleet$agents <- df.fleet$agents / sum(df.fleet$agents) * 100
df.fleet$car_seg <- factor(df.fleet$car_seg, levels = c("B", "C", "D", "Total"))

df.fleet[nrow(df.fleet) + 1,] <- c(12, "B", 'Total')
df.fleet[nrow(df.fleet) + 1,] <- c(50, "C", 'Total')
df.fleet[nrow(df.fleet) + 1,] <- c(38, "D", 'Total')
df.fleet[nrow(df.fleet) + 1,] <- c(3.45, "Total", "None")
df.fleet[nrow(df.fleet) + 1,] <- c(15.02, "Total", "<180K")
df.fleet[nrow(df.fleet) + 1,] <- c(20.50, "Total", "180K - 300K")
df.fleet[nrow(df.fleet) + 1,] <- c(30.29, "Total", "300K - 420K")
df.fleet[nrow(df.fleet) + 1,] <- c(30.74, "Total", ">420K")
df.fleet[nrow(df.fleet) + 1,] <- c(100, "Total", "Total")
df.fleet$agents <- as.numeric(df.fleet$agents)
df.fleet$income <- factor(df.fleet$income,
                             levels=c("None", "<180K", "180K - 300K", "300K - 420K", ">420K", "Total"))
g5 <- ggplot(df.fleet, aes(x=income, y=as.factor(car_seg))) +
  geom_tile(colour = "gray", fill = "white") +
  geom_text(aes(label = signif(agents, 2)), show.legend = T) +
#  scale_color_gradient(name='Share of agents (%)', low='black', high = 'blue', limits = c(0, 16)) +
  theme_minimal() +
  theme(legend.position = "bottom", legend.key.width = unit(0.5, "cm"),
        panel.grid = element_blank(),
        axis.text.x = element_text(angle = 20)) +
  scale_y_discrete(limits = rev(levels(df.fleet$car_seg))) +
  theme(plot.margin = margin(1,0,0,0, "cm")) + # axis.text.x = element_text(angle = 30, vjust=0.7),
  labs(x='Income level', y='BEV segment')

G2 <- ggarrange(g4, g5, ncol = 1, nrow = 2, labels = c('(d)', '(e)'))
G <- ggarrange(G1, G2, ncol = 2, nrow = 1, widths = c(1, 0.8))
ggsave(filename = paste0("figures/", scenario, "_inputs.png"), plot=G,
       width = 10, height = 7, unit = "in", dpi = 300)