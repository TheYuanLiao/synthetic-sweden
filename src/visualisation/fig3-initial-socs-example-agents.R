# Title     : SOC distributions and example agents' soc trajectories
# Objective : SOC distributions and example agents' soc trajectories
# Created by: Yuan Liao
# Created on: 2022-07-12

library(ggplot2)
library(dplyr)
library(plyr)
library(viridis)
library(ggpubr)
library(scales)
library(grid)
options(scipen=10000)

run_id <- ""
scenario <- "scenario_vg_car"

# a) Initial SOCs
df <- read.csv(paste0('results/', scenario, '_indi_raw_paraset1_5days.csv'))
df$Home_charger <- mapvalues(df$home_charger,
          from=c(0, 1),
          to=c("Without home charger","With home charger"))
df$Resi_charger <- mapvalues(df$residential_charger,
          from=c(0, 1),
          to=c("Without overnight charger","With overnight charger"))
df$Charging_type <- mapvalues(df$charging_type,
          from=c(1, 2, 3),
          to=c("1 Liquid-fuel","2 Plan-ahead","3 Event-actuated"))
g1 <- ggplot(data=df) +
  geom_histogram(aes(y=soc_init), fill='purple', color=NA, alpha=0.3, bins=100) +
  facet_grid(.~Charging_type) +
  labs(x='# of agents (log scale)', y='Initial SOC') +
  scale_x_log10() +
  theme_minimal()

# b) Example agents
ev.sim <- read.csv(paste0("results/", scenario, "_demo_paraset1_5days.csv"))
socColor <- c("#bdc9e1", "#037540")
distanceColor <- 'gray45'
plot.wo <- function(tp, ev.sim1, tlt, y1, y2){
  coeff1 <- max(ev.sim1$distance_driven)
  t <- c(tp*10, tp)
  g <- ggplot(data = ev.sim1[ev.sim1$charging_type %in% t, ], aes(x=time / 60/60, group=factor(charging_type, levels=c(t[1], t[2])))) +
  theme_minimal() +
  geom_line(aes(y=soc, size=factor(charging_type, levels=c(t[1], t[2])),
                color=factor(charging_type, levels=c(t[1], t[2]))), show.legend = F, alpha=1) +
  scale_color_manual(name='Charging type', values=socColor) +
  scale_size_manual(name='Charging type', values = c(2, 0.5)) + #c('0'=3, '1'=2, '2'=1, '3'=0.5)
  geom_line(aes(y=distance_driven / coeff1), size=0.5, show.legend = F, color=distanceColor) +
  xlim(c(0, 23)) +
  scale_y_continuous(

    # Features of the first axis
    name = y1,

    # Add a second axis and specify its features
    sec.axis = sec_axis(~.*coeff1, name=y2)
  ) +
  theme(
    axis.title.y = element_text(color = 'black', size=12),
    axis.title.y.right = element_text(color = distanceColor, size=12)
  ) +
  labs(title=tlt, x='Time (h)')
  return(g)
}

# Without home charger
ev.sim1 <- ev.sim[ev.sim$person == 7585605,]
g20 <- plot.wo(ev.sim1=ev.sim1, tp=20, tlt='w/o daytime charging',
               y1="SOC", y2=NULL)

g2 <- plot.wo(ev.sim1=ev.sim1, tp=2, tlt='w/ daytime charging',
              y1=NULL, y2="Distance driven (km)")

G2 <- ggarrange(g20, g2, ncol = 2, nrow = 1)

G <- ggarrange(G2, g1, ncol = 1, nrow = 2, labels = c('(b)', '(c)'))
ggsave(filename = paste0("figures/", scenario, "_socs_examples.png"), plot=G,
       width = 5, height = 4, unit = "in", dpi = 300)