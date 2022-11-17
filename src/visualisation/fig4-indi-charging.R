# Title     : Visualise the ev simulation stats at individual level
# Objective : Charging duration, time share, and energy (home charger x charging types x fast charging power)
# Created by: Yuan Liao
# Created on: 2022-05-19

library(ggplot2)
library(dplyr)
library(viridis)
library(ggpubr)
library(scales)
library(grid)
options(scipen=10000)

scenario <- 'scenario_vg_car'

# a) Charging duration
df <- read.csv(paste0('results/', scenario, '_stats_indi_5days.csv'))
df$Home_charger <- plyr::mapvalues(df$home_charger,
          from=c(0, 1),
          to=c("w/o home charger", "w/ home charger"))
# If excluding sensitivity test of SOC threshold, use the below simplification
df <- df %>%
  filter(Charging_type %in% c('1 (0.2)', '2 (0.2)', '3 (0.9)'))

df$Charging_type <- plyr::mapvalues(df$Charging_type,
          from=c('1 (0.2)', '2 (0.2)', '3 (0.9)'),
          to=c("1","2","3"))

df.inter <- df %>%
  filter(charging_time_inter > 0)

df.fast <- df %>%
  filter(charging_time_fast > 0)

cols <- c('#3c40c6', '#05c46b')

data_summary <- function(x) {
   m <- median(x)
   ymin <- unname(quantile(x, 0.25))
   ymax <- unname(quantile(x, 0.75))
   return(c(y=m,ymin=ymin,ymax=ymax))
}

g1 <- ggplot(data = df.inter, aes(x=Charging_type, y=charging_time_inter)) +
   stat_summary(fun.data=data_summary, geom="pointrange", position=position_dodge(0.3)) +
  facet_grid(.~Home_charger) +
  theme_minimal() +
  labs(title = 'Intermediate charging') +
  theme(plot.title = element_text(size=9)) +
  rremove("ylab") +
  rremove('xlab')


g2 <- ggplot(data = df.inter, aes(x=Charging_type, y=charging_time_ratio*100)) +
   stat_summary(fun.data=data_summary, geom="pointrange", position=position_dodge(0.3)) +
  facet_grid(.~Home_charger) +
  theme_minimal() +
  labs(title = 'Intermediate charging') +
  theme(plot.title = element_text(size=9)) +
  rremove("ylab") +
  rremove('xlab')

g3 <- ggplot(data = df.fast, aes(x=Charging_type, y=charging_time_fast)) + #color=as.factor(power_fast)
   stat_summary(fun.data=data_summary, geom="pointrange", position=position_dodge(0.3)) +
#  scale_colour_manual("Fast charging power", values = cols) +
  facet_grid(.~Home_charger) +
  theme_minimal() +
  labs(title = 'Fast charging') +
  theme(plot.title = element_text(size=9)) +
  rremove("ylab") +
  rremove('xlab')

g4 <- ggplot(data = df.fast, aes(x=Charging_type, y=charging_time_ratio*100)) + # color=as.factor(power_fast)
   stat_summary(fun.data=data_summary, geom="pointrange", position=position_dodge(0.3)) +
#  scale_colour_manual("Fast charging power", values = cols) +
  facet_grid(.~Home_charger) +
  theme_minimal() +
  labs(title = 'Fast charging') +
  theme(plot.title = element_text(size=9)) +
  rremove("ylab") +
  rremove('xlab')

G11 <- ggarrange(g1, g3, ncol = 1, nrow = 2, common.legend = FALSE)
G11 <- annotate_figure(G11, left = textGrob("Charging duration per agent-day (min)",
                                          rot = 90, vjust = 0.5, gp = gpar(cex = 1)),
                      bottom = textGrob("Charging strategy", gp = gpar(cex = 1)))

G12 <- ggarrange(g2, g4, ncol = 1, nrow = 2, common.legend = FALSE)
G12 <- annotate_figure(G12, left = textGrob("Share of charging time in parking (%)",
                                          rot = 90, vjust = 0.5, gp = gpar(cex = 1)),
                      bottom = textGrob("Charging strategy", gp = gpar(cex = 1)))

G1 <- ggarrange(G11, G12, ncol = 2, nrow = 1, labels = c('(a)', '(b)'))

# b) Total energy
df.total <- df %>%
  group_by(Charging_type, home_charger) %>%
  summarize(energy_daytime=(sum(charging_energy_fast) + sum(charging_energy_inter))/10^3,
            energy_overnight=sum(charging_energy_overnight)/10^3)
df.total.daytime1 <- df.total[df.total$home_charger==1, c('Charging_type', 'energy_daytime')]
df.total.daytime1 <- df.total.daytime1 %>% rename(energy_hc = energy_daytime)
df.total.daytime2 <- df.total[df.total$home_charger==0, c('Charging_type', 'energy_daytime')]
df.total.daytime2 <- df.total.daytime2 %>% rename(energy_nhc = energy_daytime)
df.total.daytime <- merge(df.total.daytime1, df.total.daytime2)
df.total.daytime$type <- rep('daytime', 3)
df.total.overnight1 <- df.total[df.total$home_charger==1, c('Charging_type', 'energy_overnight')]
df.total.overnight1 <- df.total.overnight1 %>% rename(energy_hc = energy_overnight)
df.total.overnight2 <- df.total[df.total$home_charger==0, c('Charging_type', 'energy_overnight')]
df.total.overnight2 <- df.total.overnight2 %>% rename(energy_nhc = energy_overnight)
df.total.overnight <- merge(df.total.overnight1, df.total.overnight2)
df.total.overnight$type <- rep('overnight', 3)
df.total <- rbind(df.total.daytime, df.total.overnight)
df.total <- df.total %>%
   mutate_if(is.numeric, ~ case_when(. < 10 ~ round(., 1), TRUE ~ round(., 0)))

cols <- c('#037540', '#242676')
g5 <- ggplot(data = df.total[df.total$type=='daytime',], aes(x=Charging_type)) +
  geom_segment(aes(xend=Charging_type, y=energy_nhc, yend=energy_hc), color="grey") +
  geom_point(aes(y=energy_hc, color='w/'), size=3 ) +
  geom_point(aes(y=energy_nhc, color='w/o'), size=3 ) +
  scale_color_manual(name='Home charger access', breaks=c('w/', 'w/o'),
                     values=c('w/'='#037540', 'w/o'='#242676')) +
  coord_flip() +
  scale_x_discrete(limits = c("3", "2", "1")) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=9),
    legend.position = c(0.7, 0.7),
    panel.border = element_blank(),
  ) +
  geom_text(color="black", size=2.5, vjust=-1, hjust=1,
            aes(y=energy_nhc, label=energy_nhc))+
  geom_text(aes(y=energy_hc, label=energy_hc),
            color="black", size=2.5, vjust=-1)+
  labs(title='Daytime charging') +
  xlab("Charging strategy") +
  ylab("")

g6 <- ggplot(data = df.total[df.total$type=='overnight',], aes(x=Charging_type)) +
  geom_segment(aes(xend=Charging_type, y=energy_nhc, yend=energy_hc), color="grey") +
  geom_point(aes(y=energy_hc, color='w/'), size=3 ) +
  geom_point(aes(y=energy_nhc, color='w/o'), size=3 ) +
  scale_color_manual(name='Home charger access', breaks=c('w/', 'w/o'),
                     values=c('w/'='#037540', 'w/o'='#242676')) +
  coord_flip() +
  scale_x_discrete(limits = c("3", "2", "1")) +
  theme_minimal() +
  theme(
    plot.title = element_text(size=9),
    legend.position = c(0.7, 0.7),
    panel.border = element_blank(),
  ) +
  geom_text(color="black", size=2.5, vjust=-1,
            aes(y=energy_nhc, label=energy_nhc))+
  geom_text(aes(y=energy_hc, label=energy_hc),
            color="black", size=2.5, vjust=-1, hjust=1)+
  labs(title='Overnight charging') +
  xlab("Charging strategy") +
  ylab("")

G2 <- ggarrange(g5, g6, ncol = 2, nrow = 1, common.legend = TRUE)
G2 <- annotate_figure(G2, bottom = text_grob("Total energy from chargers to BEVs (MWh)", size = 12))
G <- ggarrange(G1, G2, ncol = 2, nrow = 1, labels = c('', '(c)'), widths = c(1.2, 1))
ggsave(filename = paste0("figures/", scenario, "_indi_stats.png"), plot=G,
       width = 10, height = 4, unit = "in", dpi = 300)