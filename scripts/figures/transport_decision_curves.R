args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
suppressPackageStartupMessages({library(ggplot2);library(patchwork)})
d <- read.csv(args[1], check.names=FALSE)
labels <- c(glycemic='Glucose + HbA1c',full='Seven-measure panel',full_hba1c='Panel + HbA1c',best_default='Best default')
cols <- c(glycemic='#0072B2',full='#D55E00',full_hba1c='#56B4E9',best_default='#666666')
types <- c(glycemic='solid',full='solid',full_hba1c='longdash',best_default='dotted')
base <- subset(d,model=='assess_all');base$model <- 'best_default';base$net_benefit_per100 <- pmax(0,base$net_benefit_per100)
z <- rbind(subset(d,model %in% c('glycemic','full','full_hba1c')),base)
z$model <- factor(z$model,levels=names(labels))
lims <- range(z$net_benefit_per100);lims <- lims+c(-.3,.3)
make <- function(selectedDirection,selectedStage,title){
 a <- z[z$direction==selectedDirection & z$stage==selectedStage,]
 ggplot(a,aes(x=threshold*100,y=net_benefit_per100,colour=model,linetype=model))+
  geom_hline(yintercept=0,colour='grey85',linewidth=.25)+
  geom_vline(xintercept=10,colour='grey65',linewidth=.3,linetype='dashed')+
  geom_line(linewidth=.65)+
  geom_point(data=subset(a,threshold==.1 & model!='best_default'),size=1.3)+
  scale_colour_manual(values=cols,breaks=names(labels),labels=labels,drop=FALSE)+
  scale_linetype_manual(values=types,breaks=names(labels),labels=labels,drop=FALSE)+
  scale_x_continuous(breaks=c(5,10,15,20),limits=c(5,20))+
  coord_cartesian(ylim=lims)+
  labs(title=title,x='Illustrative threshold (%)',y='Net benefit per 100 participants',colour=NULL,linetype=NULL)+
  theme_classic(base_size=8,base_family='Arial')+
  theme(plot.title=element_text(size=8.3,face='bold'),axis.line=element_line(linewidth=.35),axis.ticks=element_line(linewidth=.3),legend.text=element_text(size=7.5),legend.key.width=grid::unit(13,'mm'),plot.margin=margin(4,6,4,4))
}
p <- (make('CHARLS to ELSA','unchanged','a  CHARLS to ELSA: frozen')+make('CHARLS to ELSA','intercept_slope','b  CHARLS to ELSA: recalibrated')) /
     (make('ELSA to CHARLS','unchanged','c  ELSA to CHARLS: frozen')+make('ELSA to CHARLS','intercept_slope','d  ELSA to CHARLS: recalibrated'))
p <- p+plot_layout(guides='collect') & theme(legend.position='bottom')
ggsave(paste0(args[2],'.png'),p,width=183,height=145,units='mm',dpi=600,device=ragg::agg_png,bg='white')
ggsave(paste0(args[2],'.svg'),p,width=183,height=145,units='mm',device=svglite::svglite,bg='white')
cat('Exported four paired decision-curve panels\n')
