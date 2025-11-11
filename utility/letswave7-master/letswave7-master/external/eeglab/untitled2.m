LW_init();option=[];
%% option for figure
option.inputfiles{1}='G:\Utilisateurs\zhenxing.hu\Project\Nap\data\Nap\ds nap1.lw6';
option.fig2_pos=[293,100,700,650];

%% option.axis{1}: Topograph1
option.ax{1}.name='Topograph1';
option.ax{1}.pos=[92,72.5,542.5,529.75];
option.ax{1}.style='Topograph';

%% option.axis{1}.content{1}: topo1
option.ax{1}.content{1}.name='topo1';
option.ax{1}.content{1}.type='topo';
option.ax{1}.content{1}.x=[0,0];
option.ax{1}.content{1}.dim='2D';
GLW_figure(option);
