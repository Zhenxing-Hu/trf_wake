clc,clear all;close all;
ls = dir('../out_trf/DLS*');
addpath('./utility/');
for i = 1:length(ls)
    folder = ['../out_trf/',ls(i).name];
    fix_lw6_struct_fields(folder);
end
