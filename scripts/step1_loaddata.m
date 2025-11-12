clc,clear all;close all;
addpath(genpath('../utility/letswave7-master'));

LW_init();
data_folder = '../data/raw';
ls = dir(fullfile(data_folder,'*.eeg'));
save_folder = '../data/preprocessed';

for f_idx = 1:length(ls)
    filename = ls(f_idx).name;
    lwdata = FLW_import_data.get_lwdata('filename',filename,'pathname',data_folder,'is_save',0);
   if ~exist('../data/preprocessed', 'dir')
        mkdir('../data/preprocessed');
    end
    CLW_save(lwdata,'path',save_folder);
end



