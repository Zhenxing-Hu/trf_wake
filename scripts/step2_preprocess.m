clc,clear all;close all;
addpath(genpath('../utility/letswave7-master'));

LW_init();

prep_folder = '../data/preprocessed';
ls = dir(fullfile(prep_folder ,'DLS*.lw6'));

%%%%%%%%%%%    Option of preprocessing %%%%%%%%%%

% Rereference to the average of M1-M2 channel
opt_reref = struct('reference_list',{{'M1','M2'}},'apply_list',{{'F3','F4','C3','C4','P3','P4','O1','O2','M1','M2'}},'suffix','reref','is_save',0);
% Filter EEG to [0.5 Hz - 8 Hz] 
opt_butt = struct('filter_type','bandpass','high_cutoff',8,'low_cutoff',0.5,'filter_order',4,'suffix','butt','is_save',0);
% downsample EEG to 100 Hz
opt_ds = struct('x_dsratio',5,'suffix','ds','is_save',0);
% segmentation option
opt_seg = struct('event_labels',{{'S 81','S 82','S 83','S 84','S 85','S 86','S 87'}},'x_start',0,'x_end',5,'x_duration',5,'suffix','ep','is_save',0);

for f_idx = 1:length(ls)
    filename = fullfile(prep_folder,ls(f_idx).name);
    opt_load = struct('filename',filename);
    lwdata= FLW_load.get_lwdata(opt_load);
    
    % Apply rereference
    lwdata= FLW_rereference.get_lwdata(lwdata,opt_reref);
    % Apply band-pass filtering
    lwdata= FLW_butterworth_filter.get_lwdata(lwdata,opt_butt);
    % Apply down-samplng
    lwdata = FLW_downsample.get_lwdata(lwdata,opt_ds);
    CLW_save(lwdata,'path',prep_folder);
    lwdata = FLW_segmentation.get_lwdata(lwdata,opt_seg);
    CLW_save(lwdata,'path',prep_folder);

end
