% Example:
%   lw_merge_avg_native('..\out_trf')
clc,clear all;close all;
addpath(genpath('..\utility/letswave7-master'));
LW_init();                     
% init Letswave
root_out_trf = fullfile('..','out_trf'); 

genres = {'Blues','Metal','Pop'};        % add/remove as needed
subdirs = dir(root_out_trf);
subdirs = subdirs([subdirs.isdir] & ~ismember({subdirs.name},{'.','..'}));

%letswve options
opt_merge = struct('type','epoch','suffix','','is_save',0);
opt_avg = struct('operation','average','suffix','','is_save',0);

for s = 1:numel(subdirs)
    subj_path = fullfile(subdirs(s).folder, subdirs(s).name);
    fprintf('\nSubject: %s\n', subdirs(s).name);

    for g = 1:numel(genres)
        genre = genres{g};

        % candidate files for this genre (inst/voc)
        f_inst = fullfile(subj_path, sprintf('recon trial_wise TRF %s-inst.lw6', genre));
        lwdata = FLW_load.get_lwdata('filename',f_inst);
        lwdata = FLW_average_epochs.get_lwdata(lwdata,opt_avg);
        CLW_save(lwdata,'path',subj_path);

        f_voc  = fullfile(subj_path, sprintf('recon trial_wise TRF %s-voc.lw6',  genre));
        lwdata = FLW_load.get_lwdata('filename',f_inst);
        lwdata = FLW_average_epochs.get_lwdata(lwdata,opt_avg);
        CLW_save(lwdata,'path',subj_path);

        filelist = {};
        if exist(f_inst,'file'), filelist{end+1} = f_inst; end 
        if exist(f_voc,'file'),  filelist{end+1} = f_voc;  end 

        if isempty(filelist)
            fprintf('  %s: no inst/voc found → skip\n', genre);
            continue
        end

        % --------- FLW pipeline (native syntax) ----------
        % load
        opt = struct('filename',{filelist});
        lwdataset = FLW_load.get_lwdataset(opt);

        % merge epochs (concatenate ep dimension)
        lwdata = FLW_merge.get_lwdata(lwdataset,opt_merge);

        % average epochs
        lwdata = FLW_average_epochs.get_lwdata(lwdata,opt_avg);
        lwdata.header.name = sprintf('avg recon trial_wise %s', genre);
        CLW_save(lwdata,'path',subj_path);
    end
end

fprintf('\nDone.\n');


