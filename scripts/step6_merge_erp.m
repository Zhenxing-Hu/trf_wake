clc,clear all;close all;
LW_init;                                 % init Letswave
root_out_trf = fullfile('..','out_trf'); 

genres = {'Blues','Metal','Pop'};        % add/remove as needed
subdirs = dir([root_out_trf,'/DLS*_E']);
% subdirs = subdirs([subdirs.isdir] & ~ismember({subdirs.name},{'.','..'}));

for s = 1:numel(subdirs)
    subj_path = fullfile(subdirs(s).folder, subdirs(s).name);
    fprintf('\nSubject: %s\n', subdirs(s).name);

    for g = 1:numel(genres)
        genre = genres{g};

        % candidate files for this genre (inst/voc)
        f_inst = fullfile(subj_path, sprintf('%s-inst merge ERP.lw6', genre));
        lwdata = FLW_load.get_lwdata('filename',f_inst);
        opt = struct('type','epoch','suffix','','is_save',0);
        lwdata = FLW_average_epochs.get_lwdata(lwdata,opt);
        lwdata.header.name = ['avg ',genre,'-inst',' ERP'];
        CLW_save(lwdata,'path',subj_path);

        f_voc  = fullfile(subj_path, sprintf('%s-voc merge ERP.lw6',  genre));
        lwdata = FLW_load.get_lwdata('filename',f_voc);
        opt = struct('type','epoch','suffix','','is_save',0);
        lwdata.header.name = ['avg ',genre,'-voc',' ERP'];

        lwdata = FLW_average_epochs.get_lwdata(lwdata,opt);
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
        opt = struct('type','epoch','suffix','','is_save',0);
        lwdata = FLW_merge.get_lwdata(lwdataset,opt);
        lwdata.header.name = sprintf('merge %s ERP', genre);

        CLW_save(lwdata,'path',subj_path);

        % average epochs
        opt = struct('operation','average','suffix','','is_save',0);
        lwdata = FLW_average_epochs.get_lwdata(lwdata,opt);
        % -------------------------------------------------

        % rename dataset and save

        lwdata.header.name = sprintf('avg %s ERP', genre);
     
        % Save with Letswave helper (writes .lw6 + .mat)
        CLW_save(lwdata,'path',subj_path);
        fprintf('  %s: saved → %s.[lw6|mat]\n', genre);
    end
end

fprintf('\nDone.\n');