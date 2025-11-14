% Segmentation of "ds butt <SUBJ>.lw6" by S81..S87, save with pretty names
% Zhenxing, this writes into: ../out_trf/<SUBJECT>/
% Requires Letswave7 on path.

clc; clear;

% ---- 0) Paths ----
rawRoot = fullfile('..','data','preprocessed');        % where your .lw6 live
outRoot = fullfile('..','out_trf');           % subject-specific output folders
addpath(genpath('..\utility/letswave7-master'));
LW_init();

% ---- 1) Subjects and input files (only the "ds butt ..." variants) ----
subjects = {'DLS32_E','DLS46_E','DLS50_E','DLS51_E','DLS52_E','DLS54_E'};   % extend if you add more
ls = dir(fullfile(outRoot,'DLS*'));

% ---- 2) Event → pretty name map (exactly as in your screenshot) ----
genre_names = containers.Map( ...
    {'S 81','S 82','S 83','S 84','S 85','S 86','S 87'}, ...
    {'Pop-inst','Pop-voc','Blues-inst','Blues-voc','Metal-inst','Metal-voc','Control'} );

% ---- 3) Segmentation window (seconds) ----
x_start    = 0.13;
x_end      = 4.33;
x_duration = 4.20;    % redundant with end-start, but matches your past settings

% ---- 4) Loop subjects ----
for s = 1:numel(ls)
    subj = ls(s).name;
    sub_fold = [outRoot,'/',ls(s).name];
    inFile = fullfile(rawRoot, sprintf('ds butt %s.lw6', subj));
    if ~exist(inFile,'file')
        warning('Missing input: %s', inFile);  %#ok<*WNTAG>
        continue;
    end

    % ensure subject out dir exists
    outDir = fullfile(outRoot, subj);
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    % Load the "ds butt" dataset
    opt = struct('filename', inFile);
    lwdata = FLW_load.get_lwdata(opt);
    option=struct('type','channel','items',{{'F3','F4','C3','C4','P3','P4','O1','O2'}},'suffix','sel_chan','is_save',0);
    lwdata = FLW_selection.get_lwdata(lwdata,option);
    % Overwrite output location so Letswave saves there
    if ~isfield(lwdata,'header'), lwdata.header = struct(); end
    lwdata.header.filepath = outDir;   % <- where FLW_segmentation will save
    
    fprintf('\nSubject %s\n', subj);

    % ---- 5) Segment for each code S 81 ... S 87 and save with pretty name ----
    evCodes = keys(genre_names);   % {'S 81', ... 'S 87'}
    for k = 1:numel(evCodes)
        code = evCodes{k};
        pretty = genre_names(code);

        % This suffix becomes part of the saved filename:
        % final filename ≈ <original>_<suffix>.lw6 under outDir

        % Run segmentation for this single event code
        segOpt = struct( ...
            'event_labels', {{code}}, ...
            'x_start',      x_start, ...
            'x_end',        x_end, ...
            'x_duration',   x_duration, ...
            'suffix',       '', ...
            'is_save',      0 ...
        );

        % Execute segmentation; files are written to lwdata.header.filepath

        lwdata_ERP = FLW_segmentation.get_lwdata(lwdata, segOpt);
        lwdata_ERP.header.name = [pretty ' merge ERP'];
        CLW_save(lwdata_ERP,'path',sub_fold)
        fprintf('  ✓ %s → %s.lw6\n', code, lwdata_ERP.header.name);
    end
end

fprintf('\nDone. Segmented files saved in %s/<SUBJECT>/\n', outRoot);
