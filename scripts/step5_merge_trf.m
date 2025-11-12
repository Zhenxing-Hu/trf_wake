%% Trialwise merging/averaging with explicit CLW_save to subject folders
clear; clc;
addpath(genpath('../utilityletswave7-master'));
LW_init();
rootDir     = fullfile('..','out_trf');
subjects    = dir(fullfile(rootDir,'*E'));     % e.g., DLS32_E, DLS46_E, DLS50_E
mainGenres  = {'Pop','Blues','Metal'};
subKinds    = {'inst','voc'};

for s = 1:numel(subjects)
    if ~subjects(s).isdir, continue; end
    subjDir  = fullfile(rootDir, subjects(s).name);
    fprintf('\n=== %s ===\n', subjects(s).name);

    makeFile = @(mg,sk) fullfile(subjDir, sprintf('trial_wise TRF %s-%s.lw6', mg, sk));

    %% ---------- OPTION 1: average across Pop/Blues/Metal (inst+voc pooled within each) ----------
    perMainAvgFiles = {};
    for mg = 1:numel(mainGenres)
        mgName = mainGenres{mg};
        f_inst = makeFile(mgName,'inst');
        f_voc  = makeFile(mgName,'voc');

        have = {};
        if exist(f_inst,'file'), have{end+1} = f_inst; end 
        if exist(f_voc,'file'), have{end+1} = f_voc;  end 
        if isempty(have), fprintf('Opt1: no %s files.\n', mgName); continue; end

        try
            % merge inst+voc for this main genre (if both exist)
            if numel(have) >= 2
                option = struct('filename',{have});
                lwdataset = FLW_load.get_lwdataset(option);
                option = struct('type','epoch','suffix','','is_save',0);
                lwdata = FLW_merge.get_lwdata(lwdataset, option);
                lwdata.header.name = ['merge trial_wise TRF ',mainGenres{mg}];
                CLW_save(lwdata,'path',subjDir);
            else
                option = struct('filename',have{1});
                lwdata = FLW_load.get_lwdata(option);
            end
            % average epochs for this main genre
            option = struct('operation','average','suffix','avg','is_save',0);
            lwdata = FLW_average_epochs.get_lwdata(lwdata, option);
            CLW_save(lwdata,'path',subjDir);
            % perMainAvgFiles{end+1} = fullfile(subjDir, lwdata.header.name); 
            fprintf('Opt1: saved %s\n', lwdata.header.name);
        catch ME
            warning('Opt1 per-genre failed (%s): %s', mgName, ME.message);
        end
    end

    %% ---------- OPTION 2: merge across genres by sub-kind (inst / voc), then average ----------
    for sk = 1:numel(subKinds)
        files_sk = {};
        for mg = 1:numel(mainGenres)
            f = makeFile(mainGenres{mg}, subKinds{sk});
            if exist(f,'file')
                files_sk{end+1} = f; 
            end 
        end
        if numel(files_sk) >= 2
            try
                option = struct('filename',{files_sk});
                lwdataset = FLW_load.get_lwdataset(option);
                option = struct('type','epoch','suffix','','is_save',0);
                lwdata = FLW_merge.get_lwdata(lwdataset, option);
                lwdata.header.name = ['merge trial_wise TRF ', subKinds{sk}];
                CLW_save(lwdata,'path',subjDir);
                option = struct('operation','average','suffix','avg','is_save',0);
                lwdata = FLW_average_epochs.get_lwdata(lwdata, option);
                CLW_save(lwdata,'path',subjDir);
                fprintf('Opt2: saved %s\n', lwdata.header.name);
            catch ME
                warning('Opt2 (%s) failed: %s', subKinds{sk}, ME.message);
            end
        else
            fprintf('Opt2: %s needs ≥2 files, got %d.\n', subKinds{sk}, numel(files_sk));
        end
    end

    %% ---------- OPTION 3: average each sub-genre separately ----------
    for mg = 1:numel(mainGenres)
        for sk = 1:numel(subKinds)
            f = makeFile(mainGenres{mg}, subKinds{sk});
            if ~exist(f,'file')
                fprintf('Opt3: missing %s-%s.\n', mainGenres{mg}, subKinds{sk});
                continue;
            end
            try
                option = struct('filename',f);
                lwdata = FLW_load.get_lwdata(option);
                option = struct('operation','average','suffix','avg','is_save',0);
                lwdata = FLW_average_epochs.get_lwdata(lwdata, option);

                % lwdata.header.name = sprintf('avg_%s-%s.lw6', mainGenres{mg}, subKinds{sk});
                CLW_save(lwdata,'path',subjDir);
                fprintf('Opt3: saved %s\n', lwdata.header.name);
            catch ME
                warning('Opt3 (%s-%s) failed: %s', mainGenres{mg}, subKinds{sk}, ME.message);
            end
        end
    end
end
