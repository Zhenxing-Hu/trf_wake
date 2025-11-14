% batch_erp_recon.m
% Loop all subjects in ../out_trf, run ERP reconstruction plotting
% for genres {'Blues','Pop','Metal'}, and save figures to ../figures/<subject>/.

clc; clear; close all;
addpath("./utility/");
ROOT_OUT = fullfile('..','out_trf');
ROOT_FIG = fullfile('..','figures');

GENRES = {'Blues','Pop','Metal'};

% --- list subject folders ---
d = dir(ROOT_OUT);
isSubj = [d.isdir] & ~ismember({d.name},{'.','..'});
subjNames = {d(isSubj).name};

fprintf('Found %d subjects in %s\n', numel(subjNames), ROOT_OUT);
fig_folder = ['../figures/'];
for k = 1:numel(subjNames)
    subj     = subjNames{k};
    subj_out = fullfile(ROOT_OUT, subj);     % ../out_trf/<subject>

    for g = 1:numel(GENRES)
        G = GENRES{g};

        % Build bases exactly like your single-subject script
        erp_base = fullfile(subj_out, sprintf('avg %s ERP', G));
        tw_base  = fullfile(subj_out, sprintf('avg recon trial_wise %s', G));

        try
            plot_erp_vs_recon(erp_base, tw_base, subj_out,subj,G);

            outFile = fullfile(fig_folder, sprintf('%s Recon %s.tiff', subj,G));
            saveas(gcf, outFile);
            close(gcf);

            fprintf('Saved %s | %s\n', subj, outFile);
        catch ME
            warning('Skipping %s | %s due to error:\n%s', subj, G, ME.message);
            close all force;
        end
    end
end

fprintf('Done.\n');
