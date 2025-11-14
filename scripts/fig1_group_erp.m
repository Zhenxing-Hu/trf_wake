
% Grand-average ERP (mean ± 95% CI) across all subjects, per genre.
% Looks for:
%   ../out_trf/DLS##_E/avg <Genre> ERP.lw6 + .mat
% Genres: Blues, Metal, Pop

clc; close all;
addpath(genpath("../utility/letswave7-master"));
out_trf_root = fullfile('..','out_trf');
genres       = {'Blues','Metal','Pop'};
fig_root     = fullfile('..','figures'); if ~exist(fig_root,'dir'), mkdir(fig_root); end

% Optional: crop to first tmax samples (set [] to keep full length)
tmax = 300;   % e.g., 300

% list subject folders
subj_dirs = dir(fullfile(out_trf_root,'DLS*_E'));
subj_dirs = subj_dirs([subj_dirs.isdir]);

% style
colMean = [0.00 0.45 0.74];
colFill = [0.70 0.82 0.94];
chLab = ["F3","F4","C3","C4","P3","P4","O1","O2"];
for g = 1:numel(genres)
    G = genres{g};

    % ---- gather ERPs for this genre ----
    X_all = [];                 % [nSub x nCh x nT]
    t_ms  = [];
    
    for s = 1:numel(subj_dirs)
        subj_id = subj_dirs(s).name;                          % 'DLS## _E'
        base    = fullfile(out_trf_root, subj_id, sprintf('avg %s ERP', G));
        [H, Y, ok] = lw_load_erp(base);                       % Y -> [nCh x nT]
        if ~ok, continue; end

        if isempty(t_ms)
            nT   = size(Y,2);
            t_ms = (H.xstart + (0:nT-1).*H.xstep) * 1000;     % ms
        end
        X_all(end+1,:,:) = Y; 
    end

    if isempty(X_all)
        warning('No ERP found for %s. Skipping.', G);
        continue
    end

    % crop if requested
    if ~isempty(tmax)
        tmax = min(tmax, size(X_all,3));
        X_all = X_all(:,:,1:tmax);
        t_ms  = t_ms(1:tmax);
    end

    nSub = size(X_all,1);
    nCh  = size(X_all,2);

    % ---- plot ----
    f = figure('Color','w','Units','pixels','Position',[100 100 900 700]);
    sgtitle(sprintf('%s — Group mean ERP (all subjects, N=%d)', G, nSub), ...
        'FontSize',18,'FontWeight','bold');

    panel = 0;
    for ch = 1:nCh
        panel = panel + 1;
        ax = subplot(3,3,panel); hold(ax,'on');

        [mu, ci] = mean_ci(squeeze(X_all(:,ch,:)));   % [1 x nT] each

        patch_ci(ax, t_ms, mu, ci, colFill);
        plot(ax, t_ms, mu, 'Color', colMean, 'LineWidth', 1.8);

        title(ax, chLab(ch), 'FontWeight','bold');
        xlabel(ax,'Time (ms)'); ylabel(ax,'Amplitude');
        set(ax,'TickDir','out'); box(ax,'off');
        xlim(ax,[t_ms(1) t_ms(end)]);
    end

    % legend panel (bottom-right)
    axL = subplot(3,3,9); hold(axL,'on');
    l_m = plot(axL, nan, nan, 'Color', colMean, 'LineWidth', 1.8);
    p_c = patch(axL, [0 1 1 0],[0 0 1 1], colFill, ...
        'Visible','off','FaceAlpha',0.35,'EdgeColor','none');
    legend(axL, [l_m, p_c], {'Group mean','95% CI (across subjects)'}, ...
        'Location','northwest','FontSize',12,'Box','off');
    axis(axL,'off');

    out_png = fullfile(fig_root, sprintf('%s_group_ERP_all.png', G));
    print(f, out_png, '-dpng', '-r200');
    fprintf('Saved: %s\n', out_png);
    close(f);
end

%% ---------- helpers ----------
function [H, Y, ok] = lw_load_erp(base_no_ext)
% Load Letswave ERP and orient to [nCh x nT]
ok = false; H = []; Y = [];
try
    S = load([base_no_ext,'.lw6'],'-mat');   % header struct in S.header
    M = load([base_no_ext,'.mat'],'-mat');   % data in M.data
    if ~isfield(S,'header') || ~isfield(M,'data'), return; end
    H = S.header;
    D = M.data;                              % [nEp x nCh x 1 x 1 x 1 x nT]
    Y = squeeze(D(:, :, 1, 1, 1, :));        % [nEp x nCh x nT]
    if ndims(Y)==3, Y = squeeze(mean(Y,1)); end  % -> [nCh x nT]
    ok = true;
catch
    ok = false;
end
end



function [mu, ci] = mean_ci(X)
    % X: [nSub × nT]
    X  = double(X);
    mu = nanmean(X,1);
    sd = nanstd (X,0,1);
    N  = sum(~isnan(X),1);
    se = sd ./ max(sqrt(N),1);
    ci = tinv(0.975, max(N-1,1)) .* se;   % 95% CI
end

function patch_ci(ax, t, mu, ci, faceCol)
    xx = [t, fliplr(t)];
    yy = [mu-ci, fliplr(mu+ci)];
    patch(ax, 'XData',xx, 'YData',yy, 'FaceColor',faceCol, ...
               'FaceAlpha',0.5, 'EdgeColor','none');
end
