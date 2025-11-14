
% Group TRF (trial-wise, averaged per trial) with 95% CI
% Two groups: Metal Hater vs Non-Hater (from Excel)
%
% Expects folder structure:
%   ../out_trf/DLS##_E/  with datasets named:
%     'avg merge trial_wise TRF <Genre>.lw6' + '.mat'
%
% Sheet columns: Subject | Group | Metal | Pop
% (Group must contain strings 'Hater' / 'NonHater')

clc; close all;

out_trf_root = fullfile('..','out_trf');
xls_path     = fullfile('..','Subjects_Distribution.xlsx');% xlsx with: Subject, Metal, Pop
genres       = {'Blues','Metal','Pop'};
fig_root     = fullfile('..','figures'); if ~exist(fig_root,'dir'), mkdir(fig_root); end

% ---- read subject → group map ----
T = readtable(xls_path);
% Subject IDs are integers matched to DLS## in folder
if any(strcmpi(T.Properties.VariableNames,'Group'))
    % If a 'Group' column exists, use it (strings 'Hater'/'NonHater')
    grp_from_table = true;
    group_label = string(T.Group);
else
    grp_from_table = false;
    group_label = strings(height(T),1);
end

subj_id_num = T.Subject;           % numeric like 14, 25, ...
metalScore  = T.Metal;

% Build subject → group map
Smap = containers.Map('KeyType','char','ValueType','char');
for i=1:height(T)
    sid = sprintf('DLS%d', subj_id_num(i)); % folder prefix (no _E yet)
    if grp_from_table
        g = upper(strtrim(group_label(i)));
        if startsWith(g,"HATER","IgnoreCase",true)
            Smap(sid) = 'Hater';
        else
            Smap(sid) = 'NonHater';
        end
    else
        % Threshold per your note: <3 => Hater, >=3 => NonHater
        if metalScore(i) < 3
            Smap(sid) = 'Hater';
        else
            Smap(sid) = 'NonHater';
        end
    end
end

% ---------- collect subject folders actually present ----------
subj_dirs = dir(fullfile(out_trf_root,'DLS*_E'));
subj_dirs = subj_dirs([subj_dirs.isdir]);

% palette
colNon = [0.00 0.45 0.74];  % blue-ish
colHat = [0.85 0.33 0.10];  % orange-ish
ciNon  = [0.70 0.82 0.94];
ciHat  = [0.97 0.80 0.68];

for g = 1:numel(genres)
    G = genres{g};

    % ---------- gather all subjects for this genre ----------
    X_non = []; X_hat = [];
    t_ms  = [];
    chLab = strings(0);

    for s = 1:numel(subj_dirs)
        subj_id   = subj_dirs(s).name;     % e.g., 'DLS32_E'
        baseKey   = erase(subj_id,'_E');   % 'DLS32'
        if ~isKey(Smap, baseKey), continue; end
        grp = Smap(baseKey);

        base = fullfile(out_trf_root, subj_id, sprintf('avg merge trial_wise TRF %s', G));
        [H, D, ok] = lw_load_trf(base);
        if ~ok, continue; end
        X = squeeze(D);
        % ensure [nCh x nLags



        % time & ch labels once
        if isempty(t_ms)
            nL = size(X,2);
            t_ms = (H.xstart + (0:nL-1).*H.xstep) * 1000;
            chLab = get_chan_labels(H, size(X,1));
        end

        if strcmpi(grp,'Hater')
            X_hat(end+1,:,:) = X;
        else
            X_non(end+1,:,:) = X; 
        end
    end

    if isempty(t_ms)
        warning('No data found for genre %s. Skipping.', G);
        continue
    end

    nCh  = max([size(X_non,2), size(X_hat,2)]);
    if nCh==0, warning('No channels for %s.',G); continue; end

    % ---------- plot ----------
    f = figure('Color','w','Units','pixels','Position',[100 100 900 700]);
    sgtitle(sprintf('%s — Group mean trial\\_wise TRF (Hater vs Non-Hater)', G), ...
            'FontSize',18,'FontWeight','bold');
    
    X_hat = X_hat(:,:,3:end-3);
    X_non = X_non(:,:,3:end-3);
    t_ms = t_ms(3:end-3);
    panel = 0;
    for ch = 1:nCh
        panel = panel + 1;
        ax = subplot(3,3,panel); hold(ax,'on');

        [muN, ciN] = mean_ci(squeeze(X_non(:,ch,:))); % Non-Hater
        [muH, ciH] = mean_ci(squeeze(X_hat(:,ch,:))); % Hater

        patch_ci(ax, t_ms, muN, ciN, ciNon);
        patch_ci(ax, t_ms, muH, ciH, ciHat);

        plot(ax, t_ms, muN, 'Color', colNon, 'LineWidth', 1.8);
        plot(ax, t_ms, muH, 'Color', colHat, 'LineWidth', 1.8);
        set(gca,'xlim',[t_ms(1),t_ms(end)]);
        title(ax,chLab{ch}, 'FontWeight','bold');
        xlabel(ax,'Lag (ms)'); ylabel(ax,'w'); set(ax,'TickDir','out'); box(ax,'off');
        xlim(ax,[t_ms(1) t_ms(end)]);
    end

    % legend (bottom-right)
    axL = subplot(3,3,9); hold(axL,'on');
    l_nonhat = plot(axL, nan, nan, 'Color', colNon, 'LineWidth', 1.8);
    l_hat = plot(axL, nan, nan, 'Color', colHat, 'LineWidth', 1.8);
    p_nonhat = patch(axL, [0 1 1 0],[0 0 1 1], ciNon,'Visible','off', 'FaceAlpha',0.25, 'EdgeColor','none');
    p_hat = patch(axL, [0 1 1 0],[0 0 1 1], ciHat,'Visible','off', 'FaceAlpha',0.25, 'EdgeColor','none');
    legend(axL, [l_nonhat,p_nonhat,l_hat,p_hat], ...
        {'Non-Hater mean','Non-Hater 95% CI','Hater mean','Hater 95% CI'}, ...
         'Location','northwest','FontSize',12,'Box','off');
    axis(axL,'off');

    out_png = fullfile(fig_root, sprintf('%s_group_TRF_Hater_vs_NonHater.png', G));
    print(f, out_png, '-dpng', '-r200');
    fprintf('Saved: %s\n', out_png);
    close all;
end
% ---------- helpers ----------
function [H, D, ok] = lw_load_trf(base_no_ext)
% Load Letswave header+data saved as .lw6 (header) + .mat (data)
ok = false; H = []; D = [];
try
    S = load([base_no_ext '.lw6'],'-mat');   % header struct in S.header
    if isfield(S,'header'), H = S.header; else, return; end
    M = load([base_no_ext '.mat'],'-mat');   % numeric tensor in M.data
    if isfield(M,'data'),   D = M.data;   else, return; end
    ok = true;
catch
    ok = false;
end
end

function lab = get_chan_labels(H, nCh)
lab = repmat("Ch",1,nCh);
try
    if isfield(H,'chanlocs') && ~isempty(H.chanlocs)
        tmp = string(arrayfun(@(c)c.labels, H.chanlocs, 'UniformOutput', false));
        if numel(tmp)==nCh, lab = tmp; end
    end
end
end

function [mu, ci] = mean_ci(X)
% X: [nSub × nLags]
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
           'FaceAlpha',0.35, 'EdgeColor','none');
end
