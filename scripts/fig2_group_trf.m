clc; close all;

addpath('./utility/');
out_trf_root = fullfile('..','out_trf');            % folder with DLS##_E folders
genres       = {'Blues','Metal','Pop'};             % adjust if you also have 'Control'
fig_root     = fullfile('..','figures');          % where to save figures
if ~exist(fig_root,'dir'); mkdir(fig_root); end

% ---- collect subject folders ----
S = dir(fullfile(out_trf_root,'DLS*_E'));
S = S([S.isdir]);

% ---- helper: nice colors ----
colMean = [0.00 0.45 0.74];           % mean line (blue-ish)
ciFill  = [0.30 0.60 0.85];           % CI fill
nCh = 8;
chLabs = ["F3","F4","C3","C4","P3","P4","O1","O2"];
for g = 1:numel(genres)
    gname = genres{g};

    % Pre-collect per-subject arrays (we'll align them after we know sizes)
    subjTRF   = {};                    % each cell -> [nCh x nLags]
    subjIDs   = strings(0,1);
    t_ms      = [];

    for s = 1:numel(S)
        subj_id   = S(s).name;                             % e.g., 'DLS32_E'
        base_name = sprintf('avg merge trial_wise TRF %s', gname);
        base_path = fullfile(out_trf_root, subj_id, base_name);

        [H, D, ok] = lw_load_trf(base_path);
        if ~ok, fprintf('[skip] %s — missing: %s.lw6/.mat\n', subj_id, base_name); continue; end

        % squeeze to [nCh x nLags]
        D = squeeze(D);

        % time axis (ms)
        if isempty(t_ms)
            nL = size(D,2);
            t_ms = (H.xstart + (0:nL-1).*H.xstep) * 1000;
        end

        % channel labels (keep once)
        % Channel labels (fallback if empty)

        subjTRF{end+1} = D; 
        subjIDs(end+1)  = string(erase(subj_id,'_E')); 
    end

    nSub = numel(subjTRF);
    if nSub==0
        warning('No subjects found for genre "%s".', gname);
        continue;
    end

    % ---- stack subjects -> [nSub x nCh x nLags] ----
    nCh   = size(subjTRF{1},1);
    nLags = size(subjTRF{1},2);
    X = nan(nSub, nCh, nLags);
    for i = 1:nSub
        Xi = subjTRF{i};
        if ~isequal(size(Xi), [nCh, nLags])
            % crude safeguard: truncate or pad with NaN if needed
            nCh_i = min(nCh, size(Xi,1));
            nLg_i = min(nLags, size(Xi,2));
            X(i,1:nCh_i,1:nLg_i) = Xi(1:nCh_i,1:nLg_i);
        else
            X(i,:,:) = Xi;
        end
    end
    X = X(:,:,3:end-3);
    t_ms = t_ms(3:end-3);
    % X is [nSub x nCh x nLags]
    mu  = squeeze(mean(X,1,'omitnan'));      % [nCh x nLags]
    sd  = squeeze(std (X,0,1,'omitnan'));    % [nCh x nLags]
    
    % how many subjects contributed at each [ch,lag]
    n   = squeeze(sum(~isnan(X),1));         % [nCh x nLags]
    n   = max(n,1);                          % avoid divide-by-zero
    
    se  = sd ./ sqrt(n);                     % [nCh x nLags]
    dof = max(n-1,1);                        % [nCh x nLags]
    tcrit = tinv(0.975, dof);                % 95% CI
    ci  = tcrit .* se;                       % [nCh x nLags]

    % ---- plot (3x3 grid, leave last empty) ----
    f = figure('Color','w','Units','pixels','Position',[100 100 900 700]);
    sgtitle(sprintf('%s — Group mean trial\\_wise TRF (N=%d)', gname, nSub), ...
            'FontSize',18,'FontWeight','bold');

    panel = 0;
    for ch = 1:nCh
        panel = panel + 1;
        subplot(3,3,panel); hold on;

        % shaded CI
        xx = [t_ms, fliplr(t_ms)];
        yy = [mu(ch,:)-ci(ch,:), fliplr(mu(ch,:)+ci(ch,:))];
        patch('XData',xx,'YData',yy, ...
              'FaceColor',ciFill,'FaceAlpha',0.25,'EdgeColor','none');

        % mean line
        plot(t_ms, mu(ch,:), 'LineWidth',1.8, 'Color',colMean);

        box off
        title(strtrim(chLabs(ch)), 'FontWeight','bold');
        xlabel('Lag (ms)'); ylabel('w');
        set(gca,'TickDir','out');
        set(gca,'xlim',[t_ms(1),t_ms(end)]);
    end

    % leave last panel blank if exactly 8 channels
    if nCh < 9
        subplot(3,3,9); axis off;
    end

    % legend block
    ax = subplot(3,3,9); hold(ax,'on');
    plot(ax, nan, nan, 'Color',colMean,'LineWidth',1.8);
    patch(ax, 'XData',[0 1 1 0], 'YData',[0 0 1 1], 'Visible','off','FaceColor',ciFill,'FaceAlpha',0.25,'EdgeColor','none');
    legend(ax, {'Group mean','95% CI (across subjects)'}, 'Location','northwest','FontSize',12,'Box','off');
    axis(ax,'off');

    % save
    out_png = fullfile(fig_root, sprintf('GROUP_%s_trialwise_TRF_CI.png', gname));
    print(f, out_png, '-dpng', '-r200');
    fprintf('Saved: %s\n', out_png);
    close(f);
end
