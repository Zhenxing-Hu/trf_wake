function plot_erp_vs_recon_3x3(erp_base, recon_tw_base,out_root,subj_id,gname)
% Compare averaged ERP vs averaged reconstructions (genre_pooled & trial_wise)
% in a 3x3 panel (last subplot blank).
%
% Inputs are basenames WITHOUT extensions for each dataset, e.g.:
%   erp_base     = 'G:\...\DLS32_E\avg Blues-inst merge ERP'
%   recon_gp_base= 'G:\...\DLS32_E\avg recon genre_pooled TRF Blues-inst'
%   recon_tw_base= 'G:\...\DLS32_E\avg recon trial_wise TRF Blues-inst'
%
% Each dataset should have a matching .lw6 (header) and .mat (data).

    % ---- Load datasets -------------------------------------------------
    chLabels = ["F3","F4","C3","C4","P3","P4","O1","O2"];
    [t, ERP]      = load_lw_avg(erp_base);      % [T x nCh]
    [t_tw, TW] = load_lw_avg(recon_tw_base); % [T x nCh]


    T = min([size(ERP,1), size(TW,1)]);
    ERP = ERP(1:T,:); TW = TW(1:T,:);
    t   = t(1:T); % assume same sampling; if small drift, this still plots fine

    % ---- Rescale predictions to ERP per-channel (match mean & std) ------
    TW_rs = rescale_to( TW, ERP );

    % ---- 3x3 panel (last empty) -----------------------------------------
    nCh = size(ERP,2);
    nPlot = min(nCh, 8);  % first 8 channels; last (9th) left empty
    fig = figure('Position',[275.4000  60.8000  900.4000  700.4000],'Name','ERP vs Recon (avg)');
    sgtitle(sprintf('%s — %s', subj_id(1:end-1), gname),'FontSize',18,'FontWeight','bold');
    for i = 1:9
        subplot(3,3,i);
        if i <= nPlot
            plot(t, ERP(:,i), 'LineWidth',1.2); hold on;
            plot(t, TW_rs(:,i),'LineWidth',1.0);
            [r_tw,p_tw] = corr(TW_rs(1:200,i), ERP(1:200,i), 'type','Pearson','rows','complete');
            
            % helper to print nice p-value
            
            % --- top label: trial-wise ---
            text(0.08, 0.98, sprintf('$r = %.2f,\\; p = %.1e$', r_tw, p_tw), ...
                'Units','normalized','Interpreter','latex','FontSize',11,'FontWeight','bold', ...
                'VerticalAlignment','top','Margin',1);
            

            title(strtrim(chLabels{i}), 'FontWeight','bold');
            xlabel('Time (s)'); ylabel('Amplitude');
            set(gca,'xlim',[t(1),t(1)+2]);
            if i== 8
                leg = legend({'avg ERP','avg recon trial\_wise'},'FontSize',12); %'avg recon genre\_pooled',
                legend boxoff;
                set(leg,'Position',[0.8 0.15 0.05 0.2])
            end

        else
            axis off; % leave bottom-right empty
        end
    end
    filename = ['../figures/',subj_id(1:end-1),'_',gname,'_erp_vs_recon.tiff'];
    saveas(gcf,filename);
end

% ---------- helpers ----------

function [t, X] = load_lw_avg(base)
% Load .lw6 header and .mat data, return:
%   t         : [T x 1] time vector (s)
%   chLabels  : cellstr channel labels (1..nCh)
%   X         : [T x nCh] average across epochs (if >1)

    S = load([base '.lw6'],'-mat');  % header only
    if ~isfield(S,'header'), error('No header in %s.lw6', base); end
    H = S.header;

    M = load([base '.mat'],'-mat');  % data only
    if ~isfield(M,'data'), error('No data in %s.mat', base); end
    D = M.data;   % [ep, ch, 1, 1, 1, T]

    % Average across epochs if multiple
    % Dm = squeeze(mean(D,1));           % [ch, 1, 1, 1, T]
    X  = squeeze(D)';     % [T x ch]

    % Time vector
    T = size(X,1);
    t = H.xstart + (0:T-1)' * H.xstep;
end

function Xrs = rescale_to(Xpred, Xref)
% Match mean and std of Xpred to Xref per channel (over time).
% Both T x nCh.
    mu_ref = mean(Xref,1);
    sd_ref = std(Xref,0,1);
    mu_pr  = mean(Xpred,1);
    sd_pr  = std(Xpred,0,1);
    sd_pr(sd_pr==0) = 1;  % avoid div-by-zero
    a = sd_ref ./ sd_pr;               % scale
    b = mu_ref - a .* mu_pr;           % offset
    % apply per channel
    Xrs = Xpred .* a + b;
end
