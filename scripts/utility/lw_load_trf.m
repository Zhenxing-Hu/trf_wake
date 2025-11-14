function [H, D, ok] = lw_load_trf(base)
% Load Letswave header & data.
% base: path without extension
% Returns:
%   H: header struct (fields: xstart, xstep, chanlocs, datasize, ...)
%   D: data as [nEp x nCh x nLags]
%   ok: true if files were found and parsed

H = struct(); D = []; ok = false;

lw6 = [base '.lw6']; mat = [base '.mat'];
if ~exist(lw6,'file') || ~exist(mat,'file'); return; end

S = load(lw6,'-mat');
if ~isfield(S,'header'); return; end
H = S.header;

M = load(mat,'-mat');
if ~isfield(M,'data'); return; end
X = M.data;

% Expect [nEp, nCh, 1, 1, 1, nLags] → [nEp x nCh x nLags]
switch ndims(X)
    case 6
        D = squeeze(permute(X,[1 2 6 3 4 5]));
    case 3
        D = X;
    otherwise
        % try to coerce generically: put ep,ch,last-dim as lags
        sz = size(X);
        if numel(sz)>=3
            D = squeeze(permute(X,[1 2 numel(sz) 3: numel(sz)-1]));
        else
            return
        end
end

ok = true;
end