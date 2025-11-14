function lab = get_chan_labels(H, nCh)
    lab = repmat("Ch",1,nCh);
    try
        if isfield(H,'chanlocs') && ~isempty(H.chanlocs)
            tmp = string(arrayfun(@(c)c.labels, H.chanlocs, 'UniformOutput', false));
            if numel(tmp)==nCh, lab = tmp; end
        end
    end
end