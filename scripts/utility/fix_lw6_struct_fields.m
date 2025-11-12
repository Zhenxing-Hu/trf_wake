function fix_lw6_struct_fields(pathOrFile)
% Convert header.history / header.chanlocs / header.events from cell -> struct arrays.
% Usage:
%   fix_lw6_struct_fields('C:\...\myfile.lw6')
%   fix_lw6_struct_fields('C:\...\folder_with_lw6')

  if isfolder(pathOrFile)
    files = dir(fullfile(pathOrFile, '*.lw6'));
    for k = 1:numel(files)
      fix_one(fullfile(files(k).folder, files(k).name));
    end
  else
    fix_one(pathOrFile);
  end
end

function fix_one(lw6_path)
  s = load(lw6_path, '-mat');
  if ~isfield(s, 'header'); warning('No header in %s', lw6_path); return; end
  hdr = s.header;

  % Convert cells-of-structs -> 1xN struct arrays (Letswave expects structs here)
  if iscell(hdr.history),  try hdr.history  = [hdr.history{:}];  end, end
  if iscell(hdr.chanlocs), try hdr.chanlocs = [hdr.chanlocs{:}]; end, end
  if iscell(hdr.events),   try hdr.events   = [hdr.events{:}];   end, end

  % Ensure datasize is a 1x6 double row vector (sometimes becomes column)
  if isfield(hdr,'datasize')
    hdr.datasize = double(reshape(hdr.datasize, 1, []));
  end

  % Write back
  header = hdr; %#ok<NASGU>
  save(lw6_path, 'header', '-mat');
  fprintf('Fixed: %s\n', lw6_path);
end
