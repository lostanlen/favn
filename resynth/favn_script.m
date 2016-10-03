folder = 'A';
parpool(8);
parfor splice_index = 1:47
    favn_synth(folder, splice_index);
end