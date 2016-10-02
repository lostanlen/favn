nSplices = 47;
folder = 'A';

splice_index = 1;

N = 2^20; % length of the signal

% Load target waveform
prefix = ['F_Tele_FAVN_03_', folder, '_splice'];
splice_str = sprintf('%0.2d', splice_index);
file_str = [prefix, splice_str, '.wav'];
path_str = fullfile(folder, file_str);
[splice, sample_rate] = eca_load(path_str, N);

%%
target_waveform = planck_taper(N) .* splice;

%
Q1 = 12; % number of filters per octave at first order
T = 2^13; % amount of invariance with respect to time translation
% The modulation setting is either 'none', 'time', or 'time-frequency'
% The wavelets setting is either 'morlet' or 'gammatone'
modulations = 'time-frequency';
wavelets = 'morlet';

% Setup reconstruction options
clear opts;
opts.is_sonified = true;
opts.is_spectrogram_displayed = false;
% (close Figure 1 to abort early)
opts.nIterations = 50;
opts.sample_rate = sample_rate;
opts.generate_text = false;
opts.is_verbose = true;
opts.initial_learning_rate = 0.1;

%%
archs_singlechunk = eca_setup_1chunk(Q1, T, modulations, wavelets, N);
%%
iterations = ...
    eca_synthesize_1chunk(target_waveform, archs_singlechunk, opts);