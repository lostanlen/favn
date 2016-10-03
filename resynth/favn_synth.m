function favn_synth(folder, splice_index)
N = 2^20; % length of the signal

% Load target waveform
prefix = ['F_Tele_FAVN_03_', folder, '_splice'];
splice_str = sprintf('%0.2d', splice_index);
file_str = [prefix, splice_str, '.wav'];
path_str = fullfile(folder, file_str);
[splice, sample_rate, bit_depth] = eca_load(path_str, N);

%
y = planck_taper(N) .* splice;

%
Q1 = 12; % number of filters per octave at first order
T = 2^13; % amount of invariance with respect to time translation
% The modulation setting is either 'none', 'time', or 'time-frequency'
% The wavelets setting is either 'morlet' or 'gammatone'
modulations = 'time-frequency';
wavelets = 'gammatone';

% Setup reconstruction options
clear opts;
opts.is_sonified = true;
opts.is_spectrogram_displayed = false;
% (close Figure 1 to abort early)
opts.nIterations = 50;
opts.sample_rate = sample_rate;
opts.generate_text = true;
opts.is_verbose = true;
opts.initial_learning_rate = 0.1;

%% Setup
archs = eca_setup_1chunk(Q1, T, modulations, wavelets, N);
opts = fill_reconstruction_opt(opts);

%% Initialization
target_S = eca_target(y, archs);
[target_norm, layer_target_norms] = sc_norm(target_S);
nLayers = length(archs);
[init, previous_loss, delta_signal] = eca_init(y, target_S, archs, opts);
iterations = cell(1, opts.nIterations);
iterations{1+0} = init;
previous_signal = iterations{1+0};
relative_loss_chart = zeros(opts.nIterations, 1);
signal_update = zeros(size(iterations{1+0}));
learning_rate = opts.initial_learning_rate;
max_nDigits = 1 + floor(log10(opts.nIterations));
sprintf_format = ['%0.', num2str(max_nDigits), 'd'];

%% Iterated reconstruction
iteration = 1;
failure_counter = 0;
tic();

while (iteration <= opts.nIterations)
    %% Signal update
    iterations{1+iteration} = ...
        update_reconstruction(previous_signal, ...
        delta_signal, ...
        signal_update, ...
        learning_rate, ...
        opts);
    
    %% Signal export
    export_file_str = [file_str(1:(end-4)), '_it', ...
        sprintf('%0.2d', iteration), ...
        '.wav'];
    export_path_str = fullfile(folder, export_file_str);
    audiowrite(export_path_str, iterations{1+iteration}, sample_rate, ...
        'BitsPerSample', bit_depth);
    %system(['git add ', export_path_str]);
    %system(['git commit -m "Upload ', export_file_str, '"']);
    %system('git push');
    
    
    %% Scattering propagation
    S = cell(1, nLayers);
    U = cell(1,nLayers);
    Y = cell(1,nLayers);
    U{1+0} = initialize_variables_auto(size(y));
    U{1+0}.data = iterations{1+iteration};
    for layer = 1:nLayers
        arch = archs{layer};
        previous_layer = layer - 1;
        if isfield(arch, 'banks')
            Y{layer} = U_to_Y(U{1+previous_layer}, arch.banks);
        else
            Y{layer} = U(1+previous_layer);
        end
        if isfield(arch, 'nonlinearity')
            U{1+layer} = Y_to_U(Y{layer}{end}, arch.nonlinearity);
        end
        if isfield(arch, 'invariants')
            S{1+previous_layer} = Y_to_S(Y{layer}, arch);
        end
    end
    
    %% Measurement of distance to target in the scattering domain
    delta_S = sc_substract(target_S,S);
    
    %% If loss has increased, step retraction and bold driver "brake"
    [loss, layer_absolute_distances] = sc_norm(delta_S);
    if opts.adapt_learning_rate && (loss > previous_loss)
        learning_rate = ...
            opts.bold_driver_brake * learning_rate;
        signal_update = ...
            opts.bold_driver_brake * signal_update;
        disp(['Learning rate = ', num2str(learning_rate)]);
        failure_counter = failure_counter + 1;
        if failure_counter > 3
            learning_rate = opts.initial_learning_rate;
        else
            continue
        end
    end
    
    %% If loss has decreased, step confirmation and bold driver "acceleration"
    iteration = iteration + 1;
    failure_counter = 0;
    relative_loss_chart(iteration) = 100 * loss / target_norm;
    previous_signal = iterations{iteration};
    previous_loss = loss;
    signal_update = ...
        opts.momentum * signal_update + ...
        learning_rate * delta_signal;
    learning_rate = ...
        opts.bold_driver_accelerator * ...
        learning_rate;
    
    %% Backpropagation
    delta_signal = sc_backpropagate(delta_S, U, Y, archs);
    
    %% Pretty-printing of scattering distances and loss function
    if opts.is_verbose
        pretty_iteration = sprintf(sprintf_format, iteration);
        layer_distances = ...
            100 * layer_absolute_distances ./ layer_target_norms;
        pretty_distances = num2str(layer_distances(2:end), '%8.2f%%');
        pretty_loss = sprintf('%.2f%%',relative_loss_chart(iteration));
        iteration_string = ['it = ', pretty_iteration, '  ;  '];
        distances_string = ...
            ['S_m distances = [ ',pretty_distances, ' ]  ;  '];
        loss_string = ['Loss = ', pretty_loss];
        disp([iteration_string, distances_string, loss_string]);
        disp(['Learning rate = ', num2str(learning_rate)]);
        toc();
        tic();
    end
end
toc();
end