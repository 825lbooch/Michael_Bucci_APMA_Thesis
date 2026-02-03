%% BATCH SIMULATION SCRIPT: OSCAR HPC Version
% Optimized for Brown University OSCAR cluster
% 500 frequency points for better resonance resolution
%
% Usage on OSCAR:
%   sbatch run_antenna_sim.sh
%
% OUTPUTS: .mat dataset AND .csv summary report
clear; clc;

%% 1. SETUP & CONFIGURATION
fprintf('=================================================================\n');
fprintf('ANTENNA BATCH SIMULATION (OSCAR HPC)\n');
fprintf('=================================================================\n\n');

input_csv = 'antenna_params_10k.csv';
output_mat = 'dataset_10k_raw.mat';
report_csv = 'simulation_summary_10k.csv';
plot_folder = 'QC_Plots';

% UPGRADED: 500 frequency points for better resonance resolution
freq_points = 500;
freq_sweep = linspace(1.5e9, 3.5e9, freq_points);

fprintf('Frequency resolution: %.2f MHz (%d points)\n', ...
    (freq_sweep(2)-freq_sweep(1))/1e6, freq_points);

% Simulation Constants
gap = 0.5e-3;
feedLength = 20e-3;
MIN_VIA_DIAMETER = 0.2e-3;

if ~exist(plot_folder, 'dir')
    mkdir(plot_folder);
end

%% 2. SETUP PARALLEL POOL
fprintf('Setting up parallel pool...\n');
delete(gcp('nocreate'));

% OSCAR: Use available cores (typically request 16-32 in SLURM)
% The pool will auto-detect from SLURM allocation
try
    pool = parpool('local');
    fprintf('  Started parallel pool with %d workers\n', pool.NumWorkers);
catch ME
    fprintf('  Warning: %s\n', ME.message);
    fprintf('  Continuing without parallel pool...\n');
end

%% 3. LOAD DATA
if ~isfile(input_csv)
    error('Input file "%s" not found!', input_csv);
end

opts = detectImportOptions(input_csv);
opts.VariableNamingRule = 'preserve';
data_table = readtable(input_csv, opts);
n_samples = height(data_table);
fprintf('  Loaded %d samples.\n\n', n_samples);

% Extract arrays
L_mm_all = data_table.L_mm;
W_mm_all = data_table.W_mm;
inset_mm_all = data_table.inset_mm;
feedWidth_mm_all = data_table.feedWidth_mm;
h_mm_all = data_table.h_mm;
eps_r_all = data_table.eps_r;

% Pre-allocate
S11_Complex_Matrix = zeros(n_samples, freq_points);
Design_Parameters = zeros(n_samples, 6);
Success_Flags = false(n_samples, 1);
S11_min_dB = nan(n_samples, 1);
Res_Freq_GHz = nan(n_samples, 1);
Bandwidth_MHz = nan(n_samples, 1);  % NEW: Track -10dB bandwidth

%% 4. BATCH SIMULATION
fprintf('Starting Batch Simulation (%d samples)...\n', n_samples);
fprintf('Estimated time: %.1f - %.1f hours\n', n_samples*0.5/60, n_samples*1.0/60);
fprintf('-----------------------------------------------------------------\n');

tic;

% Progress tracking for OSCAR logs
progress_interval = max(1, floor(n_samples / 100));  % Log every 1%

parfor i = 1:n_samples
    try
        % --- Parameters ---
        L_mm = L_mm_all(i);
        W_mm = W_mm_all(i);
        inset_mm = inset_mm_all(i);
        feedWidth_mm = feedWidth_mm_all(i);
        h_mm = h_mm_all(i);
        eps_r = eps_r_all(i);

        Design_Parameters(i, :) = [L_mm, W_mm, inset_mm, feedWidth_mm, h_mm, eps_r];

        L = L_mm / 1000;
        W = W_mm / 1000;
        inset = inset_mm / 1000;
        feedWidth = feedWidth_mm / 1000;
        h = h_mm / 1000;

        % --- Dynamic Via Logic ---
        viaDia = min(0.4 * feedWidth, 0.8e-3);
        viaDia = max(viaDia, MIN_VIA_DIAMETER);

        % --- Build Geometry ---
        patch_shape = antenna.Rectangle('Length', L, 'Width', W, 'Center', [L/2, 0]);
        notch_W = feedWidth + (2 * gap);
        notch_shape = antenna.Rectangle('Length', inset, 'Width', notch_W, 'Center', [inset/2, 0]);
        patch_with_cutout = patch_shape - notch_shape;

        feed_total_len = feedLength + inset;
        feed_center = -feedLength/2 + inset/2;
        feedStrip = antenna.Rectangle('Length', feed_total_len, 'Width', feedWidth, ...
            'Center', [feed_center, 0]);
        topMetal = patch_with_cutout + feedStrip;

        board_L = L + 40e-3;
        board_W = W + 40e-3;
        groundPlane = antenna.Rectangle('Length', board_L, 'Width', board_W, 'Center', [L/2, 0]);

        sub = dielectric('Name', sprintf('Sub_%d', i));
        sub.EpsilonR = eps_r;
        sub.LossTangent = 0.02;
        sub.Thickness = h;

        p = pcbStack;
        p.Layers = {topMetal, sub, groundPlane};
        p.BoardShape = groundPlane;
        p.BoardThickness = h;
        p.FeedDiameter = viaDia;
        p.FeedLocations = [-feedLength + 2e-3, 0, 1, 3];

        % --- Simulate ---
        S = sparameters(p, freq_sweep);

        s11_data = squeeze(S.Parameters(1,1,:));
        S11_Complex_Matrix(i, :) = s11_data.';
        Success_Flags(i) = true;

        % Calculate Metrics
        s11_db_vals = 20*log10(abs(s11_data));
        [min_val, min_idx] = min(s11_db_vals);

        S11_min_dB(i) = min_val;
        Res_Freq_GHz(i) = freq_sweep(min_idx)/1e9;

        % Calculate -10dB bandwidth
        below_10 = s11_db_vals < -10;
        if sum(below_10) >= 2
            indices = find(below_10);
            Bandwidth_MHz(i) = (freq_sweep(indices(end)) - freq_sweep(indices(1))) / 1e6;
        end

        % Progress (will show in SLURM output)
        if mod(i, progress_interval) == 0
            fprintf('  [%5d/%5d] %.1f%% complete\n', i, n_samples, i/n_samples*100);
        end

    catch ME
        Success_Flags(i) = false;
        fprintf('  [%5d] FAILED: %s\n', i, ME.message);
    end
end

elapsed_time = toc;
fprintf('-----------------------------------------------------------------\n');
fprintf('Simulation complete! Time: %.1f hours\n\n', elapsed_time/3600);

%% 5. QC PLOTS (First 100)
fprintf('Generating QC plots...\n');
n_qc_plots = min(100, n_samples);

for i = 1:n_qc_plots
    if Success_Flags(i)
        try
            f_debug = figure('Visible', 'off');
            s11_dB = 20*log10(abs(S11_Complex_Matrix(i, :)) + 1e-9);
            plot(freq_sweep/1e9, s11_dB, 'LineWidth', 1.5);
            grid on; hold on; yline(-10, '--r');
            title(sprintf('Sample %d: %.1fdB @ %.2f GHz', i, S11_min_dB(i), Res_Freq_GHz(i)));
            xlabel('Frequency (GHz)');
            ylabel('S11 (dB)');
            ylim([-40, 5]);
            saveas(f_debug, fullfile(plot_folder, sprintf('Sample_%05d.png', i)));
            close(f_debug);
        catch; end
    end
end

%% 6. SAVE DATASET (MAT)
fprintf('Saving dataset...\n');

valid_indices = Success_Flags;
Geometry = Design_Parameters(valid_indices, :);
S11_Complex = S11_Complex_Matrix(valid_indices, :);
geometry_columns = {'L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r'};

save(output_mat, 'Geometry', 'S11_Complex', 'freq_sweep', 'geometry_columns', '-v7.3');
fprintf('  Dataset saved to %s\n', output_mat);

%% 7. GENERATE SUMMARY REPORT (CSV)
fprintf('Generating summary CSV...\n');

results_table = table((1:n_samples)', ...
    L_mm_all, W_mm_all, inset_mm_all, feedWidth_mm_all, h_mm_all, eps_r_all, ...
    S11_min_dB, Res_Freq_GHz, Bandwidth_MHz, Success_Flags, ...
    'VariableNames', {'Sample_ID', 'L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', ...
                      'h_mm', 'eps_r', 'S11_min_dB', 'Res_Freq_GHz', 'Bandwidth_MHz', 'Sim_Success'});

% Add Status column
status_col = cell(n_samples, 1);
for i = 1:n_samples
    if ~Success_Flags(i)
        status_col{i} = 'FAILED_SIM';
    elseif S11_min_dB(i) < -15
        status_col{i} = 'EXCELLENT';
    elseif S11_min_dB(i) < -10
        status_col{i} = 'GOOD_MATCH';
    elseif S11_min_dB(i) < -6
        status_col{i} = 'MARGINAL';
    else
        status_col{i} = 'MISMATCHED';
    end
end
results_table.Status = status_col;

writetable(results_table, report_csv);
fprintf('  Summary report saved to %s\n', report_csv);

%% 8. FINAL STATISTICS
fprintf('\n=================================================================\n');
fprintf('SIMULATION SUMMARY\n');
fprintf('=================================================================\n');

n_success = sum(Success_Flags);
n_excellent = sum(S11_min_dB < -15 & Success_Flags);
n_good = sum(S11_min_dB >= -15 & S11_min_dB < -10 & Success_Flags);
n_marginal = sum(S11_min_dB >= -10 & S11_min_dB < -6 & Success_Flags);
n_mismatch = sum(S11_min_dB >= -6 & Success_Flags);

fprintf('  Total samples:     %d\n', n_samples);
fprintf('  Successful sims:   %d (%.1f%%)\n', n_success, n_success/n_samples*100);
fprintf('  Failed sims:       %d (%.1f%%)\n', n_samples-n_success, (n_samples-n_success)/n_samples*100);
fprintf('\n  Matching Quality:\n');
fprintf('    EXCELLENT (<-15dB): %d (%.1f%%)\n', n_excellent, n_excellent/n_success*100);
fprintf('    GOOD (<-10dB):      %d (%.1f%%)\n', n_good, n_good/n_success*100);
fprintf('    MARGINAL (<-6dB):   %d (%.1f%%)\n', n_marginal, n_marginal/n_success*100);
fprintf('    MISMATCHED:         %d (%.1f%%)\n', n_mismatch, n_mismatch/n_success*100);
fprintf('\n  Total time: %.1f hours\n', elapsed_time/3600);
fprintf('  Time per sample: %.1f seconds\n', elapsed_time/n_success);
fprintf('=================================================================\n');
