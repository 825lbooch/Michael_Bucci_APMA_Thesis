%% BATCH SIMULATION SCRIPT: OSCAR HPC Version (WITH CHECKPOINTS)
% Optimized for Brown University OSCAR cluster
% 500 frequency points for better resonance resolution
% SAVES CHECKPOINT EVERY 100 SAMPLES
%
% Usage on OSCAR:
%   sbatch run_antenna_sim.sh
%
% OUTPUTS: .mat dataset AND .csv summary report
clear; clc;

%% 1. SETUP & CONFIGURATION
fprintf('=================================================================\n');
fprintf('ANTENNA BATCH SIMULATION (OSCAR HPC) - WITH CHECKPOINTS\n');
fprintf('=================================================================\n\n');

input_csv = 'antenna_params_1k.csv';
output_mat = 'dataset_1k_raw.mat';
report_csv = 'simulation_summary_1k.csv';
checkpoint_mat = 'checkpoint_1k.mat';  % NEW: Checkpoint file
plot_folder = 'QC_Plots';

% UPGRADED: 500 frequency points for better resonance resolution
freq_points = 500;
freq_sweep = linspace(1.5e9, 3.5e9, freq_points);

% Checkpoint settings
CHUNK_SIZE = 100;  % Save every 100 samples

fprintf('Frequency resolution: %.2f MHz (%d points)\n', ...
    (freq_sweep(2)-freq_sweep(1))/1e6, freq_points);
fprintf('Checkpoint interval: every %d samples\n', CHUNK_SIZE);

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

%% 4. CHECK FOR EXISTING CHECKPOINT
start_chunk = 1;
if isfile(checkpoint_mat)
    fprintf('Found existing checkpoint! Loading...\n');
    load(checkpoint_mat, 'S11_Complex_Matrix', 'Design_Parameters', 'Success_Flags', ...
         'S11_min_dB', 'Res_Freq_GHz', 'Bandwidth_MHz', 'completed_samples', 'elapsed_time_prev');
    start_chunk = floor(completed_samples / CHUNK_SIZE) + 1;
    fprintf('  Resuming from chunk %d (sample %d)\n', start_chunk, (start_chunk-1)*CHUNK_SIZE + 1);
    fprintf('  Previous elapsed time: %.1f hours\n\n', elapsed_time_prev/3600);
else
    % Pre-allocate fresh arrays
    S11_Complex_Matrix = zeros(n_samples, freq_points);
    Design_Parameters = zeros(n_samples, 6);
    Success_Flags = false(n_samples, 1);
    S11_min_dB = nan(n_samples, 1);
    Res_Freq_GHz = nan(n_samples, 1);
    Bandwidth_MHz = nan(n_samples, 1);
    elapsed_time_prev = 0;
end

%% 5. SINGLE SAMPLE TEST (Verify toolbox works)
fprintf('Testing single sample before batch run...\n');

test_success = false;
try
    L_mm = L_mm_all(1);
    W_mm = W_mm_all(1);
    inset_mm = inset_mm_all(1);
    feedWidth_mm = feedWidth_mm_all(1);
    h_mm = h_mm_all(1);
    eps_r = eps_r_all(1);

    L = L_mm / 1000;
    W = W_mm / 1000;
    inset = inset_mm / 1000;
    feedWidth = feedWidth_mm / 1000;
    h = h_mm / 1000;

    viaDia = min(0.4 * feedWidth, 0.8e-3);
    viaDia = max(viaDia, MIN_VIA_DIAMETER);

    fprintf('  Creating antenna geometry...\n');
    patch_shape = antenna.Rectangle('Length', L, 'Width', W, 'Center', [L/2, 0]);
    notch_W = feedWidth + (2 * gap);
    notch_shape = antenna.Rectangle('Length', inset, 'Width', notch_W, 'Center', [inset/2, 0]);
    patch_with_cutout = patch_shape - notch_shape;
    feed_total_len = feedLength + inset;
    feed_center = -feedLength/2 + inset/2;
    feedStrip = antenna.Rectangle('Length', feed_total_len, 'Width', feedWidth, 'Center', [feed_center, 0]);
    topMetal = patch_with_cutout + feedStrip;
    board_L = L + 40e-3;
    board_W = W + 40e-3;
    groundPlane = antenna.Rectangle('Length', board_L, 'Width', board_W, 'Center', [L/2, 0]);

    sub = dielectric('Name', 'Sub_test');
    sub.EpsilonR = eps_r;
    sub.LossTangent = 0.02;
    sub.Thickness = h;

    p = pcbStack;
    p.Layers = {topMetal, sub, groundPlane};
    p.BoardShape = groundPlane;
    p.BoardThickness = h;
    p.FeedDiameter = viaDia;
    p.FeedLocations = [-feedLength + 2e-3, 0, 1, 3];

    fprintf('  Running test sparameters...\n');
    S = sparameters(p, freq_sweep);

    s11_data = squeeze(S.Parameters(1,1,:));
    s11_db = 20*log10(abs(s11_data));
    [min_val, min_idx] = min(s11_db);

    fprintf('  TEST RESULT: S11_min = %.1f dB @ %.2f GHz\n', min_val, freq_sweep(min_idx)/1e9);
    test_success = true;

catch ME
    fprintf('\n!!! TEST SAMPLE FAILED !!!\n');
    fprintf('Error: %s\n', ME.message);
end

if ~test_success
    fprintf('\nAborting due to test failure.\n');
    save(output_mat, 'freq_sweep', '-v7.3');
    exit(1);
end

fprintf('\nTest passed! Starting batch simulation with checkpoints...\n\n');

%% 6. BATCH SIMULATION WITH CHECKPOINTS
fprintf('Starting Batch Simulation (%d samples in chunks of %d)...\n', n_samples, CHUNK_SIZE);
fprintf('-----------------------------------------------------------------\n');

total_tic = tic;
n_chunks = ceil(n_samples / CHUNK_SIZE);

for chunk = start_chunk:n_chunks
    chunk_start = (chunk - 1) * CHUNK_SIZE + 1;
    chunk_end = min(chunk * CHUNK_SIZE, n_samples);
    chunk_indices = chunk_start:chunk_end;
    n_in_chunk = length(chunk_indices);

    fprintf('\n>>> CHUNK %d/%d (samples %d-%d) <<<\n', chunk, n_chunks, chunk_start, chunk_end);
    chunk_tic = tic;

    % Temporary arrays for this chunk
    chunk_S11 = zeros(n_in_chunk, freq_points);
    chunk_Params = zeros(n_in_chunk, 6);
    chunk_Success = false(n_in_chunk, 1);
    chunk_S11min = nan(n_in_chunk, 1);
    chunk_ResFreq = nan(n_in_chunk, 1);
    chunk_BW = nan(n_in_chunk, 1);

    % Extract chunk data
    chunk_L = L_mm_all(chunk_indices);
    chunk_W = W_mm_all(chunk_indices);
    chunk_inset = inset_mm_all(chunk_indices);
    chunk_feedW = feedWidth_mm_all(chunk_indices);
    chunk_h = h_mm_all(chunk_indices);
    chunk_eps = eps_r_all(chunk_indices);

    parfor j = 1:n_in_chunk
        try
            L_mm = chunk_L(j);
            W_mm = chunk_W(j);
            inset_mm = chunk_inset(j);
            feedWidth_mm = chunk_feedW(j);
            h_mm = chunk_h(j);
            eps_r = chunk_eps(j);

            chunk_Params(j, :) = [L_mm, W_mm, inset_mm, feedWidth_mm, h_mm, eps_r];

            L = L_mm / 1000;
            W = W_mm / 1000;
            inset = inset_mm / 1000;
            feedWidth = feedWidth_mm / 1000;
            h = h_mm / 1000;

            viaDia = min(0.4 * feedWidth, 0.8e-3);
            viaDia = max(viaDia, MIN_VIA_DIAMETER);

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

            sub = dielectric('Name', sprintf('Sub_%d', j));
            sub.EpsilonR = eps_r;
            sub.LossTangent = 0.02;
            sub.Thickness = h;

            p = pcbStack;
            p.Layers = {topMetal, sub, groundPlane};
            p.BoardShape = groundPlane;
            p.BoardThickness = h;
            p.FeedDiameter = viaDia;
            p.FeedLocations = [-feedLength + 2e-3, 0, 1, 3];

            S = sparameters(p, freq_sweep);

            s11_data = squeeze(S.Parameters(1,1,:));
            chunk_S11(j, :) = s11_data.';
            chunk_Success(j) = true;

            s11_db_vals = 20*log10(abs(s11_data));
            [min_val, min_idx] = min(s11_db_vals);

            chunk_S11min(j) = min_val;
            chunk_ResFreq(j) = freq_sweep(min_idx)/1e9;

            below_10 = s11_db_vals < -10;
            if sum(below_10) >= 2
                indices = find(below_10);
                chunk_BW(j) = (freq_sweep(indices(end)) - freq_sweep(indices(1))) / 1e6;
            end

        catch ME
            chunk_Success(j) = false;
            fprintf('  Sample %d FAILED: %s\n', chunk_start + j - 1, ME.message);
        end
    end

    % Copy chunk results to main arrays
    S11_Complex_Matrix(chunk_indices, :) = chunk_S11;
    Design_Parameters(chunk_indices, :) = chunk_Params;
    Success_Flags(chunk_indices) = chunk_Success;
    S11_min_dB(chunk_indices) = chunk_S11min;
    Res_Freq_GHz(chunk_indices) = chunk_ResFreq;
    Bandwidth_MHz(chunk_indices) = chunk_BW;

    chunk_time = toc(chunk_tic);
    completed_samples = chunk_end;
    elapsed_time_prev = elapsed_time_prev + chunk_time;

    % SAVE CHECKPOINT
    fprintf('  Chunk completed in %.1f min. Saving checkpoint...\n', chunk_time/60);
    save(checkpoint_mat, 'S11_Complex_Matrix', 'Design_Parameters', 'Success_Flags', ...
         'S11_min_dB', 'Res_Freq_GHz', 'Bandwidth_MHz', 'completed_samples', ...
         'elapsed_time_prev', 'freq_sweep', '-v7.3');

    n_done = sum(Success_Flags(1:chunk_end));
    fprintf('  CHECKPOINT SAVED: %d/%d samples complete (%.1f%%), %d successful\n', ...
        chunk_end, n_samples, chunk_end/n_samples*100, n_done);
    fprintf('  Total elapsed time: %.1f hours\n', elapsed_time_prev/3600);
end

elapsed_time = toc(total_tic) + elapsed_time_prev;
fprintf('-----------------------------------------------------------------\n');
fprintf('Simulation complete! Total time: %.1f hours\n\n', elapsed_time/3600);

%% 7. QC PLOTS (First 100)
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

%% 8. SAVE FINAL DATASET (MAT)
fprintf('Saving final dataset...\n');

valid_indices = Success_Flags;
Geometry = Design_Parameters(valid_indices, :);
S11_Complex = S11_Complex_Matrix(valid_indices, :);
geometry_columns = {'L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', 'h_mm', 'eps_r'};

save(output_mat, 'Geometry', 'S11_Complex', 'freq_sweep', 'geometry_columns', '-v7.3');
fprintf('  Dataset saved to %s\n', output_mat);

%% 9. GENERATE SUMMARY REPORT (CSV)
fprintf('Generating summary CSV...\n');

results_table = table((1:n_samples)', ...
    L_mm_all, W_mm_all, inset_mm_all, feedWidth_mm_all, h_mm_all, eps_r_all, ...
    S11_min_dB, Res_Freq_GHz, Bandwidth_MHz, Success_Flags, ...
    'VariableNames', {'Sample_ID', 'L_mm', 'W_mm', 'inset_mm', 'feedWidth_mm', ...
                      'h_mm', 'eps_r', 'S11_min_dB', 'Res_Freq_GHz', 'Bandwidth_MHz', 'Sim_Success'});

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

% Clean up checkpoint file after successful completion
if isfile(checkpoint_mat)
    delete(checkpoint_mat);
    fprintf('  Checkpoint file cleaned up.\n');
end

%% 10. FINAL STATISTICS
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
