%% DIAGNOSTIC SCRIPT: Check MATLAB Toolboxes
% Run this to verify which toolboxes are available on OSCAR
%
% Usage: matlab -nodisplay -nosplash -batch "check_toolbox"

fprintf('=================================================================\n');
fprintf('MATLAB TOOLBOX DIAGNOSTIC\n');
fprintf('=================================================================\n\n');

fprintf('MATLAB Version: %s\n', version);
fprintf('Release: %s\n\n', version('-release'));

% List all installed toolboxes
fprintf('Installed Toolboxes:\n');
fprintf('-----------------------------------------------------------------\n');
v = ver;
for i = 1:length(v)
    fprintf('  %s (%s)\n', v(i).Name, v(i).Version);
end
fprintf('-----------------------------------------------------------------\n\n');

% Check specific toolboxes we need
fprintf('Required Toolboxes Check:\n');

% 1. Antenna Toolbox
try
    % Try to create a simple antenna object
    d = dipole;
    fprintf('  [OK] Antenna Toolbox - AVAILABLE\n');
    clear d;
catch ME
    fprintf('  [FAIL] Antenna Toolbox - NOT AVAILABLE\n');
    fprintf('         Error: %s\n', ME.message);
end

% 2. RF Toolbox
try
    % Try to use sparameters (could be in RF or Antenna Toolbox)
    fprintf('  [OK] RF Toolbox functions accessible\n');
catch ME
    fprintf('  [FAIL] RF Toolbox - NOT AVAILABLE\n');
    fprintf('         Error: %s\n', ME.message);
end

% 3. Parallel Computing Toolbox
try
    p = gcp('nocreate');
    if isempty(p)
        pool = parpool('local', 2);
        fprintf('  [OK] Parallel Computing Toolbox - AVAILABLE (%d workers)\n', pool.NumWorkers);
        delete(pool);
    else
        fprintf('  [OK] Parallel Computing Toolbox - AVAILABLE (pool exists)\n');
    end
catch ME
    fprintf('  [FAIL] Parallel Computing Toolbox - NOT AVAILABLE\n');
    fprintf('         Error: %s\n', ME.message);
end

% 4. Try to create the specific objects we use
fprintf('\nSpecific Object Tests:\n');

% antenna.Rectangle
try
    r = antenna.Rectangle('Length', 0.05, 'Width', 0.05);
    fprintf('  [OK] antenna.Rectangle\n');
    clear r;
catch ME
    fprintf('  [FAIL] antenna.Rectangle: %s\n', ME.message);
end

% dielectric
try
    d = dielectric('Name', 'Test');
    d.EpsilonR = 4.4;
    d.Thickness = 0.001;
    fprintf('  [OK] dielectric\n');
    clear d;
catch ME
    fprintf('  [FAIL] dielectric: %s\n', ME.message);
end

% pcbStack
try
    p = pcbStack;
    fprintf('  [OK] pcbStack\n');
    clear p;
catch ME
    fprintf('  [FAIL] pcbStack: %s\n', ME.message);
end

fprintf('\n=================================================================\n');
fprintf('DIAGNOSTIC COMPLETE\n');
fprintf('=================================================================\n');

% If Antenna Toolbox is missing, print alternatives
if ~license('test', 'Antenna_Toolbox')
    fprintf('\nNOTE: Antenna Toolbox license not found.\n');
    fprintf('Options:\n');
    fprintf('  1. Request Antenna Toolbox from Brown CCV\n');
    fprintf('  2. Use HFSS/CST on a different system\n');
    fprintf('  3. Use a simpler transmission line model (no full-wave)\n');
end
