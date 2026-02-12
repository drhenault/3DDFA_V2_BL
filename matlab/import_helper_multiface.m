close all;
% Multi-face version of import_helper_script.m
% This script plots data for all detected faces, not just face 0
% Each face is assigned a different color for easy distinction

% Set the directory where the .csv files reside
csv_file_directory = "../dumps/";

% Import data from CSV files
facecount = import_csv_data(csv_file_directory + "face_count.csv", "face_count");
faceposition = import_csv_data(csv_file_directory + "face_position.csv", "face_orientation");
mouthposition = import_csv_data(csv_file_directory + "mouth_position.csv", "mouth_position");

vadMichal = readtable('../dumps/vad.csv');
vadMichal.vadDagcDecFinal(vadMichal.seconds > 10 & vadMichal.seconds < 16.8) = 0;
vadMichal.vadDagcDecFinal(vadMichal.seconds > 25.8) = 0;

vadMarek = readtable('../dumps/vad.csv');
vadMarek.vadDagcDecFinal(vadMarek.seconds < 10) = 0;

vadTable = readtable('../dumps/vad.csv');


% Find all unique face indices in the data
unique_face_indices = unique(faceposition.face_idx);
num_faces = length(unique_face_indices);

fprintf('Found %d unique face indices: ', num_faces);
fprintf('%d ', unique_face_indices);
fprintf('\n');

% Define colors for different faces (up to 10 faces)
face_colors = [
    0.1216, 0.4667, 0.7059;  % Blue
    1.0000, 0.4980, 0.0549;  % Orange
    0.1725, 0.6275, 0.1725;  % Green
    0.8392, 0.1529, 0.1569;  % Red
    0.5804, 0.4039, 0.7412;  % Purple
    0.5490, 0.3373, 0.2941;  % Brown
    0.8902, 0.4667, 0.7608;  % Pink
    0.4980, 0.4980, 0.4980;  % Gray
    0.7373, 0.7412, 0.1333;  % Olive
    0.0902, 0.7451, 0.8118   % Cyan
];

% Set some properties of the plots
title_font_size = 20;
axis_description_font_size = 16;
tick_font_size = 13;
marker_size = 6;

% Create a figure with white background
figure('Color',[1 1 1]);

% ============================================================
% Subplot 1 (top-left): Face count on screen
% ============================================================
subplot(2,2,1);
plot(facecount.seconds, facecount.face_count);
title("Face count on screen", 'FontSize', title_font_size)
xlabel("time [s]", 'FontSize', axis_description_font_size);
ylabel("# of faces", 'FontSize', axis_description_font_size)
set(gca, 'FontSize', tick_font_size)
max_number_of_faces = max(facecount.face_count);
ylim([-0.1 0.1+max_number_of_faces])
grid on

% ============================================================
% Subplot 2 (top-right): Face proximity (z coordinate)
% ============================================================
subplot(2,2,2);
hold on;
legend_entries = {};
for face_idx = 1:num_faces
    current_face_id = unique_face_indices(face_idx);
    
    % Get mouth position for point type 63 for this face
    mouthposition_face = mouthposition((mouthposition.face_idx == current_face_id) & ...
                                       (mouthposition.point_type == 63), :);
    
    if ~isempty(mouthposition_face)
        color_idx = mod(face_idx - 1, size(face_colors, 1)) + 1;
        plot(mouthposition_face.seconds, 100./mouthposition_face.z, ...
             "LineStyle", "none", "Marker", ".", ...
             "MarkerSize", marker_size, ...
             "Color", face_colors(color_idx, :));
        legend_entries{end+1} = sprintf("Face %d", current_face_id);
    end
end

if num_faces == 1
    title(sprintf("Face %d proximity", unique_face_indices(1)), 'FontSize', title_font_size)
else
    title("Face proximity", 'FontSize', title_font_size)
    legend(legend_entries, 'Location', 'best', 'FontSize', 10)
end
xlabel("time [s]", 'FontSize', axis_description_font_size);
set(gca, 'FontSize', tick_font_size)
hold off;
grid on

% ============================================================
% Subplot 3 (bottom-left): Point 63 position (x and y)
% ============================================================
subplot(2,2,3);
hold on;
legend_entries = {};

for face_idx = 1:num_faces
    current_face_id = unique_face_indices(face_idx);
    
    % Get mouth position for point type 63 for this face
    mouthposition_face = mouthposition((mouthposition.face_idx == current_face_id) & ...
                                       (mouthposition.point_type == 63), :);
    
    if ~isempty(mouthposition_face)
        color_idx = mod(face_idx - 1, size(face_colors, 1)) + 1;
        base_color = face_colors(color_idx, :);
        
        % Plot X coordinate with base color
        plot(mouthposition_face.seconds, mouthposition_face.x, ...
             "LineStyle", "none", "Marker", ".", ...
             "MarkerSize", marker_size, ...
             "Color", base_color);
        
        % Plot Y coordinate with darker shade of the same color
        darker_color = base_color * 0.7;
        plot(mouthposition_face.seconds, mouthposition_face.y, ...
             "LineStyle", "none", "Marker", ".", ...
             "MarkerSize", marker_size, ...
             "Color", darker_color);
        
        if num_faces == 1
            legend_entries{end+1} = "x (horizontal)";
            legend_entries{end+1} = "y (vertical)";
        else
            legend_entries{end+1} = sprintf("Face %d x", current_face_id);
            legend_entries{end+1} = sprintf("Face %d y", current_face_id);
        end
    end
end

if num_faces == 1
    title(sprintf("Face %d point 63 position", unique_face_indices(1)), 'FontSize', title_font_size)
else
    title("Face point 63 position", 'FontSize', title_font_size)
end
xlabel("time [s]", 'FontSize', axis_description_font_size);
ylabel("coordinate value [px]", 'FontSize', axis_description_font_size)
set(gca, 'FontSize', tick_font_size)
legend(legend_entries, 'Location', 'northwest', 'FontSize', 10)
hold off;
grid on

% ============================================================
% Subplot 4 (bottom-right): Face roll angle
% ============================================================
subplot(2,2,4);
hold on;
legend_entries = {};

for face_idx = 1:num_faces
    current_face_id = unique_face_indices(face_idx);
    
    % Get face position for this face
    faceposition_face = faceposition(faceposition.face_idx == current_face_id, :);
    
    if ~isempty(faceposition_face)
        color_idx = mod(face_idx - 1, size(face_colors, 1)) + 1;
        plot(faceposition_face.seconds, faceposition_face.roll, ...
             "LineStyle", "none", "Marker", ".", ...
             "MarkerSize", marker_size, ...
             "Color", face_colors(color_idx, :));
        legend_entries{end+1} = sprintf("Face %d", current_face_id);
    end
end

if num_faces == 1
    title(sprintf("Face %d roll", unique_face_indices(1)), 'FontSize', title_font_size)
else
    title("Face roll", 'FontSize', title_font_size)
    legend(legend_entries, 'Location', 'best', 'FontSize', 10)
end
xlabel("time [s]", 'FontSize', axis_description_font_size);
ylabel("roll angle [deg]", 'FontSize', axis_description_font_size)
set(gca, 'FontSize', tick_font_size)
ylim([-90 90])
hold off;
grid on

% ============================================================
% Figure 2: Audio, VAD, and Face 0 Analysis
% ============================================================
figure(2)
[input, fs] = audioread('../inputs/demo_raw_v4_audio_video.wav');
subplot(5,1,1)
plot((0:length(input)-1)/fs, input)
title('Input audio')
xlabel('time [s]')
grid on

subplot(5,1,2)
%vadData = readmatrix('../dumps/vad.csv');
%vadData = vadData(:,2);
%stairs((0:length(vadData)-1)/30, vadData);
stairs(vadMichal.seconds, vadMichal.vadDagcDecFinal);
hold on
stairs(vadMarek.seconds, vadMarek.vadDagcDecFinal + 2);
ylim([-0.1 3.1])
title('VAD')
xlabel('time [s]')
grid on
legend('Michal', 'Marek')

% Calculate and plot Face 0 lip perimeter
subplot(5,1,3)
lipsFace0Data = mouthposition(mouthposition.face_idx == 0, :);
[perimeter0, time0] = calculate_lip_perimeter(lipsFace0Data);
if ~isempty(perimeter0)
    plot(time0, perimeter0);
    title('Face 0 Lip Perimeter')
    xlabel('time [s]')
    ylabel('perimeter [px]')
    grid on
end

% Calculate and plot Face 0 lip area
subplot(5,1,4)
[area0, time0_area] = calculate_lip_area(lipsFace0Data);
if ~isempty(area0)
    plot(time0_area, area0);
    title('Face 0 Lip Area')
    xlabel('time [s]')
    ylabel('area [px^2]')
    grid on
end

% Plot Face 0 lip positions
subplot(5,1,5)
lipsFace0UpperData = mouthposition((mouthposition.face_idx == 0) & ...
    (mouthposition.point_type == 66), :);
lipsFace0LowerData = mouthposition((mouthposition.face_idx == 0) & ...
    (mouthposition.point_type == 62), :);
plot(lipsFace0UpperData.seconds, lipsFace0UpperData.y);
hold on
plot(lipsFace0LowerData.seconds, lipsFace0LowerData.y);
legend('Upper Lip', 'Lower Lip', 'Location', 'best', 'FontSize', 10)
title('Face 0 Lip Positions')
xlabel('time [s]')
ylabel('y position [px]')
grid on

% ============================================================
% Figure 3: Face 1 Analysis and Comparison Metrics
% ============================================================
figure(3)

% Calculate and plot Face 1 lip perimeter
subplot(5,1,1)
lipsFace1Data = mouthposition(mouthposition.face_idx == 1, :);
[perimeter1, time1] = calculate_lip_perimeter(lipsFace1Data);
if ~isempty(perimeter1)
    plot(time1, perimeter1);
    title('Face 1 Lip Perimeter')
    xlabel('time [s]')
    ylabel('perimeter [px]')
    grid on
end

% Calculate and plot Face 1 lip area
subplot(5,1,2)
[area1, time1_area] = calculate_lip_area(lipsFace1Data);
if ~isempty(area1)
    plot(time1_area, area1);
    hold on;
    title('Face 1 Lip Area')
    xlabel('time [s]')
    ylabel('area [px^2]')
    grid on
end

% Plot Face 1 lip positions
subplot(5,1,3)
lipsFace1UpperData = mouthposition((mouthposition.face_idx == 1) & ...
    (mouthposition.point_type == 66), :);
lipsFace1LowerData = mouthposition((mouthposition.face_idx == 1) & ...
    (mouthposition.point_type == 62), :);    
plot(lipsFace1UpperData.seconds, lipsFace1UpperData.y);
hold on
plot(lipsFace1LowerData.seconds, lipsFace1LowerData.y);    
legend('Upper Lip', 'Lower Lip', 'Location', 'best', 'FontSize', 10)
title('Face 1 Lip Positions')
xlabel('time [s]')
ylabel('y position [px]')
grid on

% Plot lip delta for all faces, compute per-face variance and Visual VAD
subplot(5,1,4)
hold on;
n_times = length(vadTable.seconds);
variances = zeros(n_times, num_faces);
lips_delta_all = NaN(n_times, num_faces);  % per-face lip delta on VAD time grid
sliding_window_length = 30;  % frames (~1 s at 30 fps) - lip movement over short window

for face_idx = 1:num_faces
    current_face_id = unique_face_indices(face_idx);
    
    % Interpolate lip delta onto VAD time grid (mouthposition may have different sampling)
    upper_y = mouthposition((mouthposition.face_idx == current_face_id) & ...
        (mouthposition.point_type == 66), :);
    lower_y = mouthposition((mouthposition.face_idx == current_face_id) & ...
        (mouthposition.point_type == 62), :);
    if ~isempty(upper_y) && ~isempty(lower_y)
        lips_delta_face = upper_y.y - lower_y.y;
        t_face = upper_y.seconds;
        lips_delta_all(:, face_idx) = interp1(t_face, lips_delta_face, vadTable.seconds, 'linear', NaN);
    end
    
    % Sliding-window variance of lip delta (high when lips move, low when still)
    valid = ~isnan(lips_delta_all(:, face_idx));
    for time_index = sliding_window_length:n_times
        window = lips_delta_all((time_index-sliding_window_length+1):time_index, face_idx);
        if all(~isnan(window))
            variances(time_index, face_idx) = var(window);
        end
    end
    
    % Per-face scaling so variance ranges are comparable across faces (optional)
    scale_factors = [0.5^2, 2.5^2, 0.5^2, 2.5^2, 0.5^2, 2.5^2, 0.5^2, 2.5^2, 0.5^2, 2.5^2];
    scale_idx = min(face_idx, length(scale_factors));
    variances(:, face_idx) = variances(:, face_idx) * scale_factors(scale_idx);
    
    color_idx = mod(face_idx - 1, size(face_colors, 1)) + 1;
    plot(vadTable.seconds, lips_delta_all(:, face_idx), ...
        "LineStyle", "none", "Marker", ".", ...
        "MarkerSize", marker_size, ...
        "Color", face_colors(color_idx, :));
    legend_entries{end+1} = sprintf("Face %d", current_face_id);
end
legend(legend_entries, 'Location', 'best', 'FontSize', 10)
title('Lip Delta (Upper - Lower)')
xlabel('time [s]')
ylabel('delta [px]')
grid on
hold off;

% Plot of variance for both speakers
subplot(5,1,5);
for k = 1:num_faces
    plot(vadTable.seconds, variances(:, k));
    hold on
end
legend(legend_entries, 'Location', 'best', 'FontSize', 10)
title('Lip Delta Variance (sliding window)')
xlabel('time [s]')
ylabel('variance [px^2]')
hold off
set(gca, 'YScale', 'log');
grid on

% ============================================================
% Visual VAD: speaking detection from lip movement variance
% ============================================================
% Algorithm: threshold sliding-window variance of lip aperture (delta).
% When variance is above threshold -> lips are moving -> likely speaking.
% Uses adaptive threshold (median + k*MAD) so it works across different
% people and lighting without manual tuning.
% ============================================================
figure(4)
vad_win_sec = 0.5;   % smooth Visual VAD over this many seconds
vad_k_mad = 2.0;     % threshold = median + k*MAD (higher = fewer false "speaking")
vad_smooth_frames = max(1, round(vad_win_sec * 30));  % ~30 fps from VAD grid

% Audio VAD (reference)
if ismember('vadDagcDecFinal', vadTable.Properties.VariableNames)
    audio_vad = vadTable.vadDagcDecFinal;
else
    audio_vad = vadTable{:, 2};  % assume second column is VAD
end
t_vad = vadTable.seconds;

% Visual VAD per face: binary from variance threshold
visual_vad = zeros(n_times, num_faces);
for face_idx = 1:num_faces
    v = variances(:, face_idx);
    v_valid = v(v > 0 & isfinite(v));
    if isempty(v_valid)
        continue;
    end
    med = median(v_valid);
    mad_val = median(abs(v_valid - med));
    if mad_val < 1e-10
        mad_val = 1e-10;
    end
    thresh = med + vad_k_mad * mad_val;
    visual_vad(:, face_idx) = double(v > thresh);
    % Optional: smooth with moving average so brief spikes don't dominate
    visual_vad(:, face_idx) = movmean(visual_vad(:, face_idx), vad_smooth_frames);
end

% Plot: Audio VAD vs Visual VAD per face
subplot(num_faces + 1, 1, 1);
stairs(t_vad, audio_vad, 'k', 'LineWidth', 1);
ylim([-0.1 1.2])
title('Audio VAD (reference)')
xlabel('time [s]')
ylabel('speaking')
grid on

for face_idx = 1:num_faces
    subplot(num_faces + 1, 1, face_idx + 1);
    color_idx = mod(face_idx - 1, size(face_colors, 1)) + 1;
    stairs(t_vad, visual_vad(:, face_idx), 'Color', face_colors(color_idx, :), 'LineWidth', 1);
    ylim([-0.1 1.2])
    title(sprintf('Visual VAD (Face %d) — lip variance threshold', unique_face_indices(face_idx)))
    xlabel('time [s]')
    ylabel('speaking')
    grid on
end
sgtitle('Speaking detection: Audio VAD vs Visual VAD (lip movement variance)', 'FontSize', 12)

% Plot perimeter^2/area ratio for both faces (compactness measure)
% subplot(5,1,5)
% hold on;
% ratio_legend_entries = {};
% 
% % Face 0 ratio
% if ~isempty(perimeter0) && ~isempty(area0)
%     ratio0 = (perimeter0.^2) ./ area0;
%     plot(time0, ratio0, 'Color', face_colors(1, :), 'LineWidth', 1.5);
%     ratio_legend_entries{end+1} = 'Face 0';
% end
% 
% % Face 1 ratio
% if ~isempty(perimeter1) && ~isempty(area1)
%     ratio1 = (perimeter1.^2) ./ area1;
%     plot(time1, ratio1, 'Color', face_colors(2, :), 'LineWidth', 1.5);
%     ratio_legend_entries{end+1} = 'Face 1';
% end
% 
% if ~isempty(ratio_legend_entries)
%     legend(ratio_legend_entries, 'Location', 'best', 'FontSize', 10)
% end
% set(gca, 'Yscale', 'log');
% title('Perimeter^2/Area Ratio (Shape Compactness)')
% xlabel('time [s]')
% ylabel('P^2/A')
% grid on
% hold off;

% % Print statistics to console
% fprintf('\n========================================\n');
% fprintf('STATISTICS\n');
% fprintf('========================================\n');
% fprintf('Total frames: %d\n', height(facecount));
% fprintf('Frames with faces: %d (%.1f%%)\n', ...
%     sum(facecount.face_count > 0), ...
%     100 * sum(facecount.face_count > 0) / height(facecount));
% fprintf('Frames with multiple faces: %d (%.1f%%)\n', ...
%     sum(facecount.face_count > 1), ...
%     100 * sum(facecount.face_count > 1) / height(facecount));
% fprintf('\n');
% 
% fprintf('Frames per face:\n');
% for face_idx = 1:num_faces
%     current_face_id = unique_face_indices(face_idx);
%     face_frames = sum(faceposition.face_idx == current_face_id);
%     fprintf('  Face %d: %d frames (%.1f%%)\n', ...
%         current_face_id, face_frames, ...
%         100 * face_frames / height(facecount));
% end
% fprintf('\n');
% 
% fprintf('Average orientation per face:\n');
% for face_idx = 1:num_faces
%     current_face_id = unique_face_indices(face_idx);
%     faceposition_face = faceposition(faceposition.face_idx == current_face_id, :);
%     
%     avg_roll = mean(faceposition_face.roll);
%     avg_pitch = mean(faceposition_face.pitch);
%     avg_yaw = mean(faceposition_face.yaw);
%     
%     fprintf('  Face %d: roll=%.1f°, pitch=%.1f°, yaw=%.1f°\n', ...
%         current_face_id, avg_roll, avg_pitch, avg_yaw);
% end
% fprintf('========================================\n\n');

%% Helper function to import CSV data
function imported_data = import_csv_data(csv_file_path, csv_file_type)
%import_csv_data A helper function to assist with importing data from CSV
%files
% csv_file_path: the location of the csv file in the system
% csv_file_type: a string, the following values are supported:
%   "face_count": a file with face count information
%   "face_orientation: a file with roll, pitch, and yaw information
%   "mouth_position": a file with information about position and proximity of
%       various points of interest

    % Set up column names and their types, depending from the type of the CSV file
    
    switch csv_file_type
        case "face_count"
           variable_names = ["seconds", "face_count"];
        case "face_orientation"
           variable_names = ["seconds", "face_idx", "roll", "pitch", "yaw"];
        case "mouth_position"
           variable_names = ["seconds", "face_idx", "point_type", "x", "y", "z"];
    end
    
    % Specify the number of imported variables and their type
    number_of_imported_variables = size(variable_names, 2);
    opts = delimitedTextImportOptions("NumVariables", number_of_imported_variables);
    opts.VariableTypes = repmat("double", 1, number_of_imported_variables);
    opts.VariableNames = variable_names;
    
    
    % Specify range and delimiter
    opts.DataLines = [2, Inf]; % data line 1 is a line of headers - don't import them
    opts.Delimiter = ","; % values are separated with a comma
    
    % Specify file level properties
    opts.ExtraColumnsRule = "ignore"; % ignore any extra columns
    opts.EmptyLineRule = "read"; % read all empty lines
    
    % Import the data
    imported_data = readtable(csv_file_path, opts);

end

%% Helper function to calculate lip perimeter
function [perimeter, time] = calculate_lip_perimeter(face_data)
%calculate_lip_perimeter Calculates the perimeter of the lip contour (points 60-67)
% face_data: table containing mouth position data for a specific face
% Returns:
%   perimeter: array of perimeter lengths at each timestamp
%   time: array of timestamps corresponding to each perimeter value

    % Filter for points 60-67
    lip_contour_data = face_data(face_data.point_type >= 60 & face_data.point_type <= 67, :);
    
    if isempty(lip_contour_data)
        perimeter = [];
        time = [];
        return;
    end
    
    % Get unique timestamps
    unique_times = unique(lip_contour_data.seconds);
    perimeter = zeros(length(unique_times), 1);
    time = unique_times;
    
    % Calculate perimeter for each timestamp
    for i = 1:length(unique_times)
        t = unique_times(i);
        
        % Get all points at this timestamp
        frame_data = lip_contour_data(lip_contour_data.seconds == t, :);
        
        % Sort by point_type to ensure correct order (60, 61, 62, ..., 67)
        frame_data = sortrows(frame_data, 'point_type');
        
        % Check if we have all 8 points
        if height(frame_data) == 8
            % Calculate distances between consecutive points
            total_distance = 0;
            for j = 1:7
                dx = frame_data.x(j+1) - frame_data.x(j);
                dy = frame_data.y(j+1) - frame_data.y(j);
                total_distance = total_distance + sqrt(dx^2 + dy^2);
            end
            
            % Close the contour: distance from point 67 back to point 60
            dx = frame_data.x(1) - frame_data.x(8);
            dy = frame_data.y(1) - frame_data.y(8);
            total_distance = total_distance + sqrt(dx^2 + dy^2);
            
            perimeter(i) = total_distance;
        else
            % If we don't have all points, set perimeter to NaN
            perimeter(i) = NaN;
        end
    end
end

%% Helper function to calculate lip area
function [area, time] = calculate_lip_area(face_data)
%calculate_lip_area Calculates the area inside the lip contour (points 60-67)
% Uses the Shoelace formula (also known as the surveyor's formula)
% face_data: table containing mouth position data for a specific face
% Returns:
%   area: array of areas at each timestamp
%   time: array of timestamps corresponding to each area value

    % Filter for points 60-67
    lip_contour_data = face_data(face_data.point_type >= 60 & face_data.point_type <= 67, :);
    
    if isempty(lip_contour_data)
        area = [];
        time = [];
        return;
    end
    
    % Get unique timestamps
    unique_times = unique(lip_contour_data.seconds);
    area = zeros(length(unique_times), 1);
    time = unique_times;
    
    % Calculate area for each timestamp using Shoelace formula
    for i = 1:length(unique_times)
        t = unique_times(i);
        
        % Get all points at this timestamp
        frame_data = lip_contour_data(lip_contour_data.seconds == t, :);
        
        % Sort by point_type to ensure correct order (60, 61, 62, ..., 67)
        frame_data = sortrows(frame_data, 'point_type');
        
        % Check if we have all 8 points
        if height(frame_data) == 8
            % Shoelace formula: A = 0.5 * |Σ(x_i * y_(i+1) - x_(i+1) * y_i)|
            sum_term = 0;
            for j = 1:7
                sum_term = sum_term + (frame_data.x(j) * frame_data.y(j+1) - ...
                                       frame_data.x(j+1) * frame_data.y(j));
            end
            
            % Close the polygon: from point 67 back to point 60
            sum_term = sum_term + (frame_data.x(8) * frame_data.y(1) - ...
                                   frame_data.x(1) * frame_data.y(8));
            
            area(i) = abs(sum_term) / 2;
        else
            % If we don't have all points, set area to NaN
            area(i) = NaN;
        end
    end
end
