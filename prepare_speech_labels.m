clear; clc; close all;

%% Settings 
% audioPath = 'inputs/';
% audioName = 'example_3.wav';
% singleTalk = [2.49 13.14; 25.19 37.79];
% doubleTalk = [13.22 25.01];

% audioPath = 'inputs/';
% audioName = 'example_11.wav';
% singleTalk = [0.52 24.80];
% doubleTalk = [];

audioPath = 'inputs/';
audioName = 'example_12.wav';
singleTalk = [1.28 18.02];
doubleTalk = [];

%% Processing
fsTarget = 48e3;
[audio, fs] = audioread([audioPath audioName]);
[P, Q] = rat(fsTarget/fs);
audio = resample(audio, P, Q);

vad = zeros(length(repelem(audio, P, Q)), 1);
if ~isempty(singleTalk)
    for singleIdx = 1:size(singleTalk, 1)
        vad(round(singleTalk(singleIdx, 1)*fsTarget):round(singleTalk(singleIdx, 2)*fsTarget)) = 1;
    end
end
if ~isempty(doubleTalk)
    for doubleIdx = 1:size(doubleTalk, 1)
        vad(round(doubleTalk(doubleIdx, 1)*fsTarget):round(doubleTalk(doubleIdx, 2)*fsTarget)) = 2;
    end
end
save([audioPath audioName(1:end-4) '_manualVAD.mat'], 'vad');

%% Plotting
fontSize = 18;
figure(); hold on;
set(gcf, 'Position', [50 50 1800 600]);
set(gca, 'FontSize', fontSize - 2);
t = (0:length(audio)-1) / fsTarget;   % time axis in seconds
fill([-2 -1 -1 -2], [0 0 1 1], [0.0 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.25);
fill([-2 -1 -1 -2], [0 0 1 1], [1.0 0.70 0.30], 'EdgeColor', 'none', 'FaceAlpha', 0.35);
yBot = -1.05 * max(abs(audio));
yTop =  1.05 * max(abs(audio));
for k = 1:size(singleTalk, 1)
    x1 = singleTalk(k, 1);
    x2 = singleTalk(k, 2);
    fill([x1 x2 x2 x1], [yBot yBot yTop yTop], [0.0 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.25);
end
for k = 1:size(doubleTalk, 1)
    x1 = doubleTalk(k, 1);
    x2 = doubleTalk(k, 2);
    fill([x1 x2 x2 x1], [yBot yBot yTop yTop], [1.0 0.70 0.30], 'EdgeColor', 'none', 'FaceAlpha', 0.35);
end
plot(t, audio, 'k');
xlim([0 t(end)]);
ylim([yBot yTop]);
xlabel('Time [s]', 'FontSize', fontSize); 
ylabel('Amplitude', 'FontSize', fontSize);
title('Audio waveform with VAD labels', 'FontSize', fontSize + 4);
legend('\bfSingle speaker', '\bfDouble talk', 'Location', 'NorthEast', 'FontSize', fontSize);
