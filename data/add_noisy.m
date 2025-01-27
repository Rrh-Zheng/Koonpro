data_name = 'cup';
target_snr_db = 20; 
filename = strcat(data_name, '.csv'); 
data = readmatrix(filename); 
first_row = data(1, :); 
data = data(2:end, :);
first_column = data(:, 1); 
data = data(:, 2:end); 

[m, ~] = size(data); 
signal = data; 
signal_power = mean(signal(:).^2);

target_snr = 10^(target_snr_db / 10); 
noise_power = signal_power / target_snr; 
noise_std = sqrt(noise_power);

noise = noise_std * randn(size(signal)); 
noisy_signal = signal + noise;

data_noisy_with_first_column = [first_column, noisy_signal];


data_final = [first_row; data_noisy_with_first_column];

save_path = strcat('noisy_', num2str(target_snr_db), 'dB');
if ~exist(save_path, 'dir')
    mkdir(save_path);
end

output_filename = strcat(save_path, '\', data_name, '.csv');
writematrix(data_final, output_filename); % 保存为 CSV
fprintf('Noise data saved as "%s"\n', output_filename);