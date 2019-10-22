function BER = checkBER(true_bits, test_bits)
% function BER = checkBER(true_bits, test_bits)
% 
% This function calculates the BER of a received string of bits as well as
% aligning a received stream with the original bit stream.
%
% Inputs
%   true_bits - the original transmitted bits
%   test_bits - the received bits
%
% Output
%   BER - bit error rate
%
%

true_bits = 2*true_bits-1;
test_bits = 2*test_bits-1;

fft_true = fft(true_bits,length(true_bits));
fft_test = fft(test_bits,length(true_bits));

ifftIn = fft_test.*conj(fft_true);
corrOut = ifft(ifftIn);

peak = max(corrOut);
N_wrong = (length(test_bits) - peak) / 2;

BER = N_wrong / length(test_bits);
end

