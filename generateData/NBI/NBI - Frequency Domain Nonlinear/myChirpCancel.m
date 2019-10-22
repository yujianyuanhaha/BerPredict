function r_clean = myChirpCancel(r)
% function r_clean = myChirpCancel(r)
%
% This function performs interfernce cancellation assuming that the
% interfering signal is a chirp.  It relies on the use of the fractional
% Fourier Transform
%
% Inputs
%   r - complex samples of input signal with SOI contaminated with chirp
%
% Outputs
%   r_clean - complex samples of signal after chirp cancellation / mitigation

maxVal = 0;
max_a = 0;
for a = 0:0.005:2
    X = fracF(r,a);
    Z = max(abs(X));
    if Z > maxVal
        maxVal = Z;
        max_a = a;
    end
end


% cancel interference
X = fracF(r,max_a);
% figure;plot(abs(X));
[chirp_val, chirp_loc] = max(abs(X));
level = mean([mean(abs(X(1:chirp_loc-1))),mean(abs(X(chirp_loc+1:end)))]);

% X(chirp_loc) = 0;% level*X(chirp_loc)/abs(chirp_loc);
% X(chirp_loc-10:chirp_loc+10) = 0;
    xx = find(abs(X)>10*level);
    X(xx)= level;
    X(xx) = 0;

r_clean = fracF(X,-max_a); % inverse
r_clean = r_clean.';


end