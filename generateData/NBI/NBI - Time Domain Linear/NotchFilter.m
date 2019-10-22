function y = NotchFilter(x, r)
% function y = NotchFilter(x,r)
%
% Inputs: x - N x 1 complex vector which contains a wideband signal
% contaminated by narrowband interference.
%         r - pole radius.  This should be close to 1.  The closer to 1 the
%         more narrow and deeper the notch.  Please see data sheet for
%         tradeoffs in choosing r.
% 
% Determine the notch frequency


X = fft(x);
X_power = abs(X).^2;
[eng,ind]=max(X_power);

numP = sum(X_power>(eng/2));

if numP >=2
    fc = 0.5;
else
    fc = ind/length(x);
end
% Determine the filter coefficients
w = 2*pi*fc;
eps = 5 * 1e-3;
r2 = 1 + eps;

a(1) = r2^(-2);              % a0
a(2) = -2*cos(w)*r2^(-2);    % a1
a(3) = r2^(-2);              % a2
b(2) = -2*cos(w)*r2^(-1);    % b1
b(3) = r2^(-2);              % b2

n=length(x);
y = zeros(1,n);

k2 = 1-r^2;
k1 = sqrt(1+r^2-2*r*cos(2*pi*fc));
w = 0;
v = 0;
 for i=1:2
     w_old = w;
     v_old = v;
     
     v = v_old + k1*w_old;
     w = w_old -(k1*v+k2*x(i)+k2*w_old);
     x_p = 0.5*(w+w_old);
     
     y(i) = x(i) + x_p;
         
 end
 
for k = 3 : n
    y(k) = a(1) * x(k) + a(2) * x(k-1) + a(3) * x(k-2) -  b(2) * y(k-1) - b(3) * y(k-2);
end

     
 end

