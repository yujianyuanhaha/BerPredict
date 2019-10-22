function [y,y1_new,y2_new,x1_new,x2_new]= NotchFilterforWindow(x, r, y1_old, y2_old, x1_old, x2_old, idx, eps, fc)
%X = fft(x);
% [eng,ind]=max(abs(X).^2);
% fc = ind/length(x);
% fc = 0.0025;
% Determine the filter coefficients
w = 2*pi*fc;

r2 = 1 + eps;

a(1) = r2^(-2);             
a(2) = -2*cos(w)*r2^(-2);   
a(3) = r2^(-2);              
b(2) = -2*cos(w)*r2^(-1);    
b(3) = r2^(-2);              

n=length(x);
y = zeros(1,n);

if idx ==1

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

else
    x1 = x1_old;
    x2 = x2_old;
    y1 = y1_old;
    y2 = y2_old;
    
end

if idx ==1
    
for k = 3 : n
    y(k) = a(1) * x(k) + a(2) * x(k-1) + a(3) * x(k-2) -  b(2) * y(k-1) - b(3) * y(k-2);
end

else
    y(1) = a(1) * x(1) + a(2) * x2 + a(3) * x1 -  b(2) * y2 - b(3) * y1;
    y(2) = a(1) * x(2) + a(2) * x(1) + a(3) * x2 -  b(2) * y(1) - b(3) * y2;
 for k = 3 : n
    y(k) = a(1) * x(k) + a(2) * x(k-1) + a(3) * x(k-2) -  b(2) * y(k-1) - b(3) * y(k-2);
end   
end
 y1_new = y(end-1);
 y2_new = y(end);
 x1_new = x(end-1);
 x2_new = x(end);
 end