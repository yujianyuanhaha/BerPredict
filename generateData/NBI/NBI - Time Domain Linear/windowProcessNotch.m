function rClean = windowProcessNotch(r,PoleRadius,windowSize)
N = length(r);
temp_tail = rem(length(r),windowSize);
if temp_tail>0
    r = [r,zeros(1,windowSize-temp_tail)];
end
windowNum = length(r)/windowSize;
rClean = [];
w_old = 0;
v_old = 0;
NumErrors = [];


y1_old = 0 ;
y2_old = 0 ;
x1_old = 0 ;
x2_old = 0 ;

if windowSize >= 4096
    eps = 5 * 1e-3;
else
    eps = (1e-2);
end

fc = 0;
r_memory = [];
noI = 0;
isI = 0;

for idxW = 1:windowNum
    temp_idx = (idxW-1) * windowSize+1;
    x_end1 = r(temp_idx:temp_idx+windowSize-1);
    
    if windowSize <= 2048
        m_time = 5;
        m_size = windowSize * m_time;
    else
        m_time = 4;
        m_size = windowSize * m_time;
    end
    
    if idxW <= m_time
        r_memory = [r_memory, x_end1];
        temp_x = r_memory;
        temp_X = fft(temp_x);
        X_power = abs(temp_X).^2;
        [eng,ind]=max(X_power);
        temp_fc = ind/length(temp_x);              
        numP = sum(X_power>(eng/2));
        if numP >=2
            noI = noI+1;
        else
            isI = isI+1;
        end
        fc = temp_fc;
    else
        r_memory = [r_memory, x_end1];
        r_memory = r_memory(end-m_size:end);
        temp_x = r_memory;
        temp_X = fft(temp_x);
        X_power = abs(temp_X).^2;
        [eng,ind]=max(X_power);
        temp_fc = ind/length(temp_x); 
        numP = sum(X_power>(eng/2));
        if idxW<=6           
            if numP >=2
                noI = noI+1;
            else
                isI = isI+1;
            end
            fc = temp_fc;
        else
            if isI >= noI
                fc = temp_fc;
            else
                fc = 0.5;
            end
            
        end
    end


    [rClean_temp,y1_new,y2_new,x1_new,x2_new]= NotchFilterforWindow(x_end1,PoleRadius, y1_old, y2_old, x1_old, x2_old, temp_idx, eps, fc);
    y1_old = y1_new ;
    y2_old = y2_new ;
    x1_old = x1_new ;
    x2_old = x2_new ;
    rClean = [rClean,rClean_temp];
end  
rClean = rClean(1:N);
end