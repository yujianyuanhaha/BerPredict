function s= myPulseShape2(s,sps,span,beta,shape)


% sps   = 4;
% 
% span  = 4;
% beta  = 0.25;
% shape = 'sqrt';
%p                 = rcosdesign(beta,span,sps,shape);
%rCosSpec          =  fdesign.pulseshaping(sps,'Raised Cosine',...
    'Nsym,Beta',span,0.25);
%rCosFlt           = design ( rCosSpec );
%rCosFlt.Numerator = rCosFlt.Numerator / max(rCosFlt.Numerator);

%upsampled = upsample( s, sps);     % upsample
%FltDelay  = (span*sps)/2;           % shift
%temp      = filter(rCosFlt , [ upsampled , zeros(1,FltDelay) ] );
%s        = temp(sps*span/2+1:end);        % to be fixed


% hardcore
p = [-7.65404249467096e-18,-0.0267646476244517,-0.0464045408764016,-0.0410411442509746,9.18807912334947e-18,0.0725901244455568,0.156842660715307,0.224258283325397,0.250000000000000,0.224258283325397,0.156842660715307,0.0725901244455568,9.18807912334947e-18,-0.0410411442509746,-0.0464045408764016,-0.0267646476244517,-7.65404249467096e-18];

% K * N, N over time
[k1,k2] = size(x);
s = zeros(k1,k2*sps+length(p)-1);
for i = 1:k1
   temp = upsample(x(i,:), sps);
   s(i,:) = conv(temp,p);
end


end