clear all;


%[po, f_fo,offset, bitsPerSym, sps, JtoS, EbNo, IntfType,duty ];
load('XS1.mat');
load('YS1.mat');

[len,~] = size(X);

idx =  find( X(:,8)==1 );
X_n = X( idx,:);
Y_n = Y( idx,:);


save('XS1_n.mat','X_n');
save('YS1_n.mat','Y_n')

