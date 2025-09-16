% Assume LocalizationsFinal, Frame_Information, Pre_A, Resolution already exist

Pre_A = 100;
Resolution = 70;


X_overall = [];

D_maxf = 0;
D_maxm = Inf;

for i = 1:length(LocalizationsFinal)
    D = pdist(LocalizationsFinal{i});
    
    D_max = max(D);
    if D_max > D_maxf
        D_maxf = D_max;
    end
    
    if D_max < D_maxm
        D_maxm = D_max;
    end
end

bins = [0:Resolution:D_maxf, Inf];

Total_Blink = [];
Total_No_Blink = [];

disp('Working on step 1')

for i = 1:length(LocalizationsFinal)
    Z2 = pdist([[1:length(Frame_Information{i})]'*0, Frame_Information{i}(:)]);
    D = pdist(LocalizationsFinal{i});
    
    D_Blink = D(Z2 < Pre_A);
    D_No_Blink = D((Z2 > Pre_A) & (Z2 < Pre_A*5));
    
    D_Counts = histcounts(D_Blink, bins, 'Normalization', 'prob');
    Total_Blink = [Total_Blink; D_Counts];
    
    D_Counts2 = histcounts(D_No_Blink, bins, 'Normalization', 'prob');
    Total_No_Blink = [Total_No_Blink; D_Counts2];
end

writematrix(bins, 'bins.csv');
writematrix(Total_Blink, 'Total_Blink.csv');
writematrix(Total_No_Blink, 'Total_No_Blink.csv');

%%
%Here we go through and scale the probablity distributions after 10 times
%the resolution of the experiment.


D_Counts=mean(Total_Blink,1);%histcounts(Total_Blink,bins,'Normalization','prob');

D_Counts2=mean(Total_No_Blink,1);%histcounts(Total_No_Blink,bins,'Normalization','prob');

%kk=find(D_Counts<D_Counts2,1);

D_Scale=sum(D_Counts(10:end))/sum(D_Counts2(10:end));

D_Counts3=D_Counts-D_Counts2*D_Scale;

D_Counts3=(D_Counts3)/sum(D_Counts3);




%Finally we clean up the distributions a little bit so that the logic is
%consistent with how the distribution should behave
good=1;
ins=0;
for i=4:length(D_Counts3)-1
    if D_Counts3(i)>0 && D_Counts3(i+1)<D_Counts3(i) && good==1
        
    else
        if ins==0
            ins=i;
        end
        good=0;
        D_Counts3(i)=0;
    end
end

D_Counts3(D_Counts3<0)=0;
D_Counts3=(D_Counts3)/sum(D_Counts3);
D_Counts3=(D_Counts3)/sum(D_Counts3);


%Anything greater than 8 times the resolution of the experiment is
%considered noise and will be eliminated from the probability distribution.
%We then warn the user if this is the case as they may want to consider a
%lower resolution.

if sum(D_Counts3(8:end)>0)>1
    disp('Warning: Eliminating Noise for higher bins')
    D_Counts3(8:end)=0;
    D_Counts3=(D_Counts3)/sum(D_Counts3);
end


% ---- Export new results for Python comparison ----
writematrix(D_Counts, 'D_Counts.csv');
writematrix(D_Counts2, 'D_Counts2.csv');
writematrix(D_Counts3, 'D_Counts3.csv');

%%CHUNK3%%
Distribution_for_Blink2=D_Counts3;


%Here we do the fitting to determine how much of each distribution makes up
%the pairwise distance distributions at each frame difference. This is
%equation 3 of the Supporting Material.

Dscale_store={};
for oinw=1:length(LocalizationsFinal)
    Dscale_store{oinw}=[];
end



disp('Still Working on step 1, wait for me')

parfor i=1:length(LocalizationsFinal)
    %if length(LocalizationsFinal{i})<9000
        disp('Still Working on step 1, wait for me')
        %disp(['Percent Done=',num2str(i/length(LocalizationsFinal)),' with STEP 1'])
        
        %Here we will gather the distance distrabutions from as many cells as
        %posible. This will allow us to define the joint probability
        %distribution more accuratly.
        
        Z2 = pdist([[1:length(Frame_Information{i})]'*0,Frame_Information{i}(:)]);
        
        D = (pdist(LocalizationsFinal{i}));
        
        D_No_Blink=D([(Z2>Pre_A)]==1);
        
        D_No_Blink=D([(Z2>Pre_A).*(Pre_A*5>Z2)]==1);
        
        True_Distribuiton2=histcounts(D_No_Blink,bins,'Normalization','prob');
        
        
        for w=1:Pre_A
            
            D_Blink=D(Z2==w);
            
            Temp_Distribution=((histcounts(D_Blink,bins,'Normalization','prob')));
            
            y=Temp_Distribution;
            
            t=[True_Distribuiton2; Distribution_for_Blink2];
            
            F=@(x,xdata) x.*xdata(1,:)+(1-x).*xdata(2,:);
            opts = optimset('Display','off');
            x0=1;
            
            [x,~,~,~,~] = lsqcurvefit(F,x0,t,y,[],[],opts);
            
            D_Scale=x;
            
            
            Dscale_store{i}(w)=D_Scale;
            if i==1 && w<=3   % save first 3 w values for debugging
                debug_filename = sprintf('debug_distributions_img%d_w%d.csv', i, w);
                debug_data = [Temp_Distribution(:), True_Distribuiton2(:), Distribution_for_Blink2(:)];
                writematrix(debug_data, debug_filename);
            end
        end
    %end
end

size(Dscale_store)



Dscale_store2=[];
for oinw=1:length(LocalizationsFinal)
    Dscale_store2(oinw,:)=Dscale_store{oinw};
end




if length(LocalizationsFinal)>1
    X_overall=mean(Dscale_store2);
    
    Dscale_store=mean(Dscale_store2);
else
    X_overall=(Dscale_store2);
    
    Dscale_store=(Dscale_store2);
end

writematrix(Dscale_store2, 'Dscale_store2_real.csv');



%Here we are going to determine the matrix M, it is better to do it over
%all of the images as there is less error if you have a lower number of
%localizations in one image.

Dscale_store(Dscale_store>1)=1;

Dscale_store(Dscale_store<0)=.0000001;

Dscale_store(end)=1;

Deviation_in_Probabilityt={};

for oinw=1:length(LocalizationsFinal)
    Deviation_in_Probabilityt{oinw}=[];
end

True_Distribuiton_all = [] 
disp('Still going, Working on step 1')
parfor i=1:length(LocalizationsFinal)
    disp('Still going, Working on step 1')
    %disp(['Percent Done=',num2str(i/length(LocalizationsFinal)),' with STEP 1'])
    
    %Here we will gather the distance distrabutions from as many cells as
    %posible. This will allow us to define the joint probability
    %distribution more accuratly.
    
    Z2 = pdist([[1:length(Frame_Information{i})]'*0,Frame_Information{i}(:)]);
    
    D = (pdist(LocalizationsFinal{i}));
    
    D_No_Blink=D([(Z2>Pre_A)]==1);
    D_No_Blink=D([(Z2>Pre_A).*(Pre_A*5>Z2)]==1);
    True_Distribuiton=histcounts(D_No_Blink,bins,'Normalization','prob');
    
    True_Distribuiton_all(i,:) = True_Distribuiton;
    for w=1:Pre_A
        
        %D_Blink=D(Z2==w);
        
        %  Temp_Distribution=((histcounts(D_Blink,bins,'Normalization','prob')));
        
        D_Scale=Dscale_store(w);
        
        Temp_Distribution2=Distribution_for_Blink2*(1-D_Scale)+True_Distribuiton*(D_Scale);
        
        %See the equation in determining the probability section of the paper.
        Combined=(Temp_Distribution2-D_Scale*True_Distribuiton)./(Temp_Distribution2);
        
        Deviation_in_Probabilityt{i}(w,:)=Combined;
        
    end
end

% Save to CSV after loop
writematrix(True_Distribuiton_all, 'True_Distribuiton_mat.csv');

all_dev = [];
row_index = [];   % to store (image, w) mapping
for i = 1:length(Deviation_in_Probabilityt)
    dev = Deviation_in_Probabilityt{i};
    all_dev = [all_dev; dev];   % Pre_A × bins
    for w = 1:size(dev,1)
        row_index = [row_index; i, w];
    end
end
writematrix(all_dev, 'Deviation_in_Probabilityt_mat.csv');
writematrix(row_index, 'Deviation_in_Probabilityt_index_mat.csv');




%Here we are going to go through and calculate M
if length(LocalizationsFinal)>1
    M_mat=[];
    for w=1:Pre_A
        M_mat_t=[];
        for i=1:length(Deviation_in_Probabilityt)
            M_mat_t(i,:)=Deviation_in_Probabilityt{i}(w,:);
        end
        
        M_mat(w,:)=mean(M_mat_t);
        
    end
else
    M_mat=Deviation_in_Probabilityt{1};
end
%%
%We dont use this, but just in case you want to mess areound
M_mat2=[];
for w=1:Pre_A
    M_mat_t=[];
    for i=1:length(Deviation_in_Probabilityt)
        M_mat_t(i,:)=Deviation_in_Probabilityt{i}(w,:);
    end
    M_mat2(w,:)=median(M_mat_t);
end
hey=1;


% ---- Export for Python comparison ----
writematrix(Dscale_store, 'Dscale_store2.csv');
writematrix(X_overall, 'X_overall.csv');
writecell(Deviation_in_Probabilityt, 'Deviation_mat.csv')
writematrix(M_mat, 'M_mat.csv');
writematrix(M_mat2, 'M_mat2.csv');