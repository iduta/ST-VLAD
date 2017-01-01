function ST_VLAD_encoding=ST_VLAD(features, vocabFeatures, positions, vocabPositions)
%ST_VLAD_encoding=ST_VLAD(features, vocabFeatures, positions, vocabPositions)
%Compute ST-VLAD (Spatio-Temporal VLAD) encoding for a set of
%features with their pozitions as presented in the paper:
%"Spatio-temporal VLAD Encoding for Human Action Recognition in Videos".
%
% input:
%    features: n x d matrix of features (n - number of features, d - the dimensionality of the features)
%    vocabFeatures: k1 x d the learned vocabulary of the features (k1 - number of visual words (centroids), d - the dimensionality of each visual word)
%    positions: n x 3 the features position (in our paper the asocieted position to each feature is represented by 3 values (x, y, t))
%    vocabPositions: k2 x 3 the learned vocabulary of the features position (k2 - the number of learned video divisions)
%
% output
%    ST_VLAD_encoding: k1×d + k2×(d+k1) ST-VLAD encoding vector. (be aware to normalize the resulted encoding vector as recommended in the paper)
%
%       Ionut Cosmin Duta - 2017


dimFeatures=size(features, 2); % the feature dimensionality
k1=size(vocabFeatures, 1);  % the size of the vocabulary for the features



%Calculate the similarity using Euclidean distance
%"distmj" is a function which efficiently compute the Euclidean distance
%between two vectors
distance=distmj(features, vocabFeatures);

%perform the hard assignment for the features
[~, assign]=min(distance, [], 2);

%Calculate VLAD for each word of the vocabulary
wordVLAD=cell(1, k1);

%to save the residual for each feature
VLADs=zeros(size(features));

%to save the membership information for each feature
memb=zeros(size(features,1), k1);

for i=1:k1
    
    assigned=(assign==i); % get the features assigned to the cluster i;
    nAssigned=sum(assigned);
    if nAssigned>0 % compute VLAD for each visual word (cluster) that has at least one feature assigned
        
        %compute the difference between features and visual word (cluster)
        diff=bsxfun(@minus, features(assigned, :), vocabFeatures(i, :));
        
        %save the residual for each feature
        VLADs(assigned, :)=diff;
        
        %save membership information for each feature
        memb(assigned, i)=1;
        
        %perform the average pooling over these differences
        wordVLAD{i}=(1.0/nAssigned)*sum(diff, 1);
        %wordVLAD{i}=mean(diff, 1);
        
    else
        % no features assigned to a cluster then put zeros
        wordVLAD{i}=zeros(1, dimFeatures);      
    end 
        
        
end

%Concatenate all the VLAD vectors for each cluster to create the final VLAD
%vector
VLAD=cat(2, wordVLAD{:});



%the assignement for pozition of the features needed for the
%spatio-temporal pooling

stDistance = distmj(positions, vocabPositions);
[~, stAssign] = min(stDistance, [], 2);

%the size of vocabulary for the location of the features
k2=size(vocabPositions, 1);

stWordVlad=cell(1, k2);
stMemb=cell(1, k2);

%compute the ST Encoding
for i=1:k2
    stAssigned=(stAssign==i);
    nAssigned=sum(stAssigned);
    
    if nAssigned>0
        %the spatio-temporal pooling of the feature residuals
        stWordVlad{i}=(1.0/nAssigned) * sum(VLADs(stAssigned, :), 1);
        
        %the spatio-temporal pooling of the membership information
        stMemb{i}=sum(memb(stAssigned, :), 1);
    else
        stWordVlad{i}=zeros(1, dimFeatures);
        stMemb{i}=zeros(1, k1);
    end
end


spVLAD=cat(2, stWordVlad{:});
stMemb=cat(2, stMemb{:});

%concatenate all information to create the final representation
ST_VLAD_encoding=cat(2, VLAD, spVLAD, stMemb);



