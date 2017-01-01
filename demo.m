%this is a toy example of how to use the encoding method ST-VLAD, presented
%in the paper: "Spatio-temporal VLAD Encoding for Human Action Recognition in Videos", 
%Ionut C. Duta;  Bogdan Ionescu; Kiyoharu Aizawa; Nicu Sebe.
%In International Conference on Multimedia Modeling, 2017.
%
%Please cite our work when using this code!


%In this example we use "fake" features by generating them randomly. 
%Of course, in practice you should use features extracted from your data. 
%For instance, for using a "real" video descriptors you can download the code for 
%descriptor extraction from here: http://disi.unitn.it/~duta/software.html

%generate the features for creating the vocabulary
nFeatVocab=1000; %the number of features from which the vocabulary is generated. Use much more features in real cases!!!!
featCreateVocab=rand(nFeatVocab, 48);

%create the features vocabulary with the standard k-means, can be quite slow process
k1=256; % the number of visual words for the vocabulary of the features
[~, vocabFeatures]=kmeans(featCreateVocab, k1); 


%generate the position of the features for creating the vocabulary
posCreateVocab=rand(nFeatVocab, 3);

%create the vocabulary of the features position with the standard k-means
k2=32; % the number of visual words for the vocabulary of the features position
[~, vocabPositions]=kmeans(posCreateVocab, k2); 

nFeatures=100;
%generate the features for which the encoding is performed
features=rand(nFeatures, 48);

%generate the features pozition
positions=rand(nFeatures, 3);


%obtain the final ST-VLAD encoding
ST_VLAD_encoding=ST_VLAD(features, vocabFeatures, positions, vocabPositions);

%after you obtain the final representation, before classification, 
%you may want to normalize the vector as in our paper: 
%apply power normalization followed by L2 normalization

%apply power normalization
alpha=0.1;
norm_ST_VLAD_encoding=PowerNormalization(ST_VLAD_encoding, alpha);

%apply L2 normalization for making unit length
norm_ST_VLAD_encoding=NormalizeRowsUnit(norm_ST_VLAD_encoding);


