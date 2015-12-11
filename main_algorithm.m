function main_algorithm(params)
%% set path for all the source code
addpath(genpath('./Tools'));
addpath(genpath('./Thresholding'));
addpath(genpath('./Layered_FG_BG'));
addpath(genpath('./FeatureDescriptors'));
addpath(genpath('./AdaptiveCB'));

%% parse all the paramters
rDir=params.rDir;
likelihood_file=params.keyPart;
%training and testing data directory
Train_Directory=params.Train_Directory;
train_img_ext=params.train_img_ext;
train_img_prix=params.train_img_prix;
train_img_method=params.train_img_method;
training_start=params.training_start;
training_end=params.training_end;
% testing settings
Test_Directory=params.Test_Directory;
test_img_ext=params.test_img_ext;
test_img_prix=params.test_img_prix;
test_img_method=params.test_img_method;%1: %d;2:%02d;3: %03d;etc..when construct the image file names
test_start=params.test_start;
test_end=params.test_end;
down_sample_rate=params.down_sample_rate;% down sample the images both in training and testing
% core algorithm parameters
STV_size=params.STV_size;% Size of video volumes
Parameters_STV=params.Parameters_STV; % spatial Overlap ratio,  number of codewords; sliding parameter:0 no sliding,1 sliding; number of frames for each batch data
Ensemble_size=params.Ensemble_size; %Ensemble size: Actual region would be 10*Ensemble_size;
Parameters_ensembles=params.Parameters_ensembles;% second parameter:frames of chunk
ACB_Parameters=params.ACB_Parameters;%adaptive codebook construction parameters
Layer_Parameters=params.Layer_Parameters;
NN_frames=params.NN_frames;%IMPORTANT additional frames to compute nearest neighbor feature descriptor,when not to compute NN,set to zero
%feature Paramters
Feature_Parameters.Save_Features=params.Feature_Parameters.Save_Features;%for init
Feature_Parameters.Load_Features=params.Feature_Parameters.Load_Features;
Feature_Parameters.Feature_Type=params.Feature_Parameters.Feature_Type;
Feature_Parameters.Save_Test_Features=params.Feature_Parameters.Save_Test_Features; %for test
Feature_Parameters.Load_Test_Features=params.Feature_Parameters.Load_Test_Features;
Feature_Parameters.STHOG=params.Feature_Parameters.STHOG;%spatial-temporal HOG:spatial bins;s-t bins;
Feature_Parameters.LNND=params.Feature_Parameters.LNND;%local NN distance descriptor
Feature_Parameters.SOD=params.Feature_Parameters.SOD;%Simplex-based feature descriptors:K
Feature_Parameters.TD=params.Feature_Parameters.TD;%temporal deriviate descriptors;
Feature_Parameters.HOG3D=params.Feature_Parameters.HOG3D;%n-polyhedren,xyCells,tCells,nPixels;
Feature_Parameters.MCSTLNND=params.Feature_Parameters.MCSTLNND;%x-y-t serach radius, number of subregions,sub-region radius square boundaries, K nearest neighbor for each sub-region
Feature_Parameters.STColorHist=params.Feature_Parameters.STColorHist;
Feature_Parameters.ORG=params.Feature_Parameters.ORG;
%% read training data
visual_cues=params.visual_cues;
use_visual_cues=params.use_visual_cues;

V_Frames=ReadImagesFromColorSpaces(Train_Directory,training_start,training_end,train_img_ext,train_img_prix,train_img_method,down_sample_rate,...
    visual_cues,use_visual_cues);

%% Initialize by constructing the codebook and ensemble topology
%Smooth video with gaussians
%V=smooth3(double(V),'gaussian',7,1.5);

disp('Init codebook and weights...');
% Initialization of the algorithm
% % M.0 ORG
% % M.2 Felix adptive codebook
[STV_codebook,Weight]=ACB_initialize_alg_STV(V_Frames,STV_size,Parameters_STV,ACB_Parameters,Feature_Parameters);%by ACB 
% % M.3 Felix load data from mat files
% dFile1=load('./Variables/STV_codebook_Init.mat');
% STV_codebook=dFile1.STV_codebook;
% dFile2=load('./Variables/Weight_Init.mat');
% Weight=dFile2.Weight;
fg_layer_mask_init=[];

disp('Init ensembles..');
% % Construct Ensembles
% % M.1
tic;
[Gamma,sz]=ACB_initialize_alg_ensembles2(Weight,Ensemble_size,Parameters_ensembles,fg_layer_mask_init); %ACB
toc
% % M.2
% dFile3=load('./Variables/gGamma_Init.mat');
% Gamma=dFile3.Gamma;

%% SAVE the init information
if  ACB_Parameters(6)
     if ~exist([rDir 'TmpVariables/'  likelihood_file '/'],'dir')
             mkdir([rDir 'TmpVariables/'  likelihood_file '/']);
     end
     save([rDir 'TmpVariables/'  likelihood_file '/STV_codebook_Init.mat'],'STV_codebook');
     %save([rDir 'TmpVariables/'  likelihood_file '/Weight_Init.mat'],'Weight','-v7.3');
     %save([rDir 'TmpVariables/'  likelihood_file '/gGamma_Init.mat'],'Gamma');
end


%[ro,co,nc,T]=size(Weight);
ro=sz(1);co=sz(2);
clear Weight;% Weight is never used in the following procedure

% % Cut several frames from the training set for the convience to compute the likelihood for the first frame at the testing stage
if Parameters_STV(3)<=1
      Overlap_step_STV_T=max(floor((1-Parameters_STV(3))*STV_size(3)),1);
else
      Overlap_step_STV_T=Parameters_STV(3);
end
%lastn=max(STV_size(3)+Ensemble_size(3)-1,(STV_size(3)+NN_frames-1)+(Ensemble_size(3)-1))+1;%%% M.3 for multiple color space
lastn=max(STV_size(3)+(Ensemble_size(3)-1)*Overlap_step_STV_T-1,2)+2;%%% M.3 only for one ensemble, note that STV_size(3)=ceil(STV_size(3)/2)+floor(STV_size(3)/2)
% temporal_shift=floor((lastn+1)/2);% here lastn is the one without max function, i.e. lastn=STV_size(3)+(Ensemble_size(3)-1)*Overlap_step_STV_T-1
Frames=cut_last_few_frames(V_Frames,lastn,use_visual_cues,visual_cues);
% %clear data
for i=1:length(use_visual_cues)
    clear V_Frames(i).data;
end

%% testing some image sequence
% % Similarity map construction
learning_rate=0.01; %%% the speed to update the topology Gamma
cur=1;
disp('Similarity map construction...');
%Likelihood_map=zeros(ro,co,test_end-test_start+1);
%entropy_map=zeros(ro,co,test_end-test_start+1);

%% for testing codeword distribution entropy without codebook updating
N_O=Layer_Parameters(4);%number of frames to observe the change of codewords
max_codewords=[];%zeros(ro,co,N_O);
cb_init_size=STV_codebook.Codewords_num;
%% for storing the best ensembel config and the weights
% % for all the testing frames
% best_config=zeros(ro,co,test_end-test_start+1,1+Ensemble_size(1)*Ensemble_size(2)*Ensemble_size(3));
% best_config_weights=zeros(ro,co,test_end-test_start+1,1+Ensemble_size(1)*Ensemble_size(2)*Ensemble_size(3));
% best_config_gamma=zeros(ro,co,test_end-test_start+1,1+Ensemble_size(1)*Ensemble_size(2)*Ensemble_size(3));

for t=test_start:test_end
    % load a new frame:
    fname=getImageFileName(t,test_img_ext,test_img_prix,test_img_method);
    img=imread([Test_Directory,fname]);
    V = (imresize(img,down_sample_rate));%down-sampled
     
    % M.1 Concatenate latest a few frames in the training set with the current testing frame to form the query
    New_frame=addcurFrame(Frames,V,visual_cues,use_visual_cues);
    %[Likelihood_map(:,:,cur), Gamma,STV_codebook, STV_W,gamma,best_config(:,:,cur,:),best_config_weights(:,:,cur,:),best_config_gamma(:,:,cur,:)]=
    [Likelihood_map, Gamma,STV_codebook, ~, ~,best_config,best_config_weights,best_config_gamma]=ACB_Similarity_measurement(New_frame,STV_codebook,Gamma,...
            STV_size,Parameters_STV,Ensemble_size,Parameters_ensembles,learning_rate,ACB_Parameters,Layer_Parameters,Feature_Parameters,fg_layer_mask_init,cb_init_size,cur,max_codewords);
     
   Frames=cut_last_few_frames(New_frame,lastn,use_visual_cues,visual_cues);
    t
    cur=cur+1;
    
    % % save current frame information
    if  ACB_Parameters(6)
         if ~exist([rDir 'TmpVariables/'  likelihood_file '/'],'dir')
                 mkdir([rDir 'TmpVariables/'  likelihood_file '/']);
         end
         save([rDir 'TmpVariables/'  likelihood_file '/STV_codebook_fr' num2str(t) '.mat'],'STV_codebook');%Codebook is changing
         %save([rDir 'TmpVariables/'  likelihood_file '/gGamma_fr' num2str(t) '.mat'],'Gamma');% changes with time
         % best configuration information
         if ~exist([rDir 'Configs/' likelihood_file '/'],'dir')
             mkdir([rDir 'Configs/' likelihood_file '/']);
         end
         save([rDir 'Configs/' likelihood_file '/bc_cw-idx_fr' num2str(t) '.mat'],'best_config','-v7.3');
         save([rDir 'Configs/' likelihood_file '/bc_weights_fr' num2str(t) '.mat'],'best_config_weights','-v7.3');
         save([rDir 'Configs/' likelihood_file '/bc_edges_fr' num2str(t) '.mat'],'best_config_gamma','-v7.3');
         
         %save likelihood map for each frame
         if ~exist([rDir 'Frame_Likelihood/' likelihood_file '/'],'dir')
              mkdir([rDir 'Frame_Likelihood/' likelihood_file '/']);
         end
         save([rDir 'Frame_Likelihood/' likelihood_file '/L_fr' num2str(t) '.mat'],'Likelihood_map','-v7.3');
     end
end

%% Save the normalized likelihood map to disk, this method is only for small dataset
%save([rDir likelihood_file '.mat'],'Likelihood_map','-v7.3');
%%as the following step may cosume much memory, we store the information at each frame
%%save([rDir  'entropy_map_'    likelihood_file   '.mat'],'entropy_map');%for test
% save([rDir  'bc_cw-idx_' likelihood_file '.mat'],'best_config','-v7.3');% the best configuration in each ensemble
% save([rDir  'bc_weights_' likelihood_file '.mat'],'best_config_weights','-v7.3');
% save([rDir  'bc_edges_' likelihood_file '.mat'],'best_config_gamma','-v7.3');
end
