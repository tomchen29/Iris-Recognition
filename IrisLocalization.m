%% This script is used to localize the pupil's and iris's circle of each image
%% All the results have been pre-saved into the folder "MatLab_circle_train" and "MatLab_circle_test"

% need to specific if you want to generate train or test files
goal = 'train';  
%goal = 'test';
current_path = mfilename('fullpath');
current_path = current_path(1:end-17);
fid1 = fopen(fullfile(current_path, strcat(goal, '_location.txt')),'r');

counter = 1;

while ~feof(fid1)
    c = {};
    img_location = fgetl(fid1);
    disp(img_location);
    origin_img = imread(img_location);
    blur = imgaussfilt(origin_img, 5);
    bm=edge(blur,'canny');
    
    %% Find inner boundary (pupil border)
    [pupil_center, pupil_radii] = imfindcircles(bm,[20 70], ...
        'ObjectPolarity','dark','Method','TwoStage','Sensitivity',0.81,'EdgeThreshold',0.1);
    
    if goal == 'train'
        %% Find testing outer boundary (iris)
        [iris_center, iris_radii] = imfindcircles(bm,[90 140], ...
            'ObjectPolarity','dark','Method','TwoStage','Sensitivity',0.96,'EdgeThreshold',0.1);
    elseif goal == 'test'
        % Find training outer boundary (iris)
        [iris_center, iris_radii] = imfindcircles(bm,[90 120], ...
            'ObjectPolarity','dark','Method','TwoStage','Sensitivity',0.99,'EdgeThreshold',0.1);
    end
    
    c = [c,pupil_center];
    c = [c, pupil_radii];
    c = [c, iris_center];
    c = [c, iris_radii];
    c = [c, img_location];
    
%     imshow(origin_img);
%     viscircles(pupil_center,pupil_radii);
%     viscircles(iris_center,iris_radii);
    
    save_name = fullfile(current_path, strcat('MatLab_circle_', goal), strcat(goal, '_circle_', num2str(counter), '.mat'));
    save(save_name, 'c');
    counter = counter+1;
end
