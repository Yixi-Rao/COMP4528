%% Read/Write an Image
 
img = imread('Lenna.png');
imshow(img);
 
imwrite(img,'new_lenna.png');

%% Image Properties

imgSize = size(img) % [rows, columns, channels]
imgEncoding = class(img)

%% data type conversion using casting
img_double = double(img);
img_uint8  = uint8(img);

%% color conversion
% RGB to gray
img_gray = rgb2gray(img);
img_hsv  = rgb2hsv(img);

% get HSV components
H = img_hsv(:,:,1);
S = img_hsv(:,:,2);
V = img_hsv(:,:,3);

%% pixel iteration
img = imread('Lenna.png');
imgGray = rgb2gray(img);
imgThr = zeros(size(imgGray));
% for each row
for row = 1 : size(imgGray,1)
    % for each column
    for col = 1 : size(imgGray,2)
        pix = imgGray(row, col);
        
        % do something with pix 
        if pix > 150
            imgThr(row,col)=1;
        else
            imgThr(row,col)=0;
        end
    end
end
imshow(imgThr)

% image threshold, the easy way
imgThr = imgGray > 150;

%% Accessing

r = 240; c = 235;
% extract RGB at (r,c)
pixRGB = img(r, c, :);
% equivalent
pixRGB = img(r, c, 1:3);
% GB channels at (r,c)
pixGB = img(r, c, 2:3);
% crop image
w = 150; h = 50;
imgCrop = img(r : r + h, c : c + w, :);
figure; imshow(imgCrop);

%% Convolution

% prepare the mask
maskSize = 15;
avgMask = ones(maskSize) / maskSize^2;
% apply filter
imgAvg = imfilter(imgGray, avgMask);
figure; imshow(imgAvg)
figure; imshow(imgGray)
%%
% X and Y masks
avgMask_Y = ones(15, 1) / 15^2;
avgMask_X = ones(1, 15) / 15^2;
% apply convolution
imgAvg = conv2(avgMask_Y, avgMask_X, imgGray, 'same');
imshow(imgAvg)
help conv2

%% Histogram
img = imread('pears.png');
imgGray = rgb2gray(img);
imshow(imgGray)
bins = 50;
% if you want to display
figure;
imhist(imgGray, bins);
% if you want to get the values
[counts, binPos] = imhist(imgGray, bins);
%% Equalizes image to a flat histogram
img = imread('pears.png');
imgGray = rgb2gray(img);
% equalizes the image to a flat histogram
bins = 50;
imgEq = histeq(imgGray, bins);
imhist(imgEq);
imshow(imgEq)

%% Morphology operators
SE = [[0 1 0];
 [1 1 1];
 [0 1 0]];
N = 5;
for i = 1:N
    imgThr = imdilate(imgThr, SE, 'same');
end
imshow(imgThr)
for i = 1:N
    imgThr = imerode(imgThr, SE, 'same');
end
figure; imshow(imgThr)

%% Resizing

% resize by scale factor
imgResize_1 = imresize(imgGray, 0.5);
% resize by desired resolution
imgResize_2 = imresize(imgGray, [240, 320]);

%% Cropping

% Image crop [x, y, width, height]
rect = [200, 250, 200, 200];
imgCrop = imcrop(imgGray, rect);
% Image crop by slicing [rows, cols]
imgCrop = imgGray(250:450, 200:400);


%% Transformations | Translate

% pixel translation [x, y]
translation = [50, -20];
imgTranslate = imtranslate(imgGray, translation);
imshow(imgTranslate)
%% Transformations | Rotation
angleDeg = 45;
% output image size different than input
imgRotate_1 = imrotate(imgGray, angleDeg,'bilinear', 'loose');
% output size equal than input
imgRotate_2 = imrotate(imgGray, angleDeg,'bilinear', 'crop');
imshow(imgRotate_1)
imshow(imgRotate_2)
%% Transformations | Warping
rad = 0.25*pi;
R = [[ cos(rad), -sin(rad), 0];
 [sin(rad), cos(rad), 0];
 [ 0, 0, 1]];
sx = 0.5; sy = 0;
S = [[ 1, sy, 0];
 [ sx, 1, 0];
 [ 0, 0, 1]];
A = S*R;
tform = affine2d(A);
imgWarp = imwarp(imgGray, tform, 'FillValues', [0]);
imshow(imgWarp)