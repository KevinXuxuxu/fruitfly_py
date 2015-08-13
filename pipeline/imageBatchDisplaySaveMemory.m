function imageBatchDisplaySaveMemory(D,width,height,nrow,ncol,colorScale,r)
% D: a list of images -- each column corresponding to an image
% width/height: the width/height of an image
% nrow/ncol: how many rows/cols in the layout?
% colorScale: can be '' or 'gray'
% r: a scaling parameter, 0.5 for 64by32 image and 0.25 for 32by16 images
% e.g. imageBatchDisplaySaveMemory(D,32,16,8,8,'gray',0.25);
template = generateTemplate();

if ~exist('r')
    template = imresize(template,0.5,'nearest');
else
    template = imresize(template,r,'nearest');
end
ind = find(template(:,:,1)==1);
Dfull = zeros(height*width,size(D,2));
Dfull(ind,:) = D;
imageBatchDisplay(Dfull, width, height, nrow,ncol,colorScale); 