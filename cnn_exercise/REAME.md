
#CNN notes
by Jiarui Fang
##function cnnConvolve

convolution is to compute σ(Wx(r,c) + b) for all valid patches in the image. That is we need the W and b, which are trained by the LinearAutoSparseCoder by sampled patches.
<p> W is visibleSize * hiddenSize
<p> b is visibleSize?

####How to get W and b after whitening?

Because the sampled batches for linearDecoder have already been whiten, to get the feature we need to do transfer with matrix: WT= W*ZACWhite.

The feature is WT. however, why the intercept became like follow?
<p> `b_mean = b - WT*meanPatch;`

####How to get new feature after convolution?

Now we have the W and b. A naive method to get convoluted features is : for each patch in the image, we do σ(Wx(r,c) + b). 
<p>That is 1. compute Wx(r,c) for all (r,c). Require a loop!
<p>2. Add b to all the computed values.
<p>3. sigmod
However, it is slow.

Convolution method: use partial W as a matrix to do convolution operation on the image.

Perform a conv2 2-D convolution, using the weight matrix for the featureNum-th feature and channel-th channel, and the image matrix for the imageNum-th image.
Here we do not get a feature value but a small image.

x_small -> σ(Wx_small(r,c) + b) -> K features
x_whole -> for every patch σ(Wx_small(r,c) + b) -> K * (r-a+1) * (c-b+1) features


##check your convolution
choose small patch the same as sampled patch for training and then do forward propogration to get hiddenSize activation. Check the activation with convolution results.

##Pooling
In order to achieve translation invariant, a pooling layer is required after a convulotion layer.

##BUG report
NO

##Questions
#######How does the WT and b_mean calculated?

#######What does the following code do in cnnConv function?
      % Flip the feature matrix because of the definition of convolution, as explained later
      feature = flipud(fliplr(squeeze(feature)));
      
      % Obtain the image
      im = squeeze(images(:, :, channel, imageNum));
