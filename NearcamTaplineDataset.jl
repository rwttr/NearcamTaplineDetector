module NearcamTaplineDataset

import JSON # groundtruth file
import CSV  # k-fold 
import FileIO
import Images
import Statistics
import Flux # dataloader
import Random
import ImageView # Visualize Sample

#global var
export dataset_path
export kfold_ksize

#functions
export init # buildDataset
export dispatchData
export resetDispatchRecord
export getDispatchRecord
export showImageSample
export estimateAnchors

const kfold_ksize = 5;

include("nearcamDataFold.jl")
end