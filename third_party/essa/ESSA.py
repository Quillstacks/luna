# Import packages
import argparse
import cv2
import logging
import numpy as np
import os
import time
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from osgeo import gdal, ogr, osr
from torchvision import tv_tensors
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from typing import Sequence, List, Optional, Tuple, Union

# Set up the logger
logging.basicConfig(level = logging.INFO,
                    format='| %(asctime)s | %(levelname)s | Message: %(message)s',
                    datefmt='%d/%m/%y @ %H:%M:%S')

# Initialise arguments parser
PARSER = argparse.ArgumentParser()

# Input directory containing 2,048x2,048 LROC NAC images tiles
PARSER.add_argument("-i", "--inputdir",
                    required=True,
                    type=str,
                    help="Input directory containing image tiles [type: str]")

# Output directory for storing geo-referenced detections
PARSER.add_argument("-o", "--outputdir",
                    required=True,
                    type=str,
                    help="Output directory for saving detections [type: str]")

# Path to the ESSA PyTorch model checkpoint file
PARSER.add_argument("-m", "--modelpath",
                    required=True,
                    type=str,
                    help="Path to IMFMapper checkpoint file [type: str]")

# Bounding box score threshold to apply to ESSA detections
PARSER.add_argument("-s", "--score",
                    default=0.8,
                    type=float,
                    help="Bounding box score threshold [type: float]")

# Parse the arguments
ARGS = PARSER.parse_args()

class PitDataset(torch.utils.data.Dataset):
    
    "Class for creating a custom PyTorch dataset."
    
    def __init__(self, 
                image_list: List[str], 
                device: Union[torch.device, str, int] = torch.device("cuda"), 
                tile_size: int = 2048):
        
        '''
        Arguments:
            image_list (list of str):           List of filepaths to each tiled image to infer the model on.
            tile_size (int):                    Tile size of imagery. X and Y tile size should be equal.
            device (torch.device):              Name of the device used for hosting tensors.
        '''
        
        self.image_list = image_list
        self.tile_size = tile_size
        self.device = device

    def __len__(self):
        
        # Get the total number of features from the mask list
        n_features = len(self.image_list)
        
        return n_features
    
    def __getitem__(self,
                    idx: int):
        
        # Get the image path
        image_path = self.image_list[idx]
                
        # Read in the image as Uint8
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(int)
        
        # Check the image was successfully read in
        if image is None:
            raise OSError(f"Image could not be read from {image_path}")
        
        # Check the shape of the image
        if image.shape != (self.tile_size, self.tile_size):
            raise ValueError(f"Incorrect tile size of image data. Expected {(self.tile_size, self.tile_size)}, got {image.shape}")
        
        # Convert the image to Float32 tensor
        image_tensor = tv_tensors.Image(torch.as_tensor(np.expand_dims(image/255, axis=0), dtype=torch.float32, device=self.device))
        
        return image_tensor, idx

# Create a shapefile layer
def create_shp_layer(
                    output_path: str, 
                    output_srs: osr.SpatialReference
                    ) -> Tuple[gdal.Dataset, ogr.Layer]:
    
    # Define the driver name for dealing with vector data
    driver2 = ogr.GetDriverByName("ESRI Shapefile")

    # Remove output shapefile if it already exists
    if os.path.exists(output_path):
        driver2.DeleteDataSource(output_path)
        
    # Get the output directory and layer name
    output_dir, output_name = os.path.split(output_path)

    # Create the output shapefile
    output_ds = driver2.CreateDataSource(output_path)
    output_layer = output_ds.CreateLayer(os.path.splitext(output_name)[0], srs=output_srs)

    # Add fields to store the image name, class label and prediction score
    image_field = ogr.FieldDefn("image_id", ogr.OFTInteger)
    output_layer.CreateField(image_field)
    class_field = ogr.FieldDefn("class", ogr.OFTInteger)
    output_layer.CreateField(class_field)
    score_field = ogr.FieldDefn("score", ogr.OFTReal)
    score_field.SetWidth(4)
    score_field.SetPrecision(1)
    output_layer.CreateField(score_field)

    return output_ds, output_layer

# Perform model inference on all batches
def inference_loop(
                model: MaskRCNN, 
                device: torch.device, 
                batches: torch.utils.data.DataLoader, 
                masks_layer: ogr.Layer,
                image_list: List[str],
                score_threshold: float
                ) -> None:
    
    # Perform the evaluation on the CPU
    cpu_device = torch.device("cpu")
    
    # Set the model to evaluation mode
    model.eval()
    
    # Ensure that no gradients are computed during inference
    with torch.no_grad():
        
        # Loop through batches
        for i, (images, idxs) in enumerate(batches):
        
            # Push tensors to same device
            images = list(image.to(device) for image in images)
            
            # Infer the model on the validation images
            outputs = model(images)
            
            # Push the outputs to the device
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            
            # Perform non-maximum suppression on the outputs (irrespective of class)
            outputs = [non_maximum_suppression(output, score_threshold, iou_threshold=0.5, device=cpu_device) for output in outputs]
            
            # Save the model detections for this batch to the merged shapefile layers
            save_outputs(masks_layer, outputs, idxs, image_list, score_threshold)

# Perform non-maximum suppression (NMS) on bounding boxes
def non_maximum_suppression(output: dict, score_threshold: float, iou_threshold: float, device: torch.device) -> dict:
    
    # Get a mask of which detections meet the score threshold
    score_indices = output["scores"] >= score_threshold
    
    # Remove all detections with a confidence score less than 50%
    filtered_output = {k: v[score_indices] for k, v in output.items()}
    
    # Get the filtered detection scores and boxes
    scores = filtered_output["scores"]
    boxes = filtered_output["boxes"]
    
    # Get the number of detections that meet the score threshold
    n_detections = torch.numel(scores)
    
    # Only perform NMS if there is more than 1 detection
    if n_detections > 1:
        
        # Sort the detections by score from highest to lowest
        sorted_scores, order = torch.sort(scores, descending=True)
        
        # Get the indices of the unsorted boxes to iterate over
        indices = torch.arange(n_detections, device=device)
        
        # Initialise a tensor of ones
        keep = torch.ones_like(indices, dtype=torch.bool, device=device)
        
        # Calculate the IoU between all boxes to get a NxN table (and set the IoU of the same boxes to zero)
        ious = box_iou(boxes[order], boxes[order]).fill_diagonal_(0).to(device)
        
        # Loop through each detection
        for i in indices:
            
            # If this box hasn't been suppressed
            if keep[i]:
        
                # Find indices of other detections with sufficient IoU
                overlapped = torch.nonzero((ious[i, :] * keep) > iou_threshold)
                
                # Remove these overlapping boxes with a lower score
                keep[overlapped] = 0
            
        # Get the indices of the detections to keep
        nms_indices = order[keep]
        
        # Index the original outputs
        suppressed_output = {k: v[nms_indices] for k, v in filtered_output.items()}

        return suppressed_output
            
    # Otherwise just return the output filtered by the score
    else:
        return filtered_output
           
# Read in a raster image
def read_raster(
                raster_path: str
                ) -> Tuple[int, int, Sequence[float], str, str]:
    
    # Check that the file exists
    if os.path.exists(raster_path):
        
        # Open the raster dataset
        raster_dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if raster_dataset is not None:
        
            # Get the geotransform of the image
            geot = raster_dataset.GetGeoTransform()
            
            # Get the projection of the image
            proj = raster_dataset.GetProjection()
            image_srs_wkt = raster_dataset.GetProjectionRef()
            image_srs = osr.SpatialReference()
            image_srs.ImportFromWkt(image_srs_wkt)
            
            # Check there is only one raster band
            n_bands = raster_dataset.RasterCount 
            try:
                assert n_bands == 1
            except:
                raise AssertionError(f"Incorrect number of bands. Expected 1, got {n_bands}.")
            
            # Get the number of pixels in the x-y direction
            xsize, ysize = raster_dataset.RasterXSize, raster_dataset.RasterYSize

            return xsize, ysize, geot, image_srs, proj

        else:
            raise OSError(f"Raster file at {raster_path} could not be opened.")
        
    else:
        raise OSError(f"No raster file was found at {raster_path}.")
    
# Reproject a geometry to a new spatial reference
def reproject_geometry(
                    geom: ogr.Geometry, 
                    source_srs: osr.SpatialReference, 
                    target_srs: osr.SpatialReference
                    ) -> ogr.Geometry:
    
    # Check that the spatial references aren't equal
    try:
        assert source_srs != target_srs
    except:
        raise AssertionError(f"Source and target SRS are equal. Got {source_srs}.")
    
    try:
        
        # Define the transform between the source and target SRS
        transform = osr.CoordinateTransformation(source_srs, target_srs)
        
        # Clone the original geometry
        new_geom = geom.Clone()
        if new_geom is None:
            raise ValueError("Cloned geometry is NoneType")
        
        # Reproject the source geometry to the target SRS
        new_geom.Transform(transform)
        
        return new_geom
        
    except:
        raise ValueError(f"Transform between source (type: {type(source_srs)}) and target ({type(target_srs)}) spatial reference systems did not work for geom (type: {geom.GetGeometryName()}).")

# Save the outputs of applying the model to a batch as geo-referenced shapefiles
def save_outputs(
                masks_layer: ogr.Layer, 
                outputs: List[dict], 
                idxs: List[int], 
                image_list: List[str], 
                score_threshold: float
                ) -> None:
    
    # Get the output spatial reference 
    output_srs = masks_layer.GetSpatialRef()
    
    # Get the layer definitions of the bounding box and mask layers
    masks_layer_defn = masks_layer.GetLayerDefn()
    
    # Loop through all outputs in this batch
    for idx, output in zip(idxs, outputs):
        
        # Get the scores of all detections
        scores = output["scores"].cpu().detach().numpy()
        
        # Only plot the detections above a certain score threshold
        valid_detections = scores >= score_threshold
        
        # Check if there are any valid detections in this tile
        if np.sum(valid_detections) > 0:
            
            # Get the bounding boxes, classifications and masks
            boxes = output["boxes"].cpu().detach().numpy()
            labels = output["labels"].cpu().detach().numpy()
            masks = output["masks"].cpu().detach().numpy()
        
            # Get just the valid detections
            boxes = boxes[valid_detections]
            labels = labels[valid_detections]
            scores = scores[valid_detections]
            masks = masks[valid_detections]
            
            # Define the path to the tile and its name (removing the rotation angle)
            tile_path = image_list[idx]
            image_name = os.path.splitext(os.path.basename(tile_path))[0]
        
            # Read in the raster information
            _, _, geot, tile_srs, proj = read_raster(tile_path)
            
            # Get the tile size from one of the masks
            tile_size, y_tilesize = masks[0].shape[1:]
            assert tile_size, y_tilesize
            
            # Loop through all detections in this tile
            for j, box in enumerate(boxes):
                
                # Threshold the mask array
                mask_array = np.where(masks[j, 0, :, :] >= 0.5, 1, 0)
                
                # If there is a mask found
                if np.sum(mask_array) != 0:
                    
                    # Rasterise the thresholded mask array
                    masks_path = os.path.join(os.getcwd(), "temp_masks.tif")
                    masks_raster_ds, masks_raster_band = write_array_to_raster(masks_path, tile_size, mask_array, geot, proj, format_type=gdal.GDT_Int16)
                    
                    # Create a temporary shapefiles for polygonising into
                    temp_masks_path = os.path.join(os.getcwd(), "temp_masks.shp")
                    temp_masks_ds, temp_masks_layer = create_shp_layer(temp_masks_path, tile_srs)
                    
                    # Check that the raster and vector layers have been created
                    if os.path.exists(masks_path) and os.path.exists(temp_masks_path):
                        
                        # Polygonise the masks rasters into the temporary shapefile
                        gdal.Polygonize(masks_raster_band, masks_raster_band, temp_masks_layer, -1, [], callback=None)
                        
                        # Check that there are valid features
                        if temp_masks_layer.GetFeatureCount() == 0:
                            raise ValueError("No valid features found in temporary polygonised detection shapefile")
                        
                        # Loop through all detections in the temporary shapefile (should only be 1)
                        for m in range(0, temp_masks_layer.GetFeatureCount()):
                            
                            # Get the feature from the temporary layer
                            feature = temp_masks_layer.GetFeature(m)
                            
                            # Get the geometry of the feature
                            geom = feature.GetGeometryRef()
                            
                            # Reproject the geometry to the spatial reference of the merged 
                            reprojected_geom = reproject_geometry(geom, tile_srs, output_srs)
                            
                            # Delete the feature
                            temp_masks_layer.DeleteFeature(feature.GetFID())
                            
                        # Create the a new feature in the masks layer
                        new_feature = ogr.Feature(masks_layer_defn)
                        
                        # Set the attributes of the feature (classification and score)
                        new_feature.SetField("image_id", image_name)
                        new_feature.SetField("class", int(labels[j]))
                        new_feature.SetField("score", np.around(scores[j]*100, decimals=1))
                    
                        # Set the geometry of the detection
                        new_feature.SetGeometry(reprojected_geom)
                        
                        # Create the feature
                        masks_layer.CreateFeature(new_feature)
                        new_feature = None
                                                
                        # Close the temporary datasource and layer
                        temp_masks_ds = temp_masks_layer = None
                        masks_raster_ds = masks_raster_band = None
                        
                        # Remove the temporary raster and vector files
                        for file in os.listdir(os.getcwd()):
                            if file.startswith("temp"):
                                os.remove(os.path.join(os.getcwd(), file))
                    
                    else:
                        raise OSError(f"Temporary raster or vector file(s) could not be found")

# Rasterise a NumPy array
def write_array_to_raster(
                        save_path: str, 
                        tile_size: int, 
                        array: np.ndarray, 
                        geot: Optional[Sequence[float]] = None, 
                        proj: Optional[str] = None, 
                        format_type = gdal.GDT_Byte
                        ) -> Tuple[gdal.Dataset, gdal.Band]:

    # Define the driver name for dealing with raster data
    driver1 = gdal.GetDriverByName('GTiff')   

    # Create the raster dataset
    raster_dataset = driver1.Create(save_path, tile_size, tile_size, 1, format_type)
    
    # Set the geotransform
    if geot is not None:
        raster_dataset.SetGeoTransform(geot)
    
    # Set the image projection
    if proj is not None:
        raster_dataset.SetProjection(proj)
    
    # Write the array to the raster band
    raster_band = raster_dataset.GetRasterBand(1)
    raster_band.SetNoDataValue(np.nan)
    raster_band.WriteArray(array)          
    raster_band.FlushCache()
    
    return raster_dataset, raster_band

def main(input_dir: str,
        output_dir: str,
        model_path: str,
        score_threshold: float):
    
    if not os.path.exists(input_dir):
        raise OSError(f"Input directory {input_dir} does not exist")
    
    logging.info(f"Inferring ESSA (Entrances to Sub-Surface Areas) on the images in: {input_dir}")
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set which device should be used for processing
    if torch.cuda.is_available():
        logging.info("Using GPU for inference.")
        device = torch.device('cuda')
    else:
        logging.info("Using CPU for inference.")
        device = torch.device('cpu')

    # List the full paths to all image tiles
    image_list = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if (filename.endswith("tif") or filename.endswith("tiff"))]
    if len(image_list) == 0:
        raise OSError(f"No image tiles (tif or tiff) were found in: {input_dir}")
    
    logging.info(f"No. of inference samples: {len(image_list)}")
    
    # Import the dataset for inferring the model on
    dataset = PitDataset(image_list=image_list,
                        device=device)

    # Allow for iterating over the dataset
    batches = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialise the Mask R-CNN model
    model = maskrcnn_resnet50_fpn_v2(weights_backbone=ResNet50_Weights.IMAGENET1K_V2, trainable_backbone_layers=5)
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)
    
    # Push the model to the device
    model.to(device)
    
    # Load model checkpoint (see https://doi.org/10.5281/zenodo.15095750)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Define the output spatial reference for the detections
    output_srs = osr.SpatialReference()
    output_srs.ImportFromProj4("+proj=longlat +a=1737400 +b=1737400 +no_defs")
    
    # Create shapefile layers for storing detections
    output_path = os.path.join(output_dir, "masks.shp")
    output_ds, output_layer = create_shp_layer(output_path=output_path,
                                            output_srs=output_srs)
    
    logging.info("Beginning inference:")
    
    # Start a timer
    begin = time.time()

    # Infer the model on the tiled images
    inference_loop(model=model,
                device=device,
                batches=batches,
                masks_layer=output_layer,
                image_list=image_list,
                score_threshold=score_threshold)
    
    # Close the vector datasets and layeres
    output_ds = output_layer = None
    
    # End the timer
    end = time.time()

    logging.info(f"Inference complete. Took approx. {(end - begin) / 60:0.2f} min(s).")
    
if __name__ == "__main__":
    main(input_dir=ARGS.inputdir,
        output_dir=ARGS.outputdir,
        model_path=ARGS.modelpath,
        score_threshold=ARGS.score)