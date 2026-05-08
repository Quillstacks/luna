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
from torchvision.models.segmentation import DeepLabV3, deeplabv3_resnet50
from typing import Sequence, List, Optional, Tuple, Union

# Set up the logger
logging.basicConfig(level = logging.INFO,
                    format='| %(asctime)s | %(levelname)s | Message: %(message)s',
                    datefmt='%d/%m/%y @ %H:%M:%S')

# Initialise arguments parser
PARSER = argparse.ArgumentParser()

# Input directory containing 512x512 LROC NAC images tiles
PARSER.add_argument("-i", "--inputdir",
                    required=True,
                    type=str,
                    help="Input directory containing image tiles [type: str]")

# Output directory for storing geo-referenced detections
PARSER.add_argument("-o", "--outputdir",
                    required=True,
                    type=str,
                    help="Output directory for saving detections [type: str]")

# Path to the IMFMapper PyTorch model checkpoint file
PARSER.add_argument("-m", "--modelpath",
                    required=True,
                    type=str,
                    help="Path to IMFMapper checkpoint file [type: str]")

# Parse the arguments
ARGS = PARSER.parse_args()

# Custom PyTorch dataset for 512x512 LROC NAC tiles of IMFs (at 1.5 m/px)
class IMFDataset(torch.utils.data.Dataset):
    
    "Class for creating a custom PyTorch dataset."
    
    def __init__(self, 
                image_list: List[str],
                device: Optional[Union[torch.device, str, int]] = torch.device("cuda"),
                tile_size: int = 512):
        
        '''
        Arguments:
            image_list (list of str):           List of filepaths to each tiled image to infer the model on.
            device (torch.device):              Name of the device used for hosting tensors.
            tile_size (int):                    Tile size of imagery. X and Y tile size should be equal.
        '''
        
        self.image_list = image_list
        self.device = device
        self.tile_size = tile_size

    def __len__(self):
        
        # Get the total number of images
        n_features = len(self.image_list)
        
        return n_features
    
    def __getitem__(self, idx: int):
        
        # Get the image path
        image_path = self.image_list[idx]
                
        # Read in the image as Uint8
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(int)
        
        # Check the image was successfully read in
        if image is None:
            raise OSError(f"Image could not be read from {image_path}")
        
        # Convert to Uint8
        image = image.astype(int)
        
        # Check the shape of the image
        if image.shape != (self.tile_size, self.tile_size):
            raise ValueError(f"Incorrect tile size of image data. Expected {(self.tile_size, self.tile_size)}, got {image.shape}")
        
        # Convert the image into a torch tensor in [C, H, W] format
        image_tensor = tv_tensors.Image(torch.as_tensor(np.repeat(np.expand_dims(image / 255, axis=0), axis=0, repeats=3), dtype=torch.float32), device=self.device)
          
        # Return the image and raster info
        return image_tensor, idx

# Create a shapefile layer
def create_shp_layer(output_path: str, output_srs: osr.SpatialReference) -> Tuple[gdal.Dataset, ogr.Layer]:
    
    # Define the driver name for dealing with vector data
    driver2 = ogr.GetDriverByName("ESRI Shapefile")
    
    # Remove output shapefile if it already exists
    if os.path.exists(output_path):
        driver2.DeleteDataSource(output_path)
    
    # Get the output directory and layer name
    output_dir, output_name = os.path.split(output_path)

    # Create the output shapefile
    output_ds = driver2.CreateDataSource(output_path)
    output_layer = output_ds.CreateLayer(output_name, srs=output_srs)
        
    return output_ds, output_layer

# Perform model inference on all batches
def inference_loop(model: DeepLabV3, batches: torch.utils.data.DataLoader, output_layer: ogr.Layer, image_list: List[str]) -> None:
    
    # Set the model to evaluation mode
    model.eval()
    
    # Ensure that no gradients are computed during test mode
    with torch.no_grad():
        
        # Loop through batches
        for batch_no, (images, idxs) in enumerate(batches):
        
            # Infer the model on the batched images
            outputs = model(images)['out']
            probabilities = torch.sigmoid(outputs)
            
            # Classify pixels
            predictions = torch.argmax(probabilities, dim=1)
            
            # Push the predictions to the cpu device
            predictions = predictions.to(dtype=torch.uint8)
            
            # Save the predictions to a georeferenced shapefile
            save_outputs(predictions, output_layer, idxs, image_list)

# Read in a raster image
def read_raster(raster_path: str) -> Tuple[int, int, Sequence[float], str, str]:
    
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
def reproject_geometry(geom: ogr.Geometry, source_srs: osr.SpatialReference, target_srs: osr.SpatialReference) -> ogr.Geometry:
    
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

# Save the outputs to a georeferenced shapefile
def save_outputs(predictions: torch.Tensor, output_layer: ogr.Layer, idxs: List[int], image_list: List[str]) -> None:
    
    # Get the output spatial reference and layer definition
    output_srs = output_layer.GetSpatialRef()
    output_layer_defn = output_layer.GetLayerDefn()
    
    # Loop through the detections in each image of the batch
    for idx, prediction in zip(idxs, predictions):
        
        # Convert the prediction to a NumPy array
        prediction_array = prediction.cpu().detach().numpy()
        
        # Check that there were detections made
        if np.sum(prediction_array) > 0:
        
            # Define the path to the tile and its name (removing the rotation angle)
            tile_path = image_list[idx]
        
            # Read in the raster information
            xsize, ysize, geot, tile_srs, proj = read_raster(tile_path)
            
            # Check that the prediction is the correct shape
            if prediction_array.shape != (xsize, ysize):
                raise ValueError(f"Incorrect prediction array shape. Expected ({xsize}, {ysize}) got ({prediction_array.shape}).")
            
            # Save the prediction as a temporary raster
            temp_raster_path = os.path.join(os.getcwd(), "temp.tif")
            temp_raster_ds, temp_raster_band = write_array_to_raster(temp_raster_path, xsize, prediction_array, geot, proj, format_type=gdal.GDT_Int16)
            
            # Create a temporary shapefile layer for polygonising into
            temp_shp_path = os.path.join(os.getcwd(), "temp.shp")
            temp_shp_ds, temp_shp_layer = create_shp_layer(temp_shp_path, tile_srs)
            
            # Check that the temporary raster and vector layers have been created
            if os.path.exists(temp_raster_path) and os.path.exists(temp_shp_path):
                
                # Polygonise the masks rasters into the temporary shapefile
                gdal.Polygonize(temp_raster_band, temp_raster_band, temp_shp_layer, -1, [], callback=None)
                
                # Check that there are valid features
                if temp_shp_layer.GetFeatureCount() == 0:
                    raise ValueError("No valid features found in temporary shapefile.")
                
                # Loop through each detection in the temporary shapefile
                for m in range(0, temp_shp_layer.GetFeatureCount()):
                    
                    # Get the feature from the temporary layer
                    temp_feature = temp_shp_layer.GetFeature(m)
                    
                    # Get the geometry of the feature
                    temp_geom = temp_feature.GetGeometryRef()
                    
                    # Reproject the geometry to the desired spatial reference 
                    new_geom = reproject_geometry(temp_geom, tile_srs, output_srs)
                    
                    # Define the new feature in the output layer
                    new_feature = ogr.Feature(output_layer_defn)
                
                    # Set the geometry of the detection
                    new_feature.SetGeometry(new_geom)
                    
                    # Create the feature in the merged layer
                    output_layer.CreateFeature(new_feature)
                    new_feature = None
                                        
                # Close the temporary layer datasource/layer
                temp_shp_ds = temp_shp_layer = None
                temp_raster_ds = temp_raster_band = None
                
                # Remove the temporary raster and vector files
                for file in os.listdir(os.getcwd()):
                    if file.startswith("temp"):
                        os.remove(os.path.join(os.getcwd(), file))

# Rasterise a NumPy array
def write_array_to_raster(save_path: str, tile_size: int, array: np.ndarray, geot: Optional[Sequence[float]] = None, proj: Optional[str] = None, format_type = gdal.GDT_Byte) -> Tuple[gdal.Dataset, gdal.Band]:

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
        model_path: str):
    
    if not os.path.exists(input_dir):
        raise OSError(f"Input directory {input_dir} does not exist")
    
    logging.info(f"Inferring IMFMapper on the images in: {input_dir}")

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set which device should be used for processing
    if torch.cuda.is_available():
        logging.info(f"Using GPU for inference.")
        device = torch.device('cuda')
    else:
        logging.info(f"Using CPU for inference.")
        device = torch.device('cpu')
        
    # List the full paths to all image tiles
    image_list = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if (filename.endswith("tif") or filename.endswith("tiff"))]
    if len(image_list) == 0:
        raise OSError(f"No image tiles (tif or tiff) were found in: {input_dir}")
    
    logging.info(f"No. of inference samples: {len(image_list)}")

    # Create a dataset of samples
    dataset = IMFDataset(image_list=image_list,
                        device=device)
    
    # Batch each image separately and don't shuffle
    batches = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialise the DeepLabV3 model with a ResNet50 backbone
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = torch.nn.LazyConv2d(2, 1)
    model.aux_classifier[4] = torch.nn.LazyConv2d(2, 1)
    
    # Push the model to the device
    model.to(device)
    
    # Load model checkpoint (see https://doi.org/10.5281/zenodo.15101175)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Define the output spatial reference for the detections
    output_srs = osr.SpatialReference()
    output_srs.ImportFromProj4("+proj=longlat +a=1737400 +b=1737400 +no_defs")
    
    # Create shapefile layers for storing detections
    output_path = os.path.join(output_dir, "detections.shp")
    output_ds, output_layer = create_shp_layer(output_path=output_path,
                                            output_srs=output_srs)
    
    logging.info(f"Beginning inference:")
    
    # Start a timer
    begin = time.time()

    # Infer the model on the tiled images
    inference_loop(model=model,
                batches=batches, 
                output_layer=output_layer, 
                image_list=image_list)
    
    # Close the output layer
    output_ds = output_layer = None
    
    # End the timer
    end = time.time()
    
    logging.info(f"Inference complete. Took approx. {(end - begin) / 60:0.2f} min(s).")
    
if __name__ == "__main__":
    main(input_dir=ARGS.inputdir,
        output_dir=ARGS.outputdir,
        model_path=ARGS.modelpath)