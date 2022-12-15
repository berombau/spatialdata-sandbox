
import os
from pathlib import Path
import logging

import spatialdata as sd
import numpy as np
import pyarrow as pa
import dask_image.imread
import shutil

from spatialdata._core.coordinate_system import CoordinateSystem
from spatialdata._core.core_utils import get_default_coordinate_system

logging.basicConfig(level=logging.INFO)


# %%

# check can find the data
data_path = Path(os.environ.get("DATA_ROOT", "data/"))
output_path = Path(os.environ.get("OUTPUT_ROOT", data_path))


# %%
def create_points_element(path: str) -> pa.Table:
    from pyarrow.csv import read_csv, ReadOptions, ParseOptions

    table = read_csv(
        path,
        read_options=ReadOptions(autogenerate_column_names=True),
        parse_options=ParseOptions(delimiter="\t"),
    )

    table = table.rename_columns(["x", "y", "z", "gene", ""]).drop([""])

    xyz = table.to_pandas()[["x", "y", "z"]].to_numpy().astype(np.float32)
    gene = pa.Table.from_pydict({"gene": table.column("gene").dictionary_encode()})

    image_translation: ndarray[Any, dtype[floating[_32Bit]]] = np.array([0, 0, 0], dtype=np.float32)
    # [resolution in x, resolution in y, resolution in z]
    image_scale_factors = np.array([.138, .138, .300], dtype=np.float32)
    translation = sd.Translation(translation=image_translation)
    scale = sd.Scale(scale=image_scale_factors)
    xyz_cs = get_default_coordinate_system(('x', 'y', 'z')) 
    composed: sd.Sequence = sd.Sequence([
        scale, 
        translation],
        # TODO: remove redundant code
        input_coordinate_system=xyz_cs,
        output_coordinate_system=xyz_cs,
    )
    # output_coordinate_system=CoordinateSystem.from_dict(
    #     {
    #         "name": "micrometers",
    #         "axes": [
    #             {"name": "x", "type": "space", "unit": "micrometer"},
    #             {"name": "y", "type": "space", "unit": "micrometer"},
    #             {"name": "z", "type": "space", "unit": "micrometer"},
    #         ]
    #     }
    # ),

    # patch units in the coordinate system
    # TODO: remove dummy patch
    # TODO: suppport > 2 dims
    # dummy = sd.PointsModel.parse(np.zeros(shape=(2, 2)), transforms=composed)
    # correct_transform = sd.get_transform(dummy)
    # for axis in correct_transform.output_coordinate_system._axes:
    #     axis.unit = "micrometer"
    
    # add points with a unit coordinate system
    t = sd.PointsModel.parse(
        coords=xyz,
        annotations=gene,
        transform=composed
    )
    # assert sd.get_transform(t).to_affine().affine[2, 3] == 2
    return t

organisms = ["ResolveHuman", "ResolveMouse"]
for o in organisms:
    points = {}
    images = {}
    for filename in os.listdir(data_path / o):
        if filename.endswith(".txt"):
            element = create_points_element(str(data_path / o / filename))
            points[filename.replace('.txt', '')] = element
            print(filename, f'converted to Points element ({type(element)}')
        elif filename.endswith(".tiff"):
            im = dask_image.imread.imread(data_path / o / filename)
            name = filename.replace(".tiff", "")
            # patch units in the coordinate system
            image_translation = np.array([0, 0, 0], dtype=np.float32)
            # [resolution in c, resolution in y, resolution in x]
            image_scale_factors = np.array([1, .138, .138], dtype=np.float32)
            translation = sd.Translation(translation=image_translation)
            scale = sd.Scale(scale=image_scale_factors)
            composed = sd.Sequence([
                scale, 
                translation,
            ]
            # , output_coordinate_system=CoordinateSystem.from_dict(
            # {
            #     "name": "micrometers",
            #     "axes": [
            #         {"name": "c", "type": "channel"},
            #         {"name": "y", "type": "space", "unit": "micrometer"},
            #         {"name": "x", "type": "space", "unit": "micrometer"},
            #     ]
            # })
            )
            multiscale_factors = [2, 4, 8, 16]

            # TODO: remove dummy patch
            dummy = sd.Image2DModel.parse(np.zeros(shape=(2, 2, 2)), dims=("c", "y", "x"), multiscale_factors=multiscale_factors, transform=composed)
            correct_transform = sd.get_transform(dummy)
            for axis in correct_transform.output_coordinate_system._axes:
                if axis.name != "c":
                    axis.unit = "micrometer"
            
            # add element with a unit coordinate system
            element = sd.Image2DModel.parse(im, dims=("c", "y", "x"), multiscale_factors=multiscale_factors, name=name,
                transform = correct_transform
            )
            
            images[name] = element
            print(filename, f'converted to Image element ({type(element)})')
    sdata = sd.SpatialData(points=points, images=images)
    zarr_path = str(output_path / f'{o}.zarr')
    if os.path.isdir(zarr_path):
        shutil.rmtree(zarr_path)
    print(f"saving SpatialData object to {zarr_path}")
    sdata.write(str(zarr_path))
    print('done')

# %%
# interactive visualization with napari (doesn't work in JupyterLab)
# from napari_spatialdata import Interactive
# sdata = sd.SpatialData.read(str(output_path / "ResolveHuman.zarr"))
# sdata

# # %%
# sdata.images['20272_slide1_A1-1_DAPI']

# # %%
# Interactive(sdata)

