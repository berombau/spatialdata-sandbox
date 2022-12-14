# %%
# setup:
# 1) mamba
#!pip install git+https://github.com/scverse/spatialdata@temp/spacehack2022
#!pip install git+https://github.com/scverse/spatialdata-io
#!pip install git+https://github.com/scverse/napari-spatialdata@spatialdata

# %%
import os
import tifffile

# to fix paths in Luca's machine:
# os.chdir('scratch/userfolders/lucamarconato/liver')

# geopandas hack
os.environ["USE_PYGEOS"] = "0"
import geopandas

import spatialdata as sd
import numpy as np
import pyarrow as pa
import dask_image.imread
import shutil

import os
from pathlib import Path
import logging
from spatialdata._core.coordinate_system import CoordinateSystem

logging.basicConfig(level=logging.INFO)


# %%

# check can find the data
data_path = Path(os.environ.get("DATA_ROOT", "data/"))
output_path = Path(os.environ.get("OUTPUT_ROOT", data_path))
# %%
# extract the data
# os.makedirs("data", exist_ok=True)
# os.system(f"tar -xvf {f}ResolveData_HCA.tar -C data")


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

    t = sd.PointsModel.parse(
        coords=xyz,
        annotations=gene,
        transforms=sd.Scale(
                # xyz
                scale=[1, 1, 1],
                # output_coordinate_system=CoordinateSystem.from_dict(
                #     {
                #         "name": "micrometers",
                #         "axes": [
                #             {"name": "x", "type": "space", "unit": "micrometer"},
                #             {"name": "y", "type": "space", "unit": "micrometer"},
                #             {"name": "z", "type": "space", "unit": "micrometer"},
                #         ]
                #     }
                # )
            )
        )
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
            element = sd.Image2DModel.parse(im, dims=("c", "y", "x"), multiscale_factors=[2, 4, 8, 16], name=name,
                transform = sd.Scale(
                    # cyx
                    scale=[1, 1, 1],
                    # TODO: get location for coordinate system `assert C not in _get_axes_names(mapper_input_coordinate_system)`
                    # output_coordinate_system=CoordinateSystem.from_dict(
                    #     {
                    #         "name": "micrometers",
                    #         "axes": [
                    #             {"name": "c", "type": "space", "unit": "micrometer"},
                    #             {"name": "y", "type": "space", "unit": "micrometer"},
                    #             {"name": "x", "type": "space", "unit": "micrometer"},
                    #         ]
                    #     }
                    # ),
                )
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
from napari_spatialdata import Interactive
sdata = sd.SpatialData.read(str(output_path / "ResolveHuman.zarr"))
sdata

# %%
sdata.images['20272_slide1_A1-1_DAPI']

# %%
Interactive(sdata)

