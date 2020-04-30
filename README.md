# 3D object reconstruction from a single 2D image: performance of two novel frameworks based on lightweight CNN architectures and free-form deformation of meshes



## Abstract
In computer vision, object reconstruction is the task of inferring the 3D shape of an object based on a single or multiple 2D images. For such purpose, most common frameworks use voxel grids and point clouds. However, both of these approaches have strong limitations. On one hand, the computational cost of using voxels grows cubically as the resolution of the voxels increases. Therefore, 3D object reconstructions are usually set to low resolution. On the other hand, point clouds are unstructured in nature and the proper definition of surfaces and contours is complex. In this study, 3D object reconstruction is carried out applying free-form deformations on pre-existent 3D meshes, through two basic learning processes: template selection and template deformation. From this approach, it is possible to generate high-quality 3D object reconstructions with a lower computational cost. Concretely, two novel lightweight CNNs models are developed and tested: a multi-target learner (Model A) and depth information learner (Model B). According to the results, the performance of the multi-target learner regarding the template selection was around three times better (lower error) than in the baseline architecture, which improved the quality of the 3D reconstructions, whereas the depth-information learner showed promising results in the reconstruction of objects with complex geometry. The inherent issue of using chamfer distance as a loss measure is also examined.

```
@inproceedings{10.1117/12.2557947,
author = {Saurabh Pradhan and Kiran Madhusudhanan and Leandro Munoz-Giraldo and MohiUddin Faruq and Hadi Jomaa},
title = {{3D object reconstruction from a single 2D image: performance of two novel frameworks based on lightweight CNN architectures and free-form deformation of meshes}},
volume = {11373},
booktitle = {Eleventh International Conference on Graphics and Image Processing (ICGIP 2019)},
editor = {Zhigeng Pan and Xun Wang},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {705 -- 715},
keywords = {3d reconstruction, free form deformation, 3d mesh, chamfer loss},
year = {2020},
doi = {10.1117/12.2557947},
URL = {https://doi.org/10.1117/12.2557947}
}
```

