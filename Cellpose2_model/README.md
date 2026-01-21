
Two trained Cellpose2 models suitable for segmentation _Xenopus.laevis_ embryos are provided:

**Model1_CP_20240615_165943_CZ** 

Model1 is trained from sections from a stage 12.5 and a stage 22 _X.laevis_ embryos with good quality of Cellbound3 staining.<br> 
This model 1 is validated for embryos from late gastrula stage 12.5 to tailbud stage 22.<br>
The default size diameter for model1 is 164.72<br>  
flow_threshold: 0.4<br>
cellprob_threshold: 0<br>
Parameter in VPT: --tile-overlap 700<br>

**Model2_CP_20250711_153352_SD**

Model2 is trained from two sections per stage from Blastula (Stage 9) and Gastrula (Stages 10.5, 11 and 12) _X.laevis_ embryos with weak Cellbound3 staining.<br>
This model 2 is valiadted for embryos from late blastula stage 9 to gastrula stage 12.<br>
The default size diameter for model1 is 267.62<br>  
flow_threshold: 0.95<br>
cellprob_threshold: 0<br>
Parameter in VPT: --tile-overlap 200

The performance of the cell segmentation is influenced by the instinct histology of the sections as well as the staining quality. According to the experiment, we recommend users to apply both models and then choose the one with better performance. <br>
The parameter of the cellpose2, especially the size diameter, could be adjusted according to the different size of cells.<br>
Our models could also be used as a start point for your own customized cellpose2 model.