
Two trained Cellpose2 models suitable for segmentation _Xenopus.laevis_ embryos are provided:

The performance of cell segmentation is influenced by the intrinsic histological characteristics of the tissue sections and the quality of staining. Based on experimental results, it is recommended that users apply both models and select the one with better performance. <br>

The parameter of the Cellpose2, particularly the size diameter, can be customized.<br>

Our models can also serve as a starting point for your customized CellPose2 model.


**Model1_CP_20240615_165943_CZ** 

Model1 is trained from one section each from a stage 12.5 and a stage 22 _X.laevis_ embryos with good quality of Cellbound3 staining.<br> 
This model 1 is validated for  _X.laevis_ embryos from late gastrula stage 12.5 to tailbud stage 22.<br>

_The default Cellpose2 parameters:_<br>
size diameter: 164.72<br>
flow_threshold: 0.4<br>
cellprob_threshold: 0<br>

_Parameter in VPT_: --tile-overlap 200<br>


**Model2_CP_20250711_153352_SD**

Model2 is trained from two sections per stage from Blastula (Stage 9) and Gastrula (Stages 10.5, 11 and 12) _X.laevis_ embryos with weak Cellbound3 staining.<br>
This model 2 is valiadted for _X.laevis_ embryos from late blastula stage 9 to gastrula stage 12.<br>

_The default Cellpose2 parameters:_<br>
size diameter: 267.62<br>
flow_threshold: 0.95<br>
cellprob_threshold: 0<br>

_Parameter in VPT_: --tile-overlap 700

