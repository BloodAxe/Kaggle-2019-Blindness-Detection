Please cite the following paper if you use this data in your research/study:

Krause, J. et al. Grader variability and the importance of reference standards
for evaluating machine learning models for diabetic retinopathy.
Ophthalmology (2018). doi:10.1016/j.ophtha.2018.01.034

About the data:

The csv file contains adjudicated 5 point ICDR grades and Referable DME grades
for the Messidor 2 dataset [1]. Each row of the csv corresponds to a single
image, with 4 columns labeled as:

image_id, adjudicated_dr_grade, adjudicated_dme, adjudicated_gradable

Column descriptions:

image_id: Filename as provided in the Messidor 2 dataset.

adjudicated_dr_grade: 5 point ICDR grade
  0=None
  1=Mild DR
  2=Moderate DR
  3=Severe DR
  4=PDR

adjudicated_dme: Referable DME defined by Hard exudates within 1DD
  0=No Referable DME
  1=Referable DME

adjudicated_gradable: Image quality grade.
  0=Ungradable, no DR or DME grade is provided
  1=Gradable, both DR and DME were graded.
Please note that adjudicated_dr_grade and adjudicated_dme columns will be empty
for images where adjudicated_gradable=0.

The grading was done according to the adjudication protocol described in
Krause et al. [2], which demonstrated improvements in data and model quality
over Gulshan et al. [3] using adjudicated data.

References:

[1] Decencière  E, Etienne  D, Xiwei  Z,  et al.  Feedback on a publicly distributed image database: the Messidor database.  Image Anal Stereol. 2014;33(3):231-234. doi:10.5566/ias.1155

[2] Krause, J. et al. Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy. Ophthalmology (2018). doi:10.1016/j.ophtha.2018.01.034

[3] Gulshan, V. et al. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. JAMA 316, 2402–2410 (2016)
