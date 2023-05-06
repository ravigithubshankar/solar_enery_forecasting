from forecast.py import *
import modelbit
solar_energy_deployment=modelbit.login()
#deployment with necessary packages
solar_energy_deployment.deploy(knn,python_packages=["scikit-learn==1.2..2"])
"""Deploying knn
Heads up!
You chose scikit-learn==1.2..2 for your production environment, but you have scikit-learn==1.2.2 locally.
To match your environment to production, run:
!pip install scikit-learn==1.2..2
To match production to your local environment, include this line in your deployment:
mb.deploy(my_deploy_function, python_packages=["scikit-learn==1.2.2"])
There is one more inconsistency. View it.
You chose scikit-learn==1.2..2 for your production environment, but you have scikit-learn==1.2.2 locally.
To match your environment to production, run:
!pip install scikit-learn==1.2..2
To match production to your local environment, include this line in your deployment:
mb.deploy(my_deploy_function, python_packages=["scikit-learn==1.2.2"])
You chose Python 3.9 for your production environment, but you have Python 3.10 locally.
To match production to your local environment, include this line in your deployment:
mb.deploy(my_deploy_function, python_version="3.10")
To match your local environment to production, consider installing Python 3.9 locally.
Uploading dependencies...
Uploading 'knn': 4.88MB [00:03, 1.46MB/s]                                       
Success!
Deployment knn will be ready in a couple minutes.
"""
#xgbregressormodel
solar_energy_deployment.deploy(xgb_regressor)
"""Deploying xgb_regressor
Heads up!
You chose scikit-learn==1.2.1 for your production environment, but you have scikit-learn==1.2.2 locally.
To match your environment to production, run:
!pip install scikit-learn==1.2.1
To match production to your local environment, include this line in your deployment:
mb.deploy(my_deploy_function, python_packages=["scikit-learn==1.2.2"])
There are 3 more inconsistencies. View all.
You chose scikit-learn==1.2.1 for your production environment, but you have scikit-learn==1.2.2 locally.
To match your environment to production, run:
!pip install scikit-learn==1.2.1
To match production to your local environment, include this line in your deployment:
mb.deploy(my_deploy_function, python_packages=["scikit-learn==1.2.2"])
You chose pandas==1.5.3 for your production environment, but you have pandas==1.3.5 locally.
To match your environment to production, run:
!pip install pandas==1.5.3
To match production to your local environment, include this line in your deployment:
mb.deploy(my_deploy_function, python_packages=["pandas==1.3.5"])
You chose xgboost==1.7.4 for your production environment, but you have xgboost==1.7.5 locally.
To match your environment to production, run:
!pip install xgboost==1.7.4
To match production to your local environment, include this line in your deployment:
mb.deploy(my_deploy_function, python_packages=["xgboost==1.7.5"])
You chose Python 3.9 for your production environment, but you have Python 3.10 locally.
To match production to your local environment, include this line in your deployment:
mb.deploy(my_deploy_function, python_version="3.10")
To match your local environment to production, consider installing Python 3.9 locally.
Uploading dependencies...
Success!
Deployment xgb_regressor will be ready in a couple minutes.
"""
#giving new samples and runing model again
def predict(a: 101.2, b: 222.34, c: 00.123, d: 123.23, e: 12345.23, f: 2000.34, g: 1234.45, h: 1122.56, i: float, j: float, k: 11234.100, l: 11002.34) -> float:
  return knn.predict([[a, b, c, d, e, f, g, h, i, j, k, l]])[0]
#predictions
  clearsky dhi    clearsky dni      clearsky ghi
-1.01842657303907	0.582411402576522	0.10852363072163
0.941864038652432	-1.04824946074613	2.53456156216685
0.548937458752015	0.286428168044837	1.1819038812496
-1.12457079360056	0.429654811338061	0.392920514082237
-0.579378173979199	-1.470136415667	-0.572897946448197
-0.255011584340265	0.144345193111564	0.874471537969756
0.349311046251627	-1.11925568137056	-0.00605243691674372
0.822951137477133	0.186040384476078	0.545043586723203
1.99158948342861	0.519138809660254	-0.80261420950339
0.464538670494523	-2.3436227217597	-0.276715415341811
-0.466357007006829	1.48826043155513	-0.296671843449791
0.587547156209216	0.610294271447043	0.500203168796813
1.91202737055772	1.04645054386991	0.321889095161961
0.544034460226425	-0.0144640176735389	-2.15099019215485
1.70336322251808	-0.547601848116578	0.205696263417081
-0.698260945360546	-1.20035811620042	0.547372590364806
-0.150954980851661	-1.9814739715834	-1.58616143656431
-0.448760136287188	0.926328286619993	-0.272267558391059
0.660696147481587	-1.06717227290584	-0.19374605425452
... only few displayed

