# GANForest


1 python ForestType.py --input data_CNA.csv.gz --cluster 6 --feature 2000 --output select_feature.csv

2 python GAN.py

3 python generate_GAN_data.py

4 python RF_Gridsearch.py --input select_feature.csv --quicktrain True --output1 feature_important.csv --output2 fin_result.csv

* Due to upload size limitation, we did not upload mRNA dataset on github, you can download it from the METABRIC database 
* or computational research platform on Code Ocean(DOI is 10.24433/CO.4942687.v1).
