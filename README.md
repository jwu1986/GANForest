# GANForest


1 python ForestType.py --input data_expression_median.csv.gz --cluster 7 --feature 2000 --output select_feature.csv

2 python GAN.py --input select_feature.csv

3 python generate_GAN_data.py --input select_feature.csv --output data_label_new.csv

4 python RF_Gridsearch.py --input data_label_new.csv --quicktrain True --output1 feature_important.csv --output2 fin_result.csv

* Due to upload size limitation, we did not upload mRNA dataset on github, you can download it from the METABRIC database 
* or computational research platform on Code Ocean(DOI is 10.24433/CO.4942687.v1).
