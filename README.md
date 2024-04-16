# KmeansClusteringHadoop

## steps to run
- git clone https://github.com/big-data-europe/docker-hadoop
- cd docker-hadoop
- docker-compose up -d
- first go into folder KmeansClusteringHadoop
- docker cp kmeans.jar namenode:/
- docker cp data namenode:/
- docker exec -it namenode /bin/bash
- hdfs dfs -mkdir /user/root/kmeans
- hdfs dfs -mkdir /user/root/kmeans/input
- hdfs dfs -put data/* /user/root/input/kmeans/input/
- hadoop jar charcount.jar org.myorg.CharacterCount kmeans/input kmeans/output <k-value> <comma separated random k means> <no of iterations>
- hdfs dfs -cat /user/root/charcount/outputfinal/* 
