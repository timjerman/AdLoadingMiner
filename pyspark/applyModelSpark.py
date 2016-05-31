# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:49:02 2016

@author: TimJ
"""

from pyspark import SQLContext
from pyspark import  SparkContext
import numpy as np
import pyspark.sql.functions as func
from  pyspark.sql import Row
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import StringIndexerModel

def applyModel(fileName, loadModelName, outlierPercentile = 100):

    sc = SparkContext( 'local', 'pyspark')
    sqlContext = SQLContext(sc)

    #########
    # load data
    #########

    data = sc.textFile(fileName)
    #extract header and remove it
    header = data.first()
    data = data.filter(lambda x:x !=header).cache()
    header = header.split('\t')
    #parse data
    data = data.map(lambda x : x.split('\t'))

    #########
    # prepare features
    #########

    df = sqlContext.createDataFrame(data, header)
    df = (df.withColumn("ADLOADINGTIME",func.regexp_replace('ADLOADINGTIME', 'null', '0').cast('float'))
         .withColumn("TIMESTAMP",func.regexp_replace('TIMESTAMP', 'null', '0').cast('int'))
         .withColumn("GEOIP_LAT",func.regexp_replace('GEOIP_LAT', 'null', '0').cast('int'))
          .withColumn("GEOIP_LNG",func.regexp_replace('GEOIP_LNG', 'null', '0').cast('int'))
          .withColumn("HOSTWINDOWHEIGHT",func.regexp_replace('HOSTWINDOWHEIGHT', 'null', '0').cast('int'))
          .withColumn("HOSTWINDOWWIDTH",func.regexp_replace('HOSTWINDOWWIDTH', 'null', '0').cast('int'))
          .withColumn("TOPMOSTREACHABLEWINDOWHEIGHT",func.regexp_replace('TOPMOSTREACHABLEWINDOWHEIGHT', 'null', '0').cast('int'))
          .withColumn("TOPMOSTREACHABLEWINDOWWIDTH",func.regexp_replace('TOPMOSTREACHABLEWINDOWWIDTH', 'null', '0').cast('int'))
         )
    thr = np.percentile(df.select("ADLOADINGTIME").rdd.collect(), outlierPercentile)
    df = df.filter(func.col('ADLOADINGTIME') < thr)
    df = df.withColumn("TOPMOSTREACHABLEWINDOWAREA", func.col("TOPMOSTREACHABLEWINDOWHEIGHT")*func.col("TOPMOSTREACHABLEWINDOWWIDTH"))
    df = df.withColumn("INTENDENTISACTUALDEVICETYPE", (func.col("ACTUALDEVICETYPE")==func.col("INTENDEDDEVICETYPE")).cast('int'))
    df = df.withColumn("COMBINEDID", 
            func.concat(
                func.col('ACCOUNTID'), 
                func.col('CAMPAIGNID'), 
                func.col('CREATIVEID'), 
                func.col('SDK')) )

    #df = df.withColumn("COMBINEDID", func.regexp_replace("COMBINEDID", '^$', 'NA'))

    df = df.withColumn("COMBINEDEXTERNALID", 
            func.concat( 
                func.regexp_replace('EXTERNALADSERVER', 'null', ''), 
                func.regexp_replace('EXTERNALPLACEMENTID', 'null', ''), 
                func.regexp_replace('EXTERNALSITEID', 'null', ''), 
                func.regexp_replace('EXTERNALSUPPLIERID', 'null', '') ))

    #df = df.withColumn("COMBINEDEXTERNALID", func.regexp_replace("COMBINEDEXTERNALID", '^$', 'NA'))

    df = df.withColumn("PLATFORMCOMBINED", 
            func.concat( 
                func.regexp_replace('PLATFORM', 'null', ''), 
                func.regexp_replace('PLATFORMVERSION', 'null', '') ))

    #df = df.withColumn("PLATFORMCOMBINED", func.regexp_replace("PLATFORMCOMBINED", '^$', 'NA'))

    df = df.withColumn("UA_OSCOMB", 
            func.concat( 
                func.regexp_replace('UA_OS', 'null', ''), 
                func.regexp_replace('UA_OSVERSION', 'null', '') ))

    #df = df.withColumn("UA_OSCOMB", func.regexp_replace("UA_OSCOMB", '^$', 'NA'))
    df = df.withColumn("FILESJSON_SIZE", 
                func.regexp_replace('FILESJSON', '[^,\d]', '') )

    df = df.withColumn("FILESJSON_SIZE", 
                func.regexp_replace('FILESJSON_SIZE', '^,', '') )

    df = df.withColumn("FILESJSON_SIZE", 
                func.regexp_replace('FILESJSON_SIZE', ',,', ',') )

    udf = func.udf(lambda x: int(np.fromstring(x,dtype=int, sep=',').sum()), IntegerType())
    df = df.withColumn("FILESJSON_SIZE", udf("FILESJSON_SIZE"))

    print('Loaded and prapared %d entries' % df.count())

    #########
    # keep only needed features
    #########   

    features = ['ADLOADINGTIME',
     'PLACEMENTID',
     'TIMESTAMP',
     'CREATIVETYPE',
     'UA_HARDWARETYPE',
     'UA_VENDOR',
     'UA_MODEL',
     'UA_BROWSER',
     'UA_BROWSERVERSION',
     'FILESJSON',
     'ERRORSJSON',
     'TOPMOSTREACHABLEWINDOWAREA',
     'FILESJSON_SIZE',
     'COMBINEDID',
     'COMBINEDEXTERNALID',
     'PLATFORMCOMBINED',
     'UA_OSCOMB',
     'SDK',
     'EXTERNALADSERVER'
       ]

    df = df.select(features)

    #########
    # Convert categorical features to numerical
    #########   


    featuresCat = [
     'PLACEMENTID',
     'CREATIVETYPE',
     'UA_HARDWARETYPE',
     'UA_VENDOR',
     'UA_MODEL',
     'UA_BROWSER',
     'UA_BROWSERVERSION',
     'FILESJSON',
     'ERRORSJSON',
     'COMBINEDID',
     'COMBINEDEXTERNALID',
     'PLATFORMCOMBINED',
     'UA_OSCOMB',
     'SDK',
     'EXTERNALADSERVER'
       ]

    for i in range(len(featuresCat)):

        indexer = StringIndexer(inputCol=featuresCat[i], outputCol='_'+featuresCat[i]).setHandleInvalid("skip").fit(df)
        df = indexer.transform(df).drop(featuresCat[i])
        writer = indexer._call_java("write")
        writer.overwrite().save("indexer_" + featuresCat[i])    

    featuresCat = [ '_' + featuresCat[i] for i in range(len(featuresCat))]    

    features = featuresCat[:]
    features.append('TIMESTAMP')    
    features.append('FILESJSON_SIZE')
    features.append('TOPMOSTREACHABLEWINDOWAREA')


    #########
    # Assemble features
    #########   


    assembler = VectorAssembler(
        inputCols=features,
        outputCol="features")

    df = assembler.transform(df)

    #########
    # Convert to labeled point
    #########   


    lp = (df.select(func.col("ADLOADINGTIME").alias("label"), func.col("features"))
      .map(lambda row: LabeledPoint(row.label, row.features)))
    lp.cache()


    #########
    # Load trained model
    #########
    
    model = RandomForestModel.load(sc, loadModelName)
    
    print('Model loaded!')
    
    predictions = model.predict(lp.map(lambda x: x.features)).collect()
    
    return predictions
    
    
if __name__ == '__main__':    
    
    fileName = 'E:/celtraMiningFrik/ccdm_medium.tsv'
    loadModelName = 'E:/celtraMiningFrik/sparkRFmodel'
    solutionFileName = 'E:/celtraMiningFrik/solutionSparkRFReg.tsv'
    
    predictions = applyModel(fileName, loadModelName)
    
    import csv
    with open(solutionFileName, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for r in predictions:
            writer.writerow([r])    